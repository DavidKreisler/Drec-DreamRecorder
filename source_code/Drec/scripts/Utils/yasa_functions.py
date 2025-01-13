import mne.io
import numpy as np
import pandas as pd
import yasa
from scipy.signal import welch
import warnings
import itertools
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from scripts.Utils.Logger import Logger

warnings.filterwarnings('ignore', message='Trying to unpickle estimator LabelEncoder from version 0.24.2 when *')

class YasaClassifier:

    @staticmethod
    def get_raw_eeg_from_edf(filepath: str) -> mne.io.Raw:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose='error')
        raw.filter(0.1, 40, verbose='error')
        return raw

    @staticmethod
    def get_sleep_hypno(raw: mne.io.Raw, channel: str, male: bool = True, age: int = 45):
        """
        :param raw: the signal loaded from get_raw_eeg_from_edf()
        :param channel: the name of the channel
        :param male: optional: metadata
        :param age: optional: metadata
        :param scorer: optional: needed to perform EpochByEpochAgreement
        :return: yasa.Hypnogram
        """
        data = raw.copy()
        data.pick(channel)
        sls = yasa.SleepStaging(data, eeg_name=channel, metadata=dict(age=age, male=male))
        pred = sls.predict()
        return pred

    @staticmethod
    def get_sleep_hypno_probs(raw: mne.io.Raw, channel: str, male: bool = True, age: int = 45) -> np.array:
        """
        :param raw: the signal loaded from get_raw_eeg_from_edf()
        :param channel: the name of the channel
        :param male: optional: metadata
        :param age: optional: metadata
        :param scorer: optional: needed to perform EpochByEpochAgreement
        :return: yasa.Hypnogram
        """
        data = raw.copy()
        data.pick(channel)
        sls = yasa.SleepStaging(data, eeg_name=channel, metadata=dict(age=age, male=male))
        pred = sls.predict_proba()
        return pred

    @staticmethod
    def get_bandpower(mne_array, channels: list, hypno: np.array = None, window_size: int = 4, sf: int = 256):
        bandpower = yasa.bandpower(mne_array, sf, channels, hypno, window_size)
        return bandpower

    @staticmethod
    def get_bandpower_per_epoch(mne_array, window_size: int = 5, sf: int = 256, epoch_len: int = 30):
        # get epochs
        data = mne_array.copy().get_data()
        _, epochs = yasa.sliding_window(data, sf, window=epoch_len)

        # calculate psd
        win = int(window_size * sf)
        freqs, psd = welch(epochs, sf, nperseg=win, axis=-1)

        # get bandpower per epoch
        bandpower = yasa.bandpower_from_psd_ndarray(psd, freqs)
        bandpower_last_epoch = bandpower[:, -1, :]  # [band, epoch, channel]

        return bandpower, bandpower_last_epoch

    @staticmethod
    def get_preds_per_epoch(mne_array, channel_name: str = 'eegl'):
        data = mne_array.copy()
        preds = YasaClassifier.get_sleep_hypno(data, channel_name)
        return preds

    @staticmethod
    def get_preds_per_sample(mne_array, predictions, channels: list, epoch_len_sec: int = 30, sf: int = 256):
        data = mne_array.copy().pick(channels)
        preds_per_sample = yasa.hypno_str_to_int(predictions)
        preds_per_sample = yasa.hypno_upsample_to_data(hypno=preds_per_sample, sf_hypno=(1 / epoch_len_sec),
                                                       data=data.get_data(), sf_data=sf)
        return preds_per_sample

    @staticmethod
    def get_eyes(mne_array, channels: list, predictions=None, sf: int = 256):
        data = mne_array.copy().pick(channels)
        hypno = None
        if predictions is not None and 'R' in predictions:
            hypno = YasaClassifier.get_preds_per_sample(mne_array=data, predictions=predictions, channels=channels)
        loc, roc = data.get_data()
        rem = yasa.rem_detect(loc, roc, sf, hypno=hypno, include=4, amplitude=(50, 325),
                              duration=(0.3, 1.2),
                              relative_prominence=0.8, freq_rem=(0.5, 5), remove_outliers=False, verbose='Error')
        return rem

    @staticmethod
    def get_scoring_metrics(mne_array, sample_rate: int = 256, channels=None):
        """
        param: mne_array
        param: sample_rate
        param: channels

        returns: a pd.DataFrame with the bandpowers as columns, aswell as a flag to specify if epoch is rem (by
                sleep staging and eyes
        """
        if channels is None:
            channels = ['eegr', 'eegl']

        # get sleep stages
        sleep_stage_predictions = YasaClassifier.get_preds_per_epoch(mne_array=mne_array)
        sleep_stage_rem = [1 if sleep_stage_predictions[i] == 'R' else 0 for i in
                           range(0, len(sleep_stage_predictions))]

        # get bandpower
        bandpower, _ = YasaClassifier.get_bandpower_per_epoch(mne_array)  # bandpwr has shape [band, epoch, channel]
        bandpower = bandpower.mean(axis=2).T  # now has the shape [epoch, band], so each 'row' is an epoch

        # get rem by eyes
        eyes = YasaClassifier.get_eyes(mne_array, channels, sleep_stage_predictions)

        # binarize results by eyes
        eyes_bin = list(np.zeros(len(sleep_stage_rem)))
        if eyes:  # eyes may be None if no rem events were found
            eyes_mask = eyes.get_mask()  # mask containing
            for bin_idx, sample_idx in enumerate(range(0, len(eyes_mask[0, 0:len(eyes_mask[0])]), 30 * sample_rate)):
                min_val = min([1, sum(eyes_mask[0][sample_idx:sample_idx+30*sample_rate])])
                eyes_bin[bin_idx] = min_val

        # combine into one df
        data = pd.DataFrame(bandpower, columns=['delta', 'theta', 'alpha', 'sigma', 'beta', 'gamma'])
        data['is_rem'] = [int(sum([sleep_stage_rem[i], eyes_bin[i]]) / 2) for i in range(0, len(sleep_stage_rem))]

        return data

    @staticmethod
    def get_epoch_by_epoch_agreement(targets: list, preds: list):
        agr = yasa.EpochByEpochAgreement(targets, preds)
        return agr

    @staticmethod
    def get_theta_delta_threshold(data: pd.DataFrame, column_name: str, n_clusters: int = 3):
        """
        data: pd.DataFrame that contains only the theta/delta ratio per epoch. SideEffect: data gets an additional
        column: cluster column_name: the name of the column within the dataframe that contains the theta/delta ratio
        """
        d = pd.DataFrame(data[column_name])
        scaler = StandardScaler()
        features = scaler.fit_transform(d)

        kmeans = KMeans(n_clusters=n_clusters)
        data['cluster'] = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_

        original_centers = scaler.inverse_transform(centers).flatten()
        original_centers.sort()
        optimum_thresh = (original_centers[-1] - original_centers[-2]) / 2

        return optimum_thresh, original_centers

    @staticmethod
    def find_optimal_rem_scoring_metrics(data: pd.DataFrame, rem_flag='is_rem', test_size=0.2, random_state=42):
        """
        Finds the optimal power bands and ratios to distinguish between REM and non-REM epochs
        and determines the optimal threshold for the best feature.

        Parameters:
        - data: pd.DataFrame
            A DataFrame containing power bands as columns and a REM flag column.
        - rem_flag: str
            The name of the column indicating REM (1) vs non-REM (0).
        - test_size: float
            Proportion of the dataset to include in the test split.
        - random_state: int
            Random seed for reproducibility.

        Returns:
        - dict
            A dictionary containing the best feature, its metrics, optimal threshold, and scores for all features.
        """
        # Separate features and target
        target = data[rem_flag]
        if len(target.unique()) == 1:
            Logger().log(f'only negative samples present when finding best scoring metrics. aborting.', 'debug')
            return None
        features = data.drop(columns=[rem_flag])
        band_names = features.columns

        # Generate all possible band ratios
        ratios = []
        for band1, band2 in itertools.combinations(band_names, 2):
            ratio_name = f"{band1}/{band2}"
            features[ratio_name] = features[band1] / features[band2]
            ratios.append(ratio_name)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size,
                                                            random_state=random_state)
        if len(y_train.unique()) <= 1 or len(y_test.unique()) <= 1:
            Logger().log(f'not enough positive samples for train_test_split! aborting', 'debug')
            return None

        # Evaluate each feature
        feature_scores = {}
        for feature in features.columns:
            # Train logistic regression using one feature
            model = LogisticRegression()
            model.fit(X_train[[feature]], y_train)
            y_pred_proba = model.predict_proba(X_test[[feature]])[:, 1]

            # Compute metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            thresholds = np.linspace(0, 1, 101)  # Test thresholds from 0 to 1
            best_threshold, best_metric = None, 0

            # Evaluate thresholds
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                acc = accuracy_score(y_test, y_pred)  # You can replace this with other metrics
                if acc > best_metric:  # Replace `acc` with your chosen metric
                    best_metric = acc
                    best_threshold = threshold

            feature_scores[feature] = {
                'AUC': auc,
                'Best Threshold': best_threshold,
                'Best Accuracy': best_metric
            }
        # Find the best feature
        best_feature = max(feature_scores, key=lambda x: feature_scores[x]['AUC'])
        best_metrics = feature_scores[best_feature]

        # Return results
        return {
            'Best Feature': best_feature,
            'Best Metrics': best_metrics,
            'All Feature Scores': feature_scores
        }


def simulate_scoring_in_live(raw: mne.io.Raw, channel_l: str, channel_r: str, sample_rate: int = 256,
                             epoch_len_in_sec: int = 30, start_sample: int = 0, end_sample: int = 0,
                             start_scoring_at_min: int = 5):
    """
    :param channel_r: the name of the right channel of the zMax Headband
    :param channel_l: the name of the left channel of the zMax Headband
    :param raw: a signal that contains the at least the l and r eeg channels specified in channels
    :param sample_rate:
    :param epoch_len_in_sec
    :param start_sample
    :param end_sample
    :param start_scoring_at_min
    :return: result Dataframe
    """

    signal = raw.copy()
    signal.pick([channel_l, channel_r])
    ch_l, ch_r = signal.get_data()

    if start_sample > len(ch_l):
        start_sample = len(ch_l)-1
    if start_sample < 0:
        start_sample = 0
    if end_sample <= start_sample:
        end_sample = len(ch_l)
    if end_sample > len(ch_l):
        end_sample = len(ch_l)
    if start_scoring_at_min < 5:
        start_scoring_at_min = 5
    if start_scoring_at_min > len(ch_l)/sample_rate*60:
        start_scoring_at_min = len(ch_l)/sample_rate*60

    first_scoring = True

    pwrband_feature = None
    pwrband_thresh = None

    live_scorings = []

    for idx in tqdm(range(start_sample, end_sample, sample_rate*epoch_len_in_sec)):
        if idx < start_scoring_at_min*sample_rate*60:
            live_scorings.append([0, 0])
            continue
        # -----------------------
        # the signal arrives
        # -----------------------
        loc = ch_l[0:idx]
        roc = ch_r[0:idx]

        info = mne.create_info(ch_names=['eegl', 'eegr'], sfreq=256, ch_types='eeg', verbose='Error')
        mne_array = mne.io.RawArray([roc.copy(), loc.copy()], info, verbose='Error')
        scoring_metrics = YasaClassifier.get_scoring_metrics(mne_array=mne_array, channels=['eegl', 'eegr'])
        # current sample is scoring_metrics[-1]

        # -----------------------
        # get optimal metrics
        # -----------------------
        if first_scoring:
            optimal_metric = YasaClassifier.find_optimal_rem_scoring_metrics(data=scoring_metrics, rem_flag='is_rem')
            if optimal_metric:
                pwrband_feature = optimal_metric['Best Feature']
                pwrband_thresh = optimal_metric['Best Metrics']['Best Threshold']
                first_scoring = False
                Logger().log(f'metrics for best pwrband: {pwrband_feature}, {pwrband_thresh}. {optimal_metric["Best Metrics"]["Best Accuracy"]}')

        rem_by_pwrband = list(np.zeros(len(scoring_metrics), dtype=int))
        if pwrband_feature and pwrband_thresh:
            if '/' in pwrband_feature:
                feat1, feat2 = pwrband_feature.split('/')
                rem_by_pwrband = (scoring_metrics[feat1] / scoring_metrics[feat2]) > pwrband_thresh
            else:
                rem_by_pwrband = (scoring_metrics[pwrband_feature]) > pwrband_thresh

        scoring_metrics['rem_by_pwrband'] = [int(x) for x in rem_by_pwrband]

        live_scorings.append([list(scoring_metrics['is_rem'])[-1], list(scoring_metrics['rem_by_pwrband'])[-1]])

    data = pd.DataFrame.from_records(np.array(live_scorings), columns=['is_rem', 'rem_by_pwrband'])
    return data


if __name__ == '__main__':
    pass

