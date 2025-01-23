import traceback

import mne.io
import numpy as np
import pandas as pd
import yasa
from scipy.signal import welch
import warnings
import itertools
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve

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
    def get_rem_bin_per_epoch(mne_array, sample_rate: int = 256, channels=None):
        if channels is None:
            channels = ['eegr', 'eegl']

        # get scoring based on sleep stage predictions
        sleep_stage_predictions = YasaClassifier.get_preds_per_epoch(mne_array=mne_array)
        sleep_stage_rem = [1 if sleep_stage_predictions[i] == 'R' else 0 for i in
                           range(0, len(sleep_stage_predictions))]

        # get scoring based on eyes
        eyes = YasaClassifier.get_eyes(mne_array, channels, sleep_stage_predictions)

        # binarize results by eyes
        eyes_bin = list(np.zeros(len(sleep_stage_rem)))
        if eyes:  # eyes may be None if no rem events were found
            try:
                eyes_mask = eyes.get_mask()  # mask containing
                for bin_idx in range(0, int(len(eyes_mask[0]) / (30 * sample_rate))):
                    sample_idx = bin_idx * 30 * sample_rate
                    min_val = min([1, sum(eyes_mask[0][sample_idx:sample_idx + 30 * sample_rate])])
                    eyes_bin[bin_idx] = min_val

            except Exception as e:
                Logger().log(f'error happened at eyes.get_mask or afterwards. error is {traceback.format_exc()}',
                             'ERROR')
                Logger().log(f'eyes is: {eyes}', 'DEBUG')

        # get scoring based on frequency bandpower
        # this doesnt work because no metric within the bandpower yields performance that is good enough (F1 score above 0.6 at least)
        #bandpower, _ = YasaClassifier.get_bandpower_per_epoch(mne_array)  # bandpwr has shape [band, epoch, channel]
        #bandpower = bandpower.mean(axis=2).T  # now has the shape [epoch, band], so each 'row' is an epoch

        data = pd.DataFrame()
        data['rem_by_prediction'] = sleep_stage_rem
        data['rem_by_eyes'] = eyes_bin

        data['rem_by_all'] = [int(sum([sleep_stage_rem[i], eyes_bin[i]]) / 2) for i in range(0, len(sleep_stage_rem))]

        return data

    @staticmethod
    def get_power_bands_and_ground_truth_per_epoch(mne_array, sample_rate: int = 256, channels=None):
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
            try:
                eyes_mask = eyes.get_mask()  # mask containing
                for bin_idx in range(0, int(len(eyes_mask[0]) / (30 * sample_rate))):
                    sample_idx = bin_idx * 30 * sample_rate
                    min_val = min([1, sum(eyes_mask[0][sample_idx:sample_idx + 30 * sample_rate])])
                    eyes_bin[bin_idx] = min_val

            except Exception as e:
                Logger().log(f'error happened at eyes.get_mask or afterwards. error is {traceback.format_exc()}', 'ERROR')
                Logger().log(f'eyes is: {eyes}', 'DEBUG')
                # eyes_bin = list(np.zeros(len(sleep_stage_rem)))

        # combine into one df
        data = pd.DataFrame(bandpower, columns=['delta', 'theta', 'alpha', 'sigma', 'beta', 'gamma'])
        data['is_rem'] = [int(sum([sleep_stage_rem[i], eyes_bin[i]]) / 2) for i in range(0, len(sleep_stage_rem))]

        return data

    @staticmethod
    def get_epoch_by_epoch_agreement(targets: list, preds: list):
        agr = yasa.EpochByEpochAgreement(targets, preds)
        return agr

    @staticmethod
    def find_best_scoring_metrics_for_powerbands(data: pd.DataFrame, target_column: str = 'is_rem'):
        """
        data has to contain each frequency band as columns and one column with the name specified in 'target_column'.
        best to use the dataframe provided by self.get_power_bands_and_ground_truth_per_epoch()
        """
        features = data.copy()
        new_columns = {}
        for band1, band2 in itertools.combinations(features.columns, 2):
            if band1 == target_column or band2 == target_column:
                continue
            ratio_name = f"{band1}/{band2}"
            new_columns[ratio_name] = features[band1] / features[band2]

        features = pd.concat([features, pd.DataFrame(new_columns, index=features.index)], axis=1)

        best_columns = []
        best_column_metrics = dict()
        ### best column ###
        # by correlation
        correlations = features.corr()[target_column].drop(target_column).abs()
        best_column_corr = correlations.idxmax()
        best_columns.append(best_column_corr)
        print(
            f"The best column by correlation is: {best_column_corr} with correlation: {correlations[best_column_corr]}")
        best_column_metrics['correlation'] = {f'{best_column_corr}': correlations[best_column_corr]}

        # by t-test
        best_pvalue = float('inf')
        best_column_ttest = None
        for column in features.columns:
            if column != target_column:
                group1 = features.loc[features[target_column] == 0][column]
                group2 = features.loc[features[target_column] == 1][column]

                _, pvalue = ttest_ind(group1, group2, equal_var=False)
                if pvalue < best_pvalue:
                    best_pvalue = pvalue
                    best_column_ttest = column

        best_columns.append(best_column_ttest)
        print(f"The best column by t-test is: {best_column_ttest} with p-value: {best_pvalue}")
        best_column_metrics['t-test'] = {f'{best_column_ttest}': best_pvalue}

        # by random forest
        # Prepare features and labels
        X = features.drop(columns=[target_column])
        y = features[target_column]

        # Train a Random Forest Classifier
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)

        # Extract feature importance
        importances = clf.feature_importances_
        best_column_index = importances.argmax()
        best_column_rf = X.columns[best_column_index]

        best_columns.append(best_column_rf)
        print(
            f"The best column by random forest is: {best_column_rf} with importance score: {importances[best_column_index]}")
        best_column_metrics['random forest'] = {f'{best_column_rf}': importances[best_column_index]}

        # by mutual info classifyer
        X = features.drop(columns=[target_column])
        y = features[target_column]

        # Compute mutual information scores
        mi_scores = mutual_info_classif(X, y)
        mi_scores_series = pd.Series(mi_scores, index=X.columns)

        best_column_mutcl = mi_scores_series.idxmax()
        best_columns.append(best_column_mutcl)
        print(
            f"The best column by mutual information score is: {best_column_mutcl} with mutual information score: {mi_scores_series[best_column_mutcl]}")
        best_column_metrics['mif'] = {f'{best_column_mutcl}': mi_scores_series[best_column_mutcl]}

        ### best threshold ###
        best_columns = list(set(best_columns))
        best_metrics = dict()
        for column in best_columns:
            best_metrics[column] = dict()
            print(f'for column {column}')
            # according to F1 score #
            precision, recall, thresholds = precision_recall_curve(features[target_column], features[column])
            # Find the threshold with the best F1-score
            f1_scores = 2 * (precision * recall) / (precision + recall + 0.00000001)
            optimal_idx = f1_scores.argmax()
            optimal_threshold = thresholds[optimal_idx]

            best_metrics[column]['threshold'] = optimal_threshold
            best_metrics[column]['F1 score'] = f1_scores.max()

            print(f"Optimal threshold: {optimal_threshold} with an F1 score of {f1_scores.max()}")

        ## optionally save the results
        # features.to_csv(f'DataExplo/{filename}')
        # with open(f'DataExplo/{filename}_bets_metrics_per_columns', 'w') as f:
        #    f.write(json.dumps(best_column_metrics))
        # with open(f'DataExplo/{filename}_best_thresh_per_columns', 'w') as f:
        #    f.write(json.dumps(best_metrics))

        return features, best_column_metrics, best_metrics


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

    live_scorings = []

    for idx in tqdm(range(start_sample, end_sample, sample_rate*epoch_len_in_sec)):
        if idx < start_scoring_at_min*sample_rate*60:
            live_scorings.append([0, 0, 0])
            continue
        # -----------------------
        # the signal arrives
        # -----------------------
        loc = ch_l[0:idx]
        roc = ch_r[0:idx]

        info = mne.create_info(ch_names=['eegl', 'eegr'], sfreq=256, ch_types='eeg', verbose='Error')
        mne_array = mne.io.RawArray([roc.copy(), loc.copy()], info, verbose='Error')
        data = YasaClassifier.get_rem_bin_per_epoch(mne_array, 256, ['eegr', 'eegl'])

        live_scorings.append([int(list(data['rem_by_prediction'])[-1]), int(list(data['rem_by_eyes'])[-1]), int(list(data['rem_by_all'])[-1])])

    data = pd.DataFrame.from_records(np.array(live_scorings), columns=['is_rem', 'rem_by_eyes', 'rem_by_all'])
    return data


if __name__ == '__main__':
    pass

