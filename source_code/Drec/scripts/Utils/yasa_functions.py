import mne.io
import numpy as np
import pandas as pd
import yasa
from scipy.signal import welch
import warnings

warnings.filterwarnings('ignore', message='Trying to unpickle estimator LabelEncoder from version 0.24.2 when *')

class YasaClassifier:

    eeg_bands = [(0.5, 4, 'Delta'),
                 (4, 6, 'Theta_low'),
                 (6, 8, 'Theta_high'),
                 (8, 11, 'Alpha'),
                 (16, 24, 'Beta'),
                 (12, 15, 'Sigma'),
                 (10, 12, 'Sigma_slow')]

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
        _, epochs = yasa.sliding_window(mne_array.copy().get_data(units='V'), sf, window=epoch_len)

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
            hypno = YasaClassifier.get_preds_per_sample(predictions, data, channels)
        loc, roc = data.get_data(units='uV')
        rem = yasa.rem_detect(loc, roc, sf, hypno=hypno, include=4, amplitude=(50, 325),
                              duration=(0.3, 1.2),
                              relative_prominence=0.8, freq_rem=(0.5, 5), remove_outliers=False, verbose='error')

        return rem

    @staticmethod
    def get_epoch_by_epoch_agreement(targets: list, preds: list):
        agr = yasa.EpochByEpochAgreement(targets, preds)
        return agr


def iterate_rec_epoch_wise(raw: mne.io.Raw, channel_l: str, channel_r: str):
    recording = np.empty((0, len(raw.ch_names)))
    epoch = []
    signal = raw.copy()
    signal.pick([channel_l, channel_r])
    ch_l, ch_r = signal.get_data()

    # simulate a signal that is received from second 0
    sf = 256
    epoch_len_in_sec = 30

    for idx in range(0, len(ch_l)):
        # the signal arrives
        loc = ch_l[idx]
        roc = ch_r[idx]

        data_entry = [loc, roc]
        epoch.append(data_entry)

        if idx % (sf * epoch_len_in_sec) == 0:
            recording = np.concatenate((recording, epoch), axis=0)
            epoch = []

    return recording


def simulate_scoring_in_live(raw: mne.io.Raw, channel_l: str, channel_r: str, male: bool = True, age: int = 45):
    """

    :param channel_r: the name of the right channel of the zMax Headband
    :param channel_l: the name of the left channel of the zMax Headband
    :param raw: a signal that contains the at least the l and r eeg channels specified in channels
    :param male: optional: metadata
    :param age: optional: metadata
    :return: None
    """

    score_by_eyes = True
    score_by_hypno = True
    score_py_psd = True

    rem_epochs = []
    psd_theta_epochs = []
    pred_rem_epochs = []
    combined_epochs = []

    global eeg_bands
    signal = raw.copy()
    signal.pick([channel_l, channel_r])
    ch_l, ch_r = signal.get_data()

    # simulate a signal that is received from second 0
    sf = 256
    epoch_len_in_sec = 30
    time_offset_in_sec = 30
    # do nothing the first 2 hours
    idx = sf * 60 * 120

    while idx + sf * epoch_len_in_sec < len(ch_l):
        # the signal arrives
        loc = ch_l[0:idx + (sf * epoch_len_in_sec)]
        roc = ch_r[0:idx + (sf * epoch_len_in_sec)]

        info = mne.create_info(ch_names=['eegl', 'eegr'], sfreq=256, ch_types='eeg')
        mne_array = mne.io.RawArray([roc.copy(), loc.copy()], info, verbose='error')

        # -----------------------
        # calculate power spectral density per 30 sec epoch
        # -----------------------
        if score_py_psd:
            bandpower, bandpower_last_epoch = YasaClassifier.get_bandpower_per_epoch(mne_array)

            delta_mean = (bandpower_last_epoch[0, 0] + bandpower_last_epoch[0, 1]) / 2
            theta_mean = (bandpower_last_epoch[1, 0] + bandpower_last_epoch[1, 1]) / 2
            # alpha_mean = (bandpower_last_epoch[2, 0] + bandpower_last_epoch[2, 1]) / 2
            # sigma_mean = (bandpower_last_epoch[3, 0] + bandpower_last_epoch[3, 1]) / 2
            # beta_mean = (bandpower_last_epoch[4, 0] + bandpower_last_epoch[4, 1]) / 2
            # gamma_mean = (bandpower_last_epoch[5, 0] + bandpower_last_epoch[5, 1]) / 2
            theta_delta_ratio = theta_mean / delta_mean

            if theta_delta_ratio > 0.22:
                psd_theta_epochs.append(1)
            else:
                psd_theta_epochs.append(0)
        # -----------------------
        # calculate hypnogram
        # -----------------------
        if score_by_hypno:
            preds = YasaClassifier.get_preds_per_epoch(mne_array=mne_array)
            if preds[-1] == 'R':
                pred_rem_epochs.append(1)
            else:
                pred_rem_epochs.append(0)
        # -----------------------
        # calculate rem phases
        # -----------------------
        if score_by_eyes:
            rem = YasaClassifier.get_eyes(mne_array, preds)
            if rem:
                rem_pd = rem.summary()
                rem_last_30_sec = rem_pd.loc[rem_pd['Start'] > (idx / 256)]
            else:
                rem_last_30_sec = pd.DataFrame()

            if not rem_last_30_sec.empty:
                rem_epochs.append(1)
            else:
                rem_epochs.append(0)

        # if all say rem
        if score_by_hypno and score_by_eyes and score_py_psd:
            if (theta_delta_ratio > 0.22) and preds[-1] == 'R' and not rem_last_30_sec.empty:
                combined_epochs.append(1)
            else:
                combined_epochs.append(0)

        # -----------------------
        # uptick the idx
        # -----------------------
        idx += sf * time_offset_in_sec

    if score_by_hypno and score_by_eyes and score_py_psd:
        print(f'combined_epochs: {combined_epochs}')
    if score_by_eyes:
        print(f'rem_epochs: {rem_epochs}')
    if score_py_psd:
        print(f'psd_theta_epochs: {psd_theta_epochs}')
    if score_by_hypno:
        print(f'pred_rem_epochs: {pred_rem_epochs}')

    scoring_df = pd.DataFrame(list(zip(combined_epochs, rem_epochs, psd_theta_epochs, pred_rem_epochs)), columns=['combines', 'eyes', 'theta/delta', 'pred'])
    print(scoring_df.head())
    scoring_df.to_csv('scoring_results.csv')



if __name__ == '__main__':
    my_raw_path = 'C:/coding/git/dreamento/dreamento-online/source_code/Drec/recordings/recording-date-2024-11-27-time-21-39-45/recording-date-2024-11-27-time-21-39-45/complete_recording.edf'
    z_max_path = 'C:/coding/git/dreamento/dreamento-online/source_code/Drec/recordings/2024 11 27 - 21 39 27/2024 11 27 - 21 39 27/'

    channels = ['eegl', 'eegr']

    raw_own = YasaClassifier.get_raw_eeg_from_edf(my_raw_path).pick('eeg')
    raw_zmax_l = YasaClassifier.get_raw_eeg_from_edf(z_max_path + 'EEG L.edf')
    raw_zmax_r = YasaClassifier.get_raw_eeg_from_edf(z_max_path + 'EEG R.edf')
    raw_zmax = mne.io.RawArray(data=[raw_zmax_l.get_data()[0], raw_zmax_r.get_data()[0]],
                               info=mne.create_info(ch_names=channels, sfreq=256, ch_types='eeg'),
                               verbose='error')

    hypno_own = YasaClassifier.get_sleep_hypno(raw_own, 'eegl')
    hypno_zmax = YasaClassifier.get_sleep_hypno(raw_zmax, 'eegl')

    hypno_probs_raw = YasaClassifier.get_sleep_hypno_probs(raw_own, 'eegl')
    hypno_probs_zmax = YasaClassifier.get_sleep_hypno_probs(raw_zmax, 'eegl')

    bandpower_p_epoch_raw = YasaClassifier.get_bandpower_per_epoch(raw_own)
    bandpower_p_epoch_zmax = YasaClassifier.get_bandpower_per_epoch(raw_zmax)

    preds_own = YasaClassifier.get_preds_per_epoch(raw_own)
    preds_zmax = YasaClassifier.get_preds_per_epoch(raw_zmax)

    preds_p_sample_own = YasaClassifier.get_preds_per_sample(raw_own, preds_own, channels)
    preds_p_sample_zmax = YasaClassifier.get_preds_per_sample(raw_zmax, preds_zmax, channels)

    eyes_own = YasaClassifier.get_eyes(raw_own, channels)
    eyes_zmax = YasaClassifier.get_eyes(raw_zmax, channels)

    bandpower_own = YasaClassifier.get_bandpower(raw_own, channels, preds_p_sample_own)
    bandpower_zmax = YasaClassifier.get_bandpower(raw_zmax, channels, preds_p_sample_zmax)

    agr_own = YasaClassifier.get_epoch_by_epoch_agreement([yasa.Hypnogram(hypno_zmax, scorer='1')],
                                                          [yasa.Hypnogram(hypno_own, scorer='2')])

# TODO:
# check if rem_detect performs exactly the same even if the signal is fed in pieces
# if yes just do rem detection of the last 15 seconds
