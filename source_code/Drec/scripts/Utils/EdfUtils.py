import os

import numpy as np
from pyedflib import highlevel

from scripts.Connection.ZmaxHeadband import ZmaxDataID


def save_edf(signals: np.ndarray, channels: list, path: str, file_name: str, sample_rate:int = 256):
    """

    :param signals: a np.ndarray where each row represents one single sample
    :param channels:
    :param path:
    :param file_name:
    :param sample_rate:
    :return:
    """
    min_eeg_val = -256**2 - 1
    max_eeg_val = 256**2 - 1
    if len(signals) <= 1:
        return

    # write an edf file
    signals_reformatted = signals.T
    #signals_reformatted = np.clip(signals_reformatted, min_eeg_val, max_eeg_val)
    signals_reformatted = np.ascontiguousarray(signals_reformatted)
    channel_names = [str(ZmaxDataID(channel)) for channel in channels]
    signal_headers = highlevel.make_signal_headers(channel_names,
                                                   sample_frequency=sample_rate,
                                                   physical_min=-256**2,
                                                   physical_max=256**2)
    header = highlevel.make_header(patientname='patient')
    try:
        highlevel.write_edf(os.path.join(path, file_name),
                            signals_reformatted,
                            signal_headers,
                            header)
    except Exception as e:
        print(f'[ERROR] when writing edf: {e}')


def save_as_txt(signals: np.ndarray, path: str, file_name: str):
    np.savetxt(os.path.join(path, file_name), signals, delimiter=',')  # save recording as txt