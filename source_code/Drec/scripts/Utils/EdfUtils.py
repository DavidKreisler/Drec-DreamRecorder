import os

import numpy as np
from pyedflib import highlevel

from scripts.Connection.ZmaxHeadband import ZmaxDataID


def save_edf(signals: np.ndarray, channels: list, path: str, file_name: str, signal_scaling_factor: int = 1, sample_rate:int = 256):
    """

    :param signals: a np.ndarray where each row represents one single sample
    :param channels:
    :param path:
    :param file_name:
    :param sample_rate:
    :return:
    """

    if len(signals) <= 1:
        return

    # reformat the signal
    signals_reformatted = signals.T * signal_scaling_factor

    # define mins and max
    digital_min = -(256 ** 2) / 2
    digital_max = (256 ** 2) / 2 - 1
    physical_min = np.ceil(max(min(min(signals_reformatted[0]), min(signals_reformatted[1])), digital_min))
    physical_max = np.floor(min(max(max(signals_reformatted[0]), max(signals_reformatted[1])), digital_max))

    # transform the signal
    signals_reformatted = np.clip(signals_reformatted, physical_min, physical_max)
    signals_reformatted = np.ascontiguousarray(signals_reformatted)

    channel_names = [str(ZmaxDataID(channel)) for channel in channels]
    signal_headers = highlevel.make_signal_headers(channel_names,
                                                   sample_frequency=sample_rate,
                                                   physical_min=physical_min,
                                                   physical_max=physical_max,
                                                   digital_min=digital_min,
                                                   digital_max=digital_max,
                                                   dimension='uV')
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