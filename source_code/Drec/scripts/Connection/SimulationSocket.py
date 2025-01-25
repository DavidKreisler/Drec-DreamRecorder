import os

import time
import mne
import numpy as np

from scripts.Utils.yasa_functions import YasaClassifier
from scripts.Utils.Encode_Decode_Utils import signal_to_hex


class SimulationSocket:
    def __init__(self, file_path: str):
        """
        param filepath: the path a file containing an eeg recording
        """
        self.filename = file_path
        self.file_ext = self.filename.split('.')[-1]
        self.recording = None

        if self.file_ext == 'txt':
            self.recording = np.loadtxt(self.filename, delimiter=',')
            ch_names = ['eegr', 'eegl', 'dx', 'dy', 'dz', 'bodytemp', 'noise', 'light', 'sample_no', 'time']
            self.recording = mne.io.RawArray(data=self.recording.T,
                                             info=mne.create_info(ch_names=ch_names, sfreq=256, units='uV',
                                                                  verbose='error'),
                                             verbose='error').get_data(units='uV')
        elif self.file_ext == 'edf':
            file_path = '/'.join(self.filename.split('/')[0:-1])
            primary_file_name = self.filename.split('/')[-1]
            second_file_name = None
            if primary_file_name == 'EEG L.edf':
                second_file_name = 'EEG R.edf'
            elif primary_file_name == 'EEG R.edf':
                second_file_name = 'EEG L.edf'

            second_path = os.path.join(file_path, second_file_name)

            left = YasaClassifier.get_raw_eeg_from_edf(self.filename).get_data(units='uV')[0]
            right = YasaClassifier.get_raw_eeg_from_edf(second_path).get_data(units='uV')[0]
            self.recording = [right, left]

        else:
            print(f'file extension {self.file_ext} not usable')

        self.curr_idx = 0

    def connect(self):
        # necessary to fit the other Socket
        pass

    def read_one_line(self):
        if self.recording is None:
            return ''

        if self.curr_idx >= len(self.recording[0])/4:
            time.sleep(3)
            return ''

        sig_l = self.recording[0][self.curr_idx]
        sig_r = self.recording[1][self.curr_idx]
        line_encoded = signal_to_hex(sig_l, sig_r)
        self.curr_idx += 1
        return line_encoded

    def stop(self):
        pass


if __name__ == '__main__':
    path = 'C:/coding/git/dreamento/dreamento-online/source_code/Drec/recordings/recording-date-2025-01-08-time-21-08-47/recording-date-2025-01-08-time-21-08-47/recording.edf'
    sim_socket = SimulationSocket(path)
    while True:
        print(sim_socket.read_one_line())
