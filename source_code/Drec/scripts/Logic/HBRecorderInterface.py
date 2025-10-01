import os
from pathlib import Path

import mne
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from scripts.Logic.RecorderThread import RecordThread
from scripts.Utils.yasa_functions import YasaClassifier
from scripts.Utils.Logger import Logger

from scripts.Utils.EdfUtils import save_edf, save_as_txt
from scripts.dreamento_original import realTimeAutoScoring


class HBRecorderInterface:
    def __init__(self):
        self.sample_rate = 256
        self.signalType = [0, 1, 2, 3, 4, 5, 7, 8]
        # [
        #   0=eegr, 1=eegl, 2=dx, 3=dy, 4=dz, 5=bodytemp,
        #   6=bat, 7=noise, 8=light, 9=nasal_l, 10=nasal_r,
        #   11=oxy_ir_ac, 12=oxy_r_ac, 13=oxy_dark_ac,
        #   14=oxy_ir_dc, 15=oxy_r_dc, 16=oxy_dark_dc
        # ]
        self.scoring_delay_in_epochs = 10  # only start scoring after 5 min as this is the minimum signal length to create a mne array
        self.recording = np.empty(shape=(0, len(self.signalType) + 2))  # +2 because we add 2 columns, sample # and time

        self.hb = None
        self.recorderThread = None

        self.isRecording = False
        self.firstRecording = True
        self.recordingFinished = True

        self.rem_by_staging_and_eyes = []
        self.dreamento_scoring = []
        self.epochCounter = 0

        # program parameters
        self.scoreSleep = False
        self.best_scoring_feature = None
        self.best_scoring_metric = None

        # webhook
        self.webHookBaseAdress = "http://127.0.0.1:5000/webhookcallback/"
        self.webhookActive = False

        self.save_dir = None

        self.scoring_model = 'both' # 'dreamento', 'yasa', 'both'
        self.dreamento_model = realTimeAutoScoring.importModel('out_QS/train/21')

    def start_recording(self):
        if self.isRecording:
            return

        Logger().log('starting recording', 'info')
        self.recorderThread = RecordThread(signalType=self.signalType)

        if self.firstRecording:
            self.firstRecording = False

        self.isRecording = True

        self.recorderThread.start()

        self.recorderThread.finished.connect(self.on_recording_finished)
        self.recorderThread.recordingFinishedSignal.connect(self.on_recording_finished_save_data)
        self.recorderThread.sendEpochDataSignal.connect(self.get_epoch_data)
        self.recordingFinished = False

        Logger().log(f'recording started', 'info')

    def stop_recording(self):
        if not self.isRecording:
            return

        Logger().log('stopping recording', 'info')
        self.recorderThread.stop()
        #self.recorderThread.quit()
        self.isRecording = False
        Logger().log('recording stopped', 'info')
        print('recording stopped')

    def on_recording_finished(self):
        Logger().log('finished signal received', 'info')
        print('recording finished')

    def on_recording_finished_save_data(self, filePath):
        Logger().log('starting to save data', 'info')
        self.recordingFinished = True

        # ensures directory exists
        Path(f"{filePath}").mkdir(parents=True, exist_ok=True)

        # save the recording
        save_edf(self.recording,
                 self.signalType,
                 filePath,
                 'recording.edf')

        save_as_txt(self.recording, filePath, 'recording.txt')

        # save the predictions
        if self.rem_by_staging_and_eyes:
            # write only the lines that are rem:
            rem_lines = []
            for time, epoch, rem_by_staging, rem_by_eyes in self.rem_by_staging_and_eyes:
                if rem_by_staging == 1 and rem_by_eyes == 1:
                    rem_lines.append((epoch, epoch*30, time.strftime("%Y-%m-%d %H:%M:%S"), 'REM detected'))

            with open(os.path.join(filePath, "rem_markers.txt"), "a") as outfile:
                outfile.write("\n".join(str(epoch) + ' - ' + str(seconds) + 's' + ' - ' +  time + ' - ' + str(is_rem)
                                        for epoch, seconds, time, is_rem in rem_lines))

        if self.dreamento_scoring:
            outfile_content = []
            stagesList = ['W', 'N1', 'N2', 'N3', 'REM', 'MOVE', 'UNK'] # taken from original repository
            for line in self.dreamento_scoring:
                outfile_content.append((line['epoch'], stagesList[line['pred']]))

            with open(os.path.join(filePath, 'dreamento_scoring.txt'), 'a') as outfile:
                outfile.writelines('\n'.join(str(epoch) + ' - ' + str(pred) for epoch, pred in outfile_content))

        # send signal to webhook if it is running
        if self.webhookActive:
            requests.post(self.webHookBaseAdress + 'finished')

        self.save_dir = filePath
        Logger().log('finished saving data', 'info')

    def start_scoring(self):
        self.scoreSleep = True
        Logger().log('scoring started, waiting for new connection to HDServer.', 'info')

    def stop_scoring(self):
        self.scoreSleep = False
        print('scoring stopped')
        Logger().log(f'scoring stopped', 'info')

    def get_epoch_data(self, data: list, epoch_counter: int):
        Logger().log('getting epoch data', 'DEBUG')
        self.recording = np.concatenate((self.recording, data), axis=0)

        if self.scoreSleep: # and epoch_counter > self.scoring_delay_in_epochs:
            Logger().log(f'scoring data', 'Debug')
            self._score_curr_data(epoch_counter, self.recording[:, 0], self.recording[:, 1])

        if self.webhookActive:  # Do this AFTER the scoring is done
            self._send_to_webhook()

    def _score_with_dreamento(self, epoch_counter, eegr, eegl):
        """
        dreamento model scoring
        eegr and eegl are expected to contain 30*256 elements each
        """
        sigRef = np.asarray(eegr)
        sigReq = np.asarray(eegl)
        sigRef = sigRef.reshape((1, sigRef.shape[0]))
        sigReq = sigReq.reshape((1, sigReq.shape[0]))
        modelPrediction = realTimeAutoScoring.Predict_array(
            output_dir="./DataiBand/output/Fp1-Fp2_filtered",
            args_log_file="info_ch_extract.log", filtering_status=True,
            lowcut=0.3, highcut=30, fs=256, signal_req=sigReq, signal_ref=sigRef, model=self.dreamento_model)

        self.dreamento_scoring.append({'pred': int(modelPrediction[0]), 'epoch': epoch_counter})

    def _score_pure_yasa(self, epoch_counter, eegr, eegl):
        """
        eegr and eegl are expected to contain at least 5 min worth of data (10*30*256)
        """
        # pure yasa scoring
        info = mne.create_info(ch_names=['eegr', 'eegl'], sfreq=self.sample_rate, ch_types='eeg', verbose='ERROR')
        mne_array = mne.io.RawArray([eegr, eegl], info, verbose='ERROR')

        data = YasaClassifier.get_rem_bin_per_epoch(mne_array, 256, ['eegr', 'eegl'])
        # data = YasaClassifier.get_power_bands_and_ground_truth_per_epoch(mne_array, 256, ['eegr', 'eegl'])

        pred_by_all = list(data['rem_by_all'])[-1]
        pred_by_scoring = list(data['rem_by_prediction'])[-1]
        pred_by_eyes = int(list(data['rem_by_eyes'])[-1])
        self.rem_by_staging_and_eyes.append((datetime.now(),
                                             epoch_counter,
                                             pred_by_scoring,
                                             pred_by_eyes))

    def _score_curr_data(self, epoch_counter, eegr, eegl):
        sigR = eegr[-30 * 256:]
        sigL = eegl[-30 * 256:]

        if self.scoring_model == 'yasa' or self.scoring_model == 'both':
            if epoch_counter > self.scoring_delay_in_epochs:
                self._score_pure_yasa(epoch_counter, eegr, eegl)

        if self.scoring_model == 'dreamento' or self.scoring_model == 'both':
            self._score_with_dreamento(epoch_counter, sigR, sigL)






    def _send_to_webhook(self):
        if len(self.rem_by_staging_and_eyes) <= 0:
            return
        rem_by_scoring = 'None'
        rem_by_eyes = 'None'
        time = 'None'
        epoch = 'None'

        if len(self.rem_by_staging_and_eyes) > 0:
            time, epoch, rem_by_scoring, rem_by_eyes = self.rem_by_staging_and_eyes[-1]

        data = {'rem_by_staging': rem_by_scoring,
                'rem_by_eyes': rem_by_eyes,
                'time': time,
                'epoch': epoch}
        try:
            requests.post(self.webHookBaseAdress + 'sleepstate', data=data)
        except Exception as e:
            print(e)
            Logger().log('error when posting request to webhook. webhook is probably not available', 'Warning')
            print('error when posting request to webhook. webhook is probably not available')

    def start_webhook(self):
        Logger().log(f'starting webhook', 'info')
        try:
            requests.post(self.webHookBaseAdress + 'hello', data={'hello': 'hello'})
            self.webhookActive = True
        except Exception as e:
            Logger().log('webhook seems to be offline. not activating', 'warning')
            self.webhookActive = False
            return
        Logger.log(f'webhook started', 'info')

    def stop_webhook(self):
        self.webhookActive = False
        Logger.log(f'webhook stopped', 'info')

    def set_signaltype(self, types=None):
        if self.isRecording:
            Logger().log('Setting signaltype is not possible during a recording. Stop the recording first and then try again.', 'warning')
            return
        if types is None:
            Logger().log('Types need to be specified to set the signaltype.', 'ERROR')
            return
        self.signalType = types
        self.recording = np.empty(shape=(0, len(self.signalType)))

    def set_scoring_delay(self, delay_in_epochs: int):
        delay_in_epochs = max(delay_in_epochs, 10)
        self.scoring_delay_in_epochs = delay_in_epochs

    def quit(self):
        Logger().log('Quit called', 'info')

        if self.recorderThread:
            self.stop_recording()

        return self.save_dir

