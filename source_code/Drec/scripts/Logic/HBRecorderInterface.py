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
        self.scoring_delay = 120  # only start scoring after 2 hours
        self.recording = np.empty(shape=(0, len(self.signalType) + 2))  # +2 because we add 2 columns, sample # and time

        self.hb = None
        self.recorderThread = None

        self.isRecording = False
        self.firstRecording = True
        self.recordingFinished = True

        self.rem_by_staging_and_eyes = []
        self.rem_by_powerbands = []
        self.epochCounter = 0

        # program parameters
        self.scoreSleep = False
        self.best_scoring_feature = None
        self.best_scoring_metric = None

        # webhook
        self.webHookBaseAdress = "http://127.0.0.1:5000/webhookcallback/"
        self.webhookActive = False

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

        print('recording started')

    def stop_recording(self):
        if not self.isRecording:
            return

        Logger().log('stopping recording', 'info')
        self.recorderThread.stop()
        #self.recorderThread.quit()
        self.isRecording = False
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
            with open(os.path.join(filePath, "rem_by_eyes_and_staging.txt"), "a") as outfile:
                outfile.write("\n".join(str(epoch) + '-' + str(pred) for time, epoch, pred in self.rem_by_staging_and_eyes))

        # save the predictions
        if self.rem_by_powerbands:
            with open(os.path.join(filePath, "rem_by_powerbands.txt"), "a") as outfile:
                outfile.write("\n".join(str(epoch) + '-' + str(pred) for time, epoch, pred in self.rem_by_powerbands))

        # send signal to webhook if it is running
        if self.webhookActive:
            requests.post(self.webHookBaseAdress + 'finished')

        Logger().log('finished saving data', 'info')

    def start_scoring(self):
        self.scoreSleep = True
        print('scoring started, waiting for new connection to HDServer.')

    def stop_scoring(self):
        self.scoreSleep = False
        print('scoring stopped')

    def get_epoch_data(self, data: list, epoch_counter: int):
        Logger().log('getting epoch data', 'DEBUG')
        self.recording = np.concatenate((self.recording, data), axis=0)

        if self.scoreSleep and epoch_counter > self.scoring_delay:
            if not self.best_scoring_feature:
                Logger().log(f'setting optimum scoring metric', 'Debug')
                self._set_optimum_pwrband_scoring()

            Logger().log(f'scoring data', 'Debug')
            self._score_curr_data(epoch_counter)

        if self.webhookActive:  # Do this AFTER the scoring is done
            self._send_to_webhook()

    def _set_optimum_pwrband_scoring(self):
        # get the best metrics for rem detection based on bower bands before the scoring starts

        ### ----------------------------------------
        # IS DATA IN UV?!
        ### ----------------------------------------
        Logger().log(f'Data has following mean values when finding best metrics: {max(self.recording[:, 0])}', 'DEBUG')

        # create array
        info = mne.create_info(ch_names=['eegr', 'eegl'], sfreq=self.sample_rate, ch_types='eeg', verbose='ERROR')
        mne_array = mne.io.RawArray([self.recording[:, 0], self.recording[:, 1]], info, verbose='ERROR')

        # find best scoring metrics
        data = YasaClassifier.get_scoring_metrics(mne_array, 256, ['eegr', 'eegl'])
        optimal_metric = YasaClassifier.find_optimal_rem_scoring_metrics(data)

        if optimal_metric:  # may be None when no rem phases are found
            self.best_scoring_feature = optimal_metric['Best Feature']
            self.best_scoring_metric = optimal_metric['Best Metrics']
            Logger().log(f'optimal scoring metric found: {self.best_scoring_feature} with accuracy {self.best_scoring_metric["Best Accuracy"]} and a threshold of {self.best_scoring_metric["Best Threshold"]}', 'info')

    def _score_curr_data(self, epoch_counter):
        eegr = self.recording[:, 0]
        eegl = self.recording[:, 1]
        info = mne.create_info(ch_names=['eegr', 'eegl'], sfreq=self.sample_rate, ch_types='eeg', verbose='ERROR')
        mne_array = mne.io.RawArray([eegr, eegl], info, verbose='ERROR')

        data = YasaClassifier.get_scoring_metrics(mne_array, 256, ['eegr', 'eegl'])

        predictionByEyesAndScoringToTransmit = list(data['is_rem'])[-1]
        self.rem_by_staging_and_eyes.append((datetime.now(),
                                             epoch_counter,
                                             predictionByEyesAndScoringToTransmit))

        if self.best_scoring_feature is not None:
            if '/' in self.best_scoring_feature:
                feat1, feat2 = self.best_scoring_feature.split('/')
                pred = (data[feat1] / data[feat2]) > self.best_scoring_metric['Best Threshold']
            else:
                pred = (data[self.best_scoring_feature]) > self.best_scoring_metric['Best Threshold']

            predictionByPwrBandToTransmit = list(pred)[-1]
            self.rem_by_powerbands.append((datetime.now(),
                                           epoch_counter,
                                           predictionByPwrBandToTransmit))

    def _send_to_webhook(self):
        if len(self.rem_by_staging_and_eyes) <= 0:
            return
        pred = 'None'
        pwr_band = 'None'
        time = 'None'
        epoch = 'None'

        if len(self.rem_by_staging_and_eyes) > 0:
            time, epoch, pred = self.rem_by_staging_and_eyes[-1]
        if len(self.rem_by_powerbands) > 0:
            Logger().log(f'rem_by_pwrband: {self.rem_by_powerbands}', 'debug')
            time_pwr_band, epoch_pwr_band, pwr_band = self.rem_by_powerbands[-1]

        data = {'rem_by_staging_and_eyes': pred,
                'rem_by_powerbands': pwr_band,
                'time': time,
                'epoch': epoch}
        try:
            requests.post(self.webHookBaseAdress + 'sleepstate', data=data)
        except Exception as e:
            print(e)
            print('error when posting request to webhook. webhook is probably not available')

    def start_webhook(self):
        try:
            requests.post(self.webHookBaseAdress + 'hello', data={'hello': 'hello'})
            self.webhookActive = True
        except Exception as e:
            #print(e)
            print('webhook seems to be offline. not activating')
            return
        print('webhook started')

    def stop_webhook(self):
        self.webhookActive = False
        print('webhook stopped')

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
        self.scoring_delay = delay_in_epochs

    def quit(self):
        Logger().log('Quit called', 'info')

        if self.recorderThread:
            self.stop_recording()

