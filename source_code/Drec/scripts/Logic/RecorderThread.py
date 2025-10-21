from datetime import datetime
import time
from PyQt5.QtCore import QThread, pyqtSignal


from scripts.Connection.ZmaxHeadband import ZmaxDataID, ZmaxHeadband
from scripts.Utils.Logger import Logger


class RecordThread(QThread):
    recordingFinishedSignal = pyqtSignal(str)
    sendEpochDataSignal = pyqtSignal(object, int)

    def __init__(self, parent=None, signalType=None):
        super(RecordThread, self).__init__(parent)
        if signalType is None:
            signalType = [0, 1, 5, 2, 3, 4]

        self.model_CNNLSTM = None
        self.threadactive = True
        self.signalType = signalType  # "EEGR, EEGL, TEMP, DX, DY, DZ"
        self.stimulationType = ""
        self.secondCounter = 0
        self.dataSampleCounter = 0
        self.totalDataSampleCounter = 0
        self.epochCounter = 0
        self.sample_rate = 256

    def sendEpochData(self, data):
        Logger().log('sending epoch data', 'debug')
        self.sendEpochDataSignal.emit(data, self.epochCounter)

    def run(self):
        Logger().log('starting the recorder thread', 'info')
        recording = []
        cols = self.signalType
        cols.extend([998, 999])  # add two columns for sample number, sample time
        recording.append(cols)  # first row of received data is the col_id. eg: 0 => eegr
        hb = ZmaxHeadband()  # create a new client on the server, therefore we use it only for reading the stream

        now = datetime.now()  # for file name
        dt_string = now.strftime("recording-date-%Y-%m-%d-time-%H-%M-%S")
        file_path = f".\\recordings\\{dt_string}"

        actual_start_time = time.time()
        Logger().log((f'actual start time {actual_start_time}'), 'info')

        buffer = []
        buffer2analyzeIsReady = False
        dataSamplesToAnalyzeCounter = 0  # count samples, when reach 30*256, feed all to deep learning model

        self.secondCounter = 0
        self.epochCounter = 0  # each epoch is 30 seconds

        while True:
            self.dataSampleCounter = 0  # count samples in each second
            self.secondCounter += 1

            t_end = time.time() + 1

            while time.time() < t_end:
                try:
                    x = hb.read(cols[:-2])
                except Exception as e:
                    #print(f'[ERROR] at ZMaxHeadband.read(): {e}')
                    Logger().log(e, 'ERROR')
                    x = []
                if x:
                    for line in x:
                        dataEntry = line
                        dataEntry.extend([self.dataSampleCounter, self.secondCounter])
                        self.dataSampleCounter += 1
                        self.totalDataSampleCounter += 1
                        recording.append(dataEntry)
                        buffer.append(dataEntry)
                        if not buffer2analyzeIsReady:
                            if self.secondCounter >= 2:  # ignore 1st second for analysis, because it is unstable
                                dataSamplesToAnalyzeCounter += 1

                                if dataSamplesToAnalyzeCounter >= 30 * self.sample_rate:
                                    buffer2analyzeIsReady = True
                                    self.epochCounter += 1

                else:
                    Logger().log('No data at hb.read()', 'DEBUG')
                    continue

            if buffer2analyzeIsReady:
                # send all epoch data
                while len(buffer) >= 30*self.sample_rate:
                    self.sendEpochData(buffer[0:30*self.sample_rate])
                    buffer = buffer[30*self.sample_rate:]

                # reset
                dataSamplesToAnalyzeCounter = 0

                buffer2analyzeIsReady = False

            if self.threadactive is False:
                Logger().log('stopping headband', 'info')
                hb.stop()
                break

        self.recordingFinishedSignal.emit(f"{file_path}")  # send path of recorded file to mainWindow

    def stop(self):
        Logger().log('stopping recorder Thread', 'info')
        self.threadactive = False

