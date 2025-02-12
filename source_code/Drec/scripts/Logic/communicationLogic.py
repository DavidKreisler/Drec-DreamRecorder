import os
import shutil
import sys

from PyQt5.QtWidgets import QApplication

from scripts.UI.CLI import CLIThread
from scripts.Logic.HBRecorderInterface import HBRecorderInterface

from scripts.Utils.Logger import Logger


class CommunicationLogic:
    def __init__(self):
        self.app = QApplication(sys.argv)

        self.cliThread = CLIThread()
        self.hbif = HBRecorderInterface()

    def start(self):
        self.cliThread.start()

        self._connectSignals()

        self.app.exec_()

    def _connectSignals(self):
        # CLI
        self.cliThread.cli.start_signal.connect(self.startRecording)
        self.cliThread.cli.stop_signal.connect(self.stopRecording)
        self.cliThread.cli.start_scoring_signal.connect(self.startScoring)
        self.cliThread.cli.stop_scoring_signal.connect(self.stopScoring)
        self.cliThread.cli.start_webhook_signal.connect(self.startWebhook)
        self.cliThread.cli.stop_webhook_signal.connect(self.stopWebhook)
        self.cliThread.cli.set_signaltype_signal.connect(self.setSignaltype)
        self.cliThread.cli.set_scoring_delay_signal.connect(self.setScoringDelay)
        self.cliThread.cli.quit_signal.connect(self.quit)

    def startRecording(self):
        self.hbif.start_recording()

    def stopRecording(self):
        self.hbif.stop_recording()

    def startScoring(self):
        self.hbif.start_scoring()

    def stopScoring(self):
        self.hbif.stop_scoring()

    def startWebhook(self):
        self.hbif.start_webhook()

    def stopWebhook(self):
        self.hbif.stop_webhook()

    def setSignaltype(self, signalTypes: list):
        self.hbif.set_signaltype(signalTypes)

    def setScoringDelay(self, delay_in_epochs: int):
        self.hbif.set_scoring_delay(delay_in_epochs)

    def quit(self, _: bool):

        if self.hbif.isRecording:
            # gracefully stop recording if it is running to allow saving files. When it is called only once from within the
            # class it has to receive a signal again to start saving. Therefore we do it here already
            self.hbif.stop_recording()
            print('recording was still running, quitting aborted and recording stopped. please try again later!')
            return

        if not self.hbif.recordingFinished:
            print('did not receive the finished signal from the recorder thread yet. It may still be occupied with '
                  'saving your files. \n If this takes longer than 20 minutes or you did not start a recording please '
                  'file an issue at the github repository.')
            return

        save_dir = self.hbif.quit()

        self.cliThread.stop()
        self.cliThread.quit()

        self.app.quit()

        # move the log file into the folder
        Logger().close()
        try:
            shutil.move("log.log", f"{save_dir}/log.log")
        except Exception as e:
            print(f"{save_dir}/log.log")
            print(e)






