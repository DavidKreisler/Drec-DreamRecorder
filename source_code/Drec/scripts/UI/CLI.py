import cmd
from PyQt5.QtCore import QObject, pyqtSignal, QThread

from scripts.Utils.Logger import Logger


class CLIThread(QThread):
    def __init__(self):
        super().__init__()
        self.cli = SleepRecorderCLI()

    def run(self):
        self.cli.cmdloop()

    def stop(self):
        self.cli.stop()
        self.quit()


class SleepRecorderCLI(cmd.Cmd, QObject):
    prompt = '>> '
    intro = 'Welcome to the sleep recoder CLI. type "help" for all available commands and "help <cmd>" to see info on the specific command.'

    start_signal = pyqtSignal(bool)
    stop_signal = pyqtSignal(bool)
    start_scoring_signal = pyqtSignal(bool)
    stop_scoring_signal = pyqtSignal(bool)
    start_webhook_signal = pyqtSignal(bool)
    stop_webhook_signal = pyqtSignal(bool)
    set_signaltype_signal = pyqtSignal(list)
    set_scoring_delay_signal = pyqtSignal(int)
    quit_signal = pyqtSignal(bool)

    def __init__(self):
        cmd.Cmd.__init__(self)
        QObject.__init__(self)  # Initialize QObject

        self._is_running = True

    def stop(self):
        """Stop the CLI"""""
        self._is_running = False
        return True

    def do_quit(self, line):
        """Exit the CLI."""
        self.quit_signal.emit(True)

    def do_start(self, line):
        """start the recoring, scoring and the webhook. If the webhook startup fails, the recording and scoring are
        still started."""
        Logger().log('start signal emitted', 'DEBUG')
        self.start_signal.emit(True)
        self.start_scoring_signal.emit(True)
        self.start_webhook_signal.emit(True)
        info_message = 'Waiting for new tcp connection.\nStart the recording by connecting the HDRecorder to the HDServer. \nThe recording is running when you see a [CONNECTED] message.'
        print(info_message)

    def do_stop(self, line):
        """stops recording, scoring and webhook"""
        Logger().log('stop signal emitted', 'DEBUG')
        self.stop_signal.emit(True)

    def do_start_recording(self, line):
        """Start the recoring"""
        self.start_signal.emit(True)

    def do_start_scoring(self, line):
        """start scoring the eeg signal."""
        self.start_scoring_signal.emit(True)

    def do_stop_scoring(self, line):
        """stops scoring the eeg signal"""
        self.stop_scoring_signal.emit(True)

    def do_start_webhook(self, line):
        """start the webhook so other programs can read the prediction status from the port 5000. This recorder sends the data there every 30 seconds."""
        self.start_webhook_signal.emit(True)

    def do_stop_webhook(self, line):
        """stop the webhook"""
        self.stop_webhook_signal.emit(True)

    def do_set_signaltype(self, line):
        """set the signals that should be recorded. pass as a comma separated numbers. For possible signals type 'show_possible_signals'"""
        numbers = line.split(',')
        try:
            numbers = [int(n) for n in numbers]
            self.set_signaltype_signal.emit(numbers)
        except Exception:
            print('pass signals as integers according to "show_possible_signals", e.g. " set_signaltype 1,2,3"')

    def do_show_possible_signals(self, line):
        """shows all possible eeg signals to be set by 'set_signaltype'."""
        mes = '[0=eegr, 1=eegl, 2=dx, 3=dy, 4=dz, 5=bodytemp, 6=bat, 7=noise, 8=light, 9=nasal_l, 10=nasal_r, 11=oxy_ir_ac, 12=oxy_r_ac, 13=oxy_dark_ac, 14=oxy_ir_dc, 15=oxy_r_dc, 16=oxy_dark_dc]'
        print(mes)

    def do_set_scoring_delay(self, line):
        """set the number of epochs to wait before scoring is started. Epoch length si 30 seconds. Min value is 10 (5 min)"""
        try:
            val = int(line)
            self.set_scoring_delay_signal.emit(val)
        except ValueError:
            print(f'please provide a numer. "{line}" was not interpretable as integer.')

