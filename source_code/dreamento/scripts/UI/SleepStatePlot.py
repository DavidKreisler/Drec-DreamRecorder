import sys

import numpy as np
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg  # for eeg plotting


class SleepStatePlot:
    def __init__(self):
        self.desiredXrange = 5  # set default (0,5) - (5,10) - (10-15) - ...
        self.desiredYrange = 60  # set default (-60,60)

        self.app = QApplication(sys.argv)

        self.plotWidget = pg.PlotWidget()

        self.EEGLinePen1 = pg.mkPen(color=(100, 90, 150), width=1.5)
        self.EEGLinePen2 = pg.mkPen(color=(90, 170, 160), width=1.5)

        t = [number / self.sample_rate for number in range(self.sample_rate * 30)]

        self.eegLine1 = self.plotWidget.plot(t, np.random.randn(30 * self.sample_rate), self.EEGLinePen1)
        self.eegLine2 = self.plotWidget.plot(t, np.random.randn(30 * self.sample_rate), self.EEGLinePen2)

        self.plotWidget.setWindowTitle('EEG Signal')

    def show(self):
        self.plotWidget.show()
        self.app.exec_()

    def stop(self):
        self.app.quit()
        self.app.exit(0)

    def setData(self, t, sigR, sigL):
        self.eegLine1.setData(t, sigR, pen=self.EEGLinePen1)
        self.eegLine2.setData(t, sigL, pen=self.EEGLinePen2)

        self.displayedXrangeCounter = len(sigL)  # for plotting Xrange — number of displayed samples on screen

        sec = int(np.floor(self.displayedXrangeCounter / self.sample_rate))
        if sec % self.desiredXrange == 0:
            random_reset_timer_variable = 30
            k = int(np.floor(sec / self.desiredXrange))
            if self.desiredXrange * k < random_reset_timer_variable:
                xMin = self.desiredXrange * k
                xMax = self.desiredXrange * (k + 1)
            else:
                xMin = 0
                xMax = self.desiredXrange
            a_X = self.plotWidget.getAxis('bottom')
            ticks = range(xMin, xMax, 1)
            a_X.setTicks([[(v, str(v)) for v in ticks]])
            self.plotWidget.setXRange(xMin, xMax, padding=0)

