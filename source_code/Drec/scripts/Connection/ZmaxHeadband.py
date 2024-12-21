import time

import numpy as np
import enum

from scripts.Connection.TcpSniffSocket import TcpSniffSocket
from scripts.Utils.Logger import Logger


class ZmaxDataID(enum.Enum):
    eegr = 0
    eegl = 1
    dx = 2
    dy = 3
    dz = 4
    bodytemp = 5
    bat = 6
    noise = 7
    light = 8
    nasal_l = 9
    nasal_r = 10
    oxy_ir_ac = 11
    oxy_r_ac = 12
    oxy_dark_ac = 13
    oxy_ir_dc = 14
    oxy_r_dc = 15
    oxy_dark_dc = 16
    sample_number = 998
    sample_time = 999

    def __str__(self):
        return self.name


def connect():
    sock = TcpSniffSocket()
    sock.connect()
    return sock


class ZmaxHeadband():
    def __init__(self):
        self.buf_size = 3 * 256  # 3 seconds at 256 frames per second (plotting can be SLOW)
        self.buf_eeg1 = np.zeros((self.buf_size, 1))
        self.buf_eeg2 = np.zeros((self.buf_size, 1))
        self.buf_dx = np.zeros((self.buf_size, 1))
        self.buf_dy = np.zeros((self.buf_size, 1))
        self.buf_dz = np.zeros((self.buf_size, 1))
        self.sock = connect()
        self.msgn = 1  # message number for sending stimulation

    def read(self, reqIDs=None):
        """
        output refers to a list of lists of the desired outputs of the function for example [0,1,3] returns [[eegl, eegr, dy], [eegl, eegr, dy]]
        [0=eegr, 1=eegl, 2=dx, 3=dy, 4=dz, 5=bodytemp, 6=bat, 7=noise, 8=light, 9=nasal_l, 10=nasal_r, 11=oxy_ir_ac,
            12=oxy_r_ac, 13=oxy_dark_ac, 14=oxy_ir_dc, 15=oxy_r_dc, 16=oxy_dark_dc]
        """
        if reqIDs is None:
            reqIDs = [0, 1]

        reqVals = []
        try:
            buf = self.sock.read_one_line()
        except Exception as e:
            Logger().log(f'exception at TcpSniffSocket.read_one_line: {e}')
            buf = ''

        for line in buf.split('\n'):
            if len(line) <= 0:
                continue

            if str.startswith(line, 'DEBUG'):  # ignore debugging messages from server
                continue

            if str.startswith(line, 'D'):  # only process data packets
                p = line.split('.')
                if len(p) == 2:
                    line = p[1]
                    packet_type = self.getbyteat(line, 0)
                    if (packet_type >= 1) and (packet_type <= 11):  # packet type within correct range
                        if len(line) == 120: #119
                            # EEG channels
                            eegr = self.getwordat(line, 1)
                            eegl = self.getwordat(line, 3)
                            # Accelerometer channels
                            dx = self.getwordat(line, 5)
                            dy = self.getwordat(line, 7)
                            dz = self.getwordat(line, 9)
                            # PPG channels (not plotted)
                            oxy_ir_ac = self.getwordat(line, 27)  # requires external nasal sensor
                            oxy_r_ac = self.getwordat(line, 25)  # requires external nasal sensor
                            oxy_dark_ac = self.getwordat(line, 34)  # requires external nasal sensor
                            oxy_ir_dc = self.getwordat(line, 17)  # requires external nasal sensor
                            oxy_r_dc = self.getwordat(line, 15)  # requires external nasal sensor
                            oxy_dark_dc = self.getwordat(line, 32)  # requires external nasal sensor
                            # other channels (not plotted)
                            bodytemp = self.getwordat(line, 36)
                            nasal_l = self.getwordat(line, 11)  # requires external nasal sensor
                            nasal_r = self.getwordat(line, 13)  # requires external nasal sensor
                            light = self.getwordat(line, 21)
                            bat = self.getwordat(line, 23)
                            noise = self.getwordat(line, 19)
                            # convert
                            eegr, eegl = self.ScaleEEG(eegr), self.ScaleEEG(eegl)
                            dx, dy, dz = self.ScaleAccel(dx), self.ScaleAccel(dy), self.ScaleAccel(dz)
                            bodytemp = self.BodyTemp(bodytemp)
                            bat = self.BatteryVoltage(bat)
                            # for function return
                            result = [eegr, eegl, dx, dy, dz, bodytemp, bat, noise, light, nasal_l, nasal_r,
                                      oxy_ir_ac, oxy_r_ac, oxy_dark_ac, oxy_ir_dc, oxy_r_dc, oxy_dark_dc]
                            vals = []
                            for i in reqIDs:
                                vals.append(result[i])
                            reqVals.append(vals)
            else:
                Logger().log(f'data in unexpected format received: {line}', 'DEBUG')

        return reqVals

    def stop(self):
        Logger().log('stopping socket in Headband', 'debug')
        self.sock.stop()

    def __del__(self):
        del self.sock

    def getbyteat(self, buf, idx=0):
        """
        for example getbyteat("08-80-56-7F-EA",0) -> hex2dec(08)
                    getbyteat("08-80-56-7F-EA",2) -> hex2dec(56)
        """
        s = buf[idx * 3:idx * 3 + 2]
        return self.hex2dec(s)

    def getwordat(self, buf, idx=0):
        w = self.getbyteat(buf, idx) * 256 + self.getbyteat(buf, idx + 1)
        return w

    def ScaleEEG(self, e):  # word value to uV
        uvRange = 3952
        d = e - 32768
        d = d * uvRange
        d = d / 65536
        return d

    def ScaleAccel(self, dx):  # word value to 'g'
        d = dx * 4 / 4096 - 2
        return d

    def BatteryVoltage(self, vbat):  # word value to Volts
        v = vbat / 1024 * 6.60
        return v

    def BodyTemp(self, bodytemp):  # word value to degrees C
        v = bodytemp / 1024 * 3.3
        t = 15 + ((v - 1.0446) / 0.0565537333333333)
        return t

    def hex2dec(self, s):
        """return the integer value of a hexadecimal string s"""
        return int(s, 16)

    def dec2hex(self, n, pad=0):
        """return the hexadecimal string representation of integer n"""
        s = "%X" % n
        if pad == 0:
            return s
        else:
            # for example if pad = 3, the dec2hex(5,2) = '005'
            return s.rjust(pad, '0')


if __name__ == '__main__':
    hb = ZmaxHeadband()
    i = 0
    while True:
        print(hb.read([0, 1]))
        if i > 1000:
            break
        i += 1
    hb.stop()
    while True:
        print(hb.read([0, 1]))
        time.sleep(1)