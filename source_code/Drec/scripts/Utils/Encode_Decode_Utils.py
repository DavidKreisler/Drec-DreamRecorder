from scripts.Utils.Int_Hex_conversion import numberToWord, descaleEEG


# ----------------------------------------
# signal transformation methods
# ----------------------------------------
def descaleAccel(sig):
    d = (sig + 2) * 4096 / 4
    return d


def BatteryVoltage(v):  # Volts to word value
    vbat = v / 6.60 * 1024
    return vbat


def BodyTemp(t):  # degree C to word value
    a = ((t - 15) * 0.0565537333333333) + 1.0446
    bodytemp = a * 1024 / 3.3
    return bodytemp


def hex2dec(s):
    """return the integer value of a hexadecimal string s"""
    return int(s, 16)


def dec2hex(n, pad=0):
    """return the hexadecimal string representation of integer n"""
    s = "%X" % n
    if pad == 0:
        return s
    else:
        # for example if pad = 3, the dec2hex(5,2) = '005'
        return s.rjust(pad, '0')


def signal_to_hex(sigl, sigr):
    descaled_sigl = descaleEEG(sigl)
    descaled_sigr = descaleEEG(sigr)
    buf = f"D.06-"\
          f"{numberToWord(descaled_sigl)}-"\
          f"{numberToWord(descaled_sigr)}-"\
          f"00-00-00-00-" \
          f"00-00-00-00-" \
          f"00-00-00-00-" \
          f"00-00-00-00-" \
          f"00-00-00-00-" \
          f"00-00-00-00-" \
          f"00-00-00-00-" \
          f"00-00-00-00-" \
          f"00-00-00" \

    return buf