

### -----------------------
# hex to dec
### -----------------------
def wordToNum(word):
    first, second = word.split('-')
    num = hex2dec(first) * 256 + hex2dec(second)
    return num


def hex2dec(s):
    """return the integer value of a hexadecimal string s"""
    return int(s, 16)


def getbyteat(buf, idx=0):
    """
    for example getbyteat("08-80-56-7F-EA",0) -> hex2dec(08)
                getbyteat("08-80-56-7F-EA",2) -> hex2dec(56)
    """
    s = buf[idx * 3:idx * 3 + 2]
    return hex2dec(s)


def getwordat(buf, idx=0):
    w = getbyteat(buf, idx) * 256 + getbyteat(buf, idx + 1)
    return w




### -----------------------------------------
# dec to hex
### -----------------------------------------
def dec2hex(n, pad=0):
    """return the hexadecimal string representation of integer n"""
    n = round(n)
    s = "%X" % n
    if pad == 0:
        return s
    else:
        # for example if pad = 3, the dec2hex(5,2) = '005'
        return s.rjust(pad, '0')


def numberToWord(n):
    if n <= 256:
        return f'00-{dec2hex(n, pad=2)}'
    message = f'{dec2hex(int(n / 256), pad=2)}-{dec2hex(int(n % 256), pad=2)}'
    return message


def descaleEEG(sig):  # uV to word value
    uvRange = 3952
    d = sig #% (256**2)  # this modulo messes up negative numbers
    d = d * (256**2)
    d = d / uvRange
    d = d + (256**2)/2
    return d


def ScaleEEG(e):  # word value to uV
    uvRange = 3952
    d = e #% (256**2)  # this modulo messes up negative numbers
    d = d - (256**2)/2
    d = d * uvRange
    d = d / (256**2)
    return round(d)


if __name__ == '__main__':
    num = 1234

    scal_num = descaleEEG(num)
    word = numberToWord(scal_num)
    n = wordToNum(word)
    desc_n = ScaleEEG(n)

    print(num, scal_num, word, n, desc_n)

