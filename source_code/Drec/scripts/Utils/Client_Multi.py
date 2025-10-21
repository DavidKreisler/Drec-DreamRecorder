import socket

import mne
import numpy as np

HOST = '127.0.0.1'
PORT = 8000


def combine_raw_instances(raw1, raw2):
    """Combine two Raw instances into one."""
    # Extract data and metadata from each Raw instance
    data1 = raw1.get_data()
    data2 = raw2.get_data()

    # Check if sampling rates match
    assert raw1.info['sfreq'] == raw2.info['sfreq'], "Sampling rates must match!"
    assert raw1.times.shape == raw2.times.shape, "Time bases must align!"

    # Combine data (stack channels)
    combined_data = np.vstack((data1, data2))

    # Combine channel names and types
    combined_ch_names = raw1.ch_names + raw2.ch_names
    combined_ch_types = [raw1.get_channel_types()] + [raw2.get_channel_types()]
    combined_ch_types = [t for sublist in combined_ch_types for t in sublist]  # Flatten list

    # Create a new Info object
    combined_info = mne.create_info(ch_names=combined_ch_names, sfreq=raw1.info['sfreq'],
                                    ch_types=combined_ch_types)

    # Create a new Raw object
    combined_raw = mne.io.RawArray(combined_data, combined_info)
    return combined_raw


def load_edf(file_path='C:/coding/git/dreamento/dreamento-online/source_code/Drec/recordings/2024 12 9 - 21 52 57/2024 12 9 - 21 52 57/'):
    """Loads an EDF file using MNE and extracts EEG data."""
    try:
        # Load the EDF file
        edf_file_path_L = f'{file_path}EEG L.edf'  # Replace with your EDF file path
        edf_file_path_R = f'{file_path}EEG R.edf'  # Replace with your EDF file path
        x = f'{file_path}'
        raw_eegl = mne.io.read_raw_edf(edf_file_path_L, preload=True)
        raw_eegr = mne.io.read_raw_edf(edf_file_path_R, preload=True)

        eeg = combine_raw_instances(raw_eegl, raw_eegr)
        eeg_data = eeg.get_data(picks='eeg')

        print(f"[EDF LOADED] Loaded EEG data with shape {eeg_data.shape}")

    except Exception as e:
        print(f"[ERROR] Failed to load EDF file: {e}")
        eeg_data = None

    return eeg_data


def descaleEEG( sig):  # uV to word value
    uvRange = 3952
    d = sig * 65536
    d = d / uvRange
    d = d + 32768
    return d


def numberToWord(n):
    if n <= 256:
        return f'00-{dec2hex(n, pad=2)}'
    message = f'{dec2hex(int(n / 256), pad=2)}-{dec2hex(int(n % 256), pad=2)}'
    return message


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
          f"00-00-00-" \

    return buf


def send_to_server():
    data = load_edf()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))
        print("[CONNECTED] Connected to the server.")
        try:
            for idx in range(0, len(data[0])):
                r = data[0][idx]
                l = data[1][idx]
                message = signal_to_hex(l, r) + '\r\n'
                client_socket.send(message.encode('utf-8'))
        except KeyboardInterrupt:
            print("[DISCONNECTING] Closing connection.")
        finally:
            client_socket.close()


def read_from_server():
    all_data = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))
        print("[CONNECTED] Connected to the server.")
        client_socket.sendall(b"HELLO\n")  # Send the HELLO message
        packet_no = 0
        try:
            while True:
                data = client_socket.recv(65000)  # Buffer size
                if not data:
                    continue
                packet_no += 1
                if packet_no % 10000 == 0:
                    print(f'{packet_no / 10000} * 10k packets received')
                #print(data.decode('utf-8'))
        except KeyboardInterrupt:
            print("[DISCONNECTING] Closing connection.")
        finally:
            client_socket.close()


if __name__ == "__main__":
    read_from_server()
