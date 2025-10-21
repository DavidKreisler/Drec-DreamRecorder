import socket
import threading
import time
import mne
import numpy as np

from scripts.Utils.Encode_Decode_Utils import signal_to_hex


class HD_Server_Sim:
    def __init__(self):
        # Server settings
        self.HOST = '0.0.0.0'
        self.PORT = 8000
        self.clients = []

        # Global variables for EEG data
        self.eeg_data = None
        self.sampling_rate = 1024 # 256 Hz # CHANGE SF FOR FASTER TRANSMITION
        self.current_sample = 0  # Pointer to track the current sample being streamed
        self.streaming = False

    def broadcast_data(self, data):
        """Sends data to all connected clients."""
        disconnected_clients = []
        for client in self.clients:
            try:
                client.sendall(data.encode('utf-8'))
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                # Mark the client as disconnected
                disconnected_clients.append(client)
                print("[ERROR] Failed to send data. Removing disconnected client.")

        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.remove(client)

    def stream_data(self):
        """Streams 256 samples per second to simulate real-time EEG recording."""
        # CHANGE self.sampling_rate to 1024 FOR FASTER TRANSMITION #
        n_samples = self.eeg_data.shape[1]  # Total number of samples per channel
        chunk_size = 1  # Number of samples to send per second
        interval = chunk_size / self.sampling_rate  # Send chunks every second

        while True:
            if len(self.clients) > 0 and self.streaming:
                # Calculate the range of samples to send
                start = self.current_sample
                end = self.current_sample + chunk_size

                if end > n_samples:
                    print(f'[INFO] End of file reached. stopping stream.')
                    self.streaming = False
                else:
                    chunk = self.eeg_data[:, start:end]
                    self.current_sample = end
                    if self.current_sample % (n_samples / 10) == 0:
                        print(f'{(self.current_sample / n_samples) * 100}% done')

                # convert the chunks into a message
                accumulated_message = ''
                for idx in range(0, len(chunk[0])):
                    sigl = chunk[0][idx]
                    sigr = chunk[1][idx]
                    accumulated_message += signal_to_hex(sigl, sigr) + '\r\n'

                self.broadcast_data(accumulated_message)

            else:
                time.sleep(1)

            time.sleep(interval)  # Wait for 1 second before sending the next chunk

    def handle_client(self, client_socket, address):
        """Handles communication with a single client."""
        print(f"[NEW CONNECTION] {address} connected.")
        self.clients.append(client_socket)
        try:
            # Wait for the "HELLO" message to start streaming
            while True:
                message = client_socket.recv(1024).decode('utf-8').strip()
                if message == "HELLO":
                    print(f"[HELLO RECEIVED] Starting stream for {address}")
                    self.streaming = True
                    time.sleep(1)
                    break
        except (ConnectionResetError, BrokenPipeError):
            print(f"[DISCONNECT] {address} disconnected.")

        finally:
            while self.streaming:
                time.sleep(1)
            # Ensure the client is removed from the list
            if client_socket in self.clients:
                self.clients.remove(client_socket)
            client_socket.close()

    def start_server(self):
        """Sets up the server and handles incoming connections."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.HOST, self.PORT))
        server_socket.listen()
        print(f"[LISTENING] Server is listening on {self.HOST}:{self.PORT}")

        # Start the data streaming thread
        threading.Thread(target=self.stream_data, daemon=True).start()

        try:
            while True:
                client_socket, address = server_socket.accept()
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket, address), daemon=True)
                client_thread.start()
        except KeyboardInterrupt:
            print("[SHUTDOWN] Server is shutting down.")
            for client in self.clients:
                client.close()
            server_socket.close()

    # ------------------------
    # EDF Helper Methods (should be functions somewhere)
    # ------------------------
    def combine_raw_instances(self, raw1, raw2):
        """Combine two Raw instances into one."""
        # Extract data and metadata from each Raw instance
        data1 = raw1.get_data()
        data2 = raw2.get_data()

        physical_min = -1975  # np.ceil(max(min(min(signals_reformatted[0]), min(signals_reformatted[1])), -10000))
        physical_max = 1975  # np.floor(min(max(max(signals_reformatted[0]), max(signals_reformatted[1])), 10000))

        # transform the signal
        data1 = np.clip(data1, physical_min, physical_max)
        data2 = np.clip(data2, physical_min, physical_max)

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

    def load_edf(self, file_path=None):
        """Loads an EDF file using MNE and extracts EEG data."""
        if file_path is None:
            print('No path specified! No EDF file is loaded. Bradcasting will not be possible.')
            return
        try:
            # Load the EDF file
            edf_file_path_L = f'{file_path}EEG L.edf'  # Replace with your EDF file path
            edf_file_path_R = f'{file_path}EEG R.edf'  # Replace with your EDF file path
            x = f'{file_path}'
            raw_eegl = mne.io.read_raw_edf(edf_file_path_L, preload=True)
            raw_eegr = mne.io.read_raw_edf(edf_file_path_R, preload=True)

            eeg = self.combine_raw_instances(raw_eegl, raw_eegr)
            self.eeg_data = eeg.get_data(picks='eeg', units='uV')

            print(f"[EDF LOADED] Loaded EEG data with shape {self.eeg_data.shape}")

        except Exception as e:
            print(f"[ERROR] Failed to load EDF file: {e}")
            self.eeg_data = None


if __name__ == "__main__":

    server = HD_Server_Sim()
    server.load_edf('path/to/recording/folder/') # like C:/Drec/recordings/2024 12 9 - 21 52 57/2024 12 9 - 21 52 57/
                                                 # ein zmax recording von einer deiner naechte

    # Start the server
    server.start_server()
