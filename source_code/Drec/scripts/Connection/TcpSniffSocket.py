import datetime
import threading
import time

from scapy.all import sniff, IP, TCP
from queue import Queue

from scripts.Connection.TcpConnection import Connection
from scripts.Utils.Logger import Logger


class TcpSniffSocket:
    def __init__(self):
        self.sniffer_thread = None
        self.connections = {}
        self.data_queue = Queue()
        self.stop_sniffing = True

    def connect(self):
        Logger().log('starting TcpSniffSocket')
        self.sniffer_thread = threading.Thread(target=self._start_sniffer)
        self.sniffer_thread.daemon = True
        self.sniffer_thread.start()
        self.stop_sniffing = False

    def read_live(self):
        while True:
            if not self.data_queue.empty():
                print(self.data_queue.get())

    def read_one_line(self):
        if self.stop_sniffing:
            return ''
        start_time = datetime.datetime.now()
        while self.data_queue.empty():
            if datetime.datetime.now()-start_time > datetime.timedelta(seconds=5):
                Logger().log('no data available at queue. returning nothing', 'debug')
                return ''
            time.sleep(0.1)
        return self.data_queue.get().decode("utf-8")

    def _start_sniffer(self):
        try:
            # Start sniffing traffic on the localhost interface
            sniff(filter=f"tcp port 8000",
                  iface="\\Device\\NPF_Loopback",
                  prn=self._sniffer_callback,
                  stop_filter=lambda p: self.stop_sniffing,
                  store=False)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping sniffer.")
        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")

    def _sniffer_callback(self, packet):
        # Check if the packet has the necessary layers (IP and TCP)
        if IP in packet and TCP in packet:
            ip_layer = packet[IP]
            tcp_layer = packet[TCP]

            # Create a unique connection identifier (tuple of src, sport, dst, dport)
            conn_id = (ip_layer.src, tcp_layer.sport, ip_layer.dst, tcp_layer.dport)
            reverse_conn_id = (ip_layer.dst, tcp_layer.dport, ip_layer.src, tcp_layer.sport)

            # Handle SYN packets to start tracking a new connection
            if tcp_layer.flags == "S":
                # con = Connection(ip_layer.src, tcp_layer.sport, ip_layer.dst, tcp_layer.dport, tcp_layer.seq)
                # connections[conn_id] = con

                rev_con = Connection(ip_layer.dst, tcp_layer.dport, ip_layer.src, tcp_layer.sport, None)
                self.connections[reverse_conn_id] = rev_con

                print(f"[INFO] New connection started: {conn_id}")

            # handle payloads
            for id in [self.connections[key].get_id() for key in self.connections.keys()]:
                active_conn = id  # Handle which direction of the connection to track
                # Determine packet direction
                if tcp_layer.dport == id[1]:  # Client -> Server
                    continue
                elif tcp_layer.sport == id[1]:  # Server -> Client
                    pass
                else:  # unknown
                    continue

                # Add the payload to the connection data, sorted by sequence number
                seq = tcp_layer.seq
                payload = bytes(tcp_layer.payload)
                if payload != b'':
                    self.connections[active_conn].parse_payload(seq, payload, self.data_queue)
                    # print(f"[+] Packet captured in connection {active_conn}: Payload length: {len(payload)} bytes, Seq: {seq}, Payload {payload.decode()}")

                # Handle FIN or RST packets to close the connection
                if tcp_layer.flags == "F" or tcp_layer.flags == "FA" or tcp_layer.flags == "R" or tcp_layer.flags == "RA":
                    print(f"[INFO] Connection closed: {active_conn}")

                    # Clean up the connection data
                    del self.connections[active_conn]

    def stop(self):
        Logger().log('stopping TcpSniffSocket')
        self.stop_sniffing = True


if __name__ == '__main__':
    sock = TcpSniffSocket()
    sock.connect()
    i = 0
    while True:
        if i > 1000:
            sock.stop()
            break
        else:
            print(sock.read_one_line())
            i += 1

    print('reading again')
    while True:
        print(sock.read_one_line())

