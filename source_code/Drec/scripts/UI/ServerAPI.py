from typing import List

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from flask import Flask, request, jsonify
from werkzeug.serving import make_server

from scripts.Utils.Logger import Logger

# ---------------------------------------------------------------------
# SleepRecorderAPI Class
# ---------------------------------------------------------------------
class SleepRecorderAPI(QObject):
    start_signal = pyqtSignal(bool)
    stop_signal = pyqtSignal(bool)
    start_scoring_signal = pyqtSignal(bool)
    stop_scoring_signal = pyqtSignal(bool)
    start_webhook_signal = pyqtSignal(bool)
    stop_webhook_signal = pyqtSignal(bool)
    set_signaltype_signal = pyqtSignal(list)
    set_scoring_delay_signal = pyqtSignal(int)
    set_webhookip_signal = pyqtSignal(str)
    quit_signal = pyqtSignal(bool)

    def __init__(self):
        QObject.__init__(self)  # Initialize QObject
        self._is_running = True

    # Methods below just emit the signals (like the CLI commands did)
    def start(self):
        Logger().log("start signal emitted", 'DEBUG')
        self.start_signal.emit(True)
        self.start_scoring_signal.emit(True)
        self.start_webhook_signal.emit(True)
        return {"message": "Started recording, scoring, and webhook."}

    def stop(self):
        Logger().log("stop signal emitted", 'DEBUG')
        self.stop_signal.emit(True)
        return {"message": "Stopped all processes."}

    def start_recording(self):
        self.start_signal.emit(True)
        return {"message": "Started recording."}

    def start_scoring(self):
        self.start_scoring_signal.emit(True)
        return {"message": "Started scoring."}

    def stop_scoring(self):
        self.stop_scoring_signal.emit(True)
        return {"message": "Stopped scoring."}

    def start_webhook(self):
        self.start_webhook_signal.emit(True)
        return {"message": "Started webhook."}

    def stop_webhook(self):
        self.stop_webhook_signal.emit(True)
        return {"message": "Stopped webhook."}

    def set_signaltype(self, numbers):
        try:
            numbers = [int(n) for n in numbers]
            self.set_signaltype_signal.emit(numbers)
            return {"message": f"Signal types set to {numbers}"}
        except ValueError:
            return {"error": "Invalid signal numbers."}, 400

    def set_scoring_delay(self, delay):
        try:
            val = int(delay)
            self.set_scoring_delay_signal.emit(val)
            return {"message": f"Scoring delay set to {val} epochs."}
        except ValueError:
            return {"error": f'"{delay}" is not a valid integer.'}, 400

    def set_webhook_ip(self, address):
        self.set_webhookip_signal.emit(address)
        return {'message': 'Webhook ip changed'}

    def quit(self):
        Logger().log("Quit signal emitted", 'INFO')
        self.quit_signal.emit(True)
        self._is_running = False
        return {"message": "Server quitting..."}

# ---------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------
class FlaskApp:
    def __init__(self):
        self.api = SleepRecorderAPI()
        self.app = Flask(__name__)
        self._setup_routes()

    def run(self):
        print("FlaskThread started — launching Flask server")
        self.app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)

    def _setup_routes(self):
        app = self.app
        api = self.api

        @app.route("/start", methods=["POST"])
        def start():
            return jsonify(api.start())

        @app.route("/stop", methods=["POST"])
        def stop():
            return jsonify(api.stop())

        @app.route("/start_recording", methods=["POST"])
        def start_recording():
            return jsonify(api.start_recording())

        @app.route("/start_scoring", methods=["POST"])
        def start_scoring():
            return jsonify(api.start_scoring())

        @app.route("/stop_scoring", methods=["POST"])
        def stop_scoring():
            return jsonify(api.stop_scoring())

        @app.route("/start_webhook", methods=["POST"])
        def start_webhook():
            return jsonify(api.start_webhook())

        @app.route("/stop_webhook", methods=["POST"])
        def stop_webhook():
            return jsonify(api.stop_webhook())

        @app.route("/set_signaltype", methods=["POST"])
        def set_signaltype():
            data = request.json or {}
            numbers = data.get("numbers", [])
            return jsonify(api.set_signaltype(numbers))

        @app.route("/set_scoring_delay", methods=["POST"])
        def set_scoring_delay():
            data = request.json or {}
            delay = data.get("delay")
            return jsonify(api.set_scoring_delay(delay))

        @app.route("/websocket_ip", methods=["POST"])
        def set_websocket_ip():
            data = request.json or {}
            ip = data.get("ip")
            return jsonify(api.set_webhook_ip(ip))

        @app.route("/quit", methods=["POST"])
        def quit():
            resp = api.quit()
            return jsonify(resp)

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "running" if api._is_running else "stopped"})

# ---------------------------------------------------------------------
# Flask Thread Wrapper
# ---------------------------------------------------------------------
class FlaskThread(QThread):
    def __init__(self):
        super().__init__()
        self.fl_app = FlaskApp()
        self.comm_if = self.fl_app.api
        self.app = self.fl_app.app
        self.server = make_server('127.0.0.1', 5001, self.app)
        self.ctx = self.app.app_context()
        self.ctx.push()

    def run(self):
        print("Running Flask server (non-blocking mode)")
        while True:
            self.server.handle_request()

    def stop(self):
        self.comm_if.stop()
        self.quit()


if __name__ == "__main__":
    server_thread = FlaskThread()
    server_thread.start()
