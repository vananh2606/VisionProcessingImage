import socket
import threading
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QListWidget,
    QLabel,
)
from PyQt6.QtCore import pyqtSignal, QObject


class Client(QObject):
    logSignal = pyqtSignal(str)
    connectedSignal = pyqtSignal(bool)

    def __init__(self, host="127.0.0.1", port=8080, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.client_socket = None
        self.connected = False

    def connect_to_server(self):
        if self.connected:
            self.logSignal.emit("Already connected to the server.")
            return

        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            self.connected = True
            self.logSignal.emit(f"Connected to server at {self.host}:{self.port}")
            self.connectedSignal.emit(True)
            threading.Thread(target=self.receive_data, daemon=True).start()
        except Exception as e:
            self.logSignal.emit(f"Failed to connect to server: {e}")

    def disconnect_from_server(self):
        if not self.connected:
            self.logSignal.emit("No connection to disconnect.")
            return

        try:
            self.client_socket.close()
            self.connected = False
            self.logSignal.emit("Disconnected from the server.")
            self.connectedSignal.emit(False)
        except Exception as e:
            self.logSignal.emit(f"Error while disconnecting: {e}")

    def send_data(self, message):
        if not self.connected:
            self.logSignal.emit("Not connected to the server. Cannot send data.")
            return

        try:
            self.client_socket.sendall(message.encode("utf-8"))
            self.logSignal.emit(f"Sent: {message}")
        except Exception as e:
            self.logSignal.emit(f"Error while sending data: {e}")

    def receive_data(self):
        while self.connected:
            try:
                data = self.client_socket.recv(1024)
                if not data:
                    self.logSignal.emit("Server closed the connection.")
                    self.disconnect_from_server()
                    break
                self.logSignal.emit(f"Received: {data.decode('utf-8')}")
            except Exception as e:
                self.logSignal.emit(f"Error while receiving data: {e}")
                self.disconnect_from_server()
                break


class ClientGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hercules-like Client")
        self.resize(400, 300)

        self.client = Client()
        self.client.logSignal.connect(self.update_log)
        self.client.connectedSignal.connect(self.update_connection_status)

        # GUI Components
        self.layout = QVBoxLayout()

        self.status_label = QLabel("Status: Disconnected")
        self.layout.addWidget(self.status_label)

        self.log_list = QListWidget()
        self.layout.addWidget(self.log_list)

        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Enter message to send")
        self.layout.addWidget(self.message_input)

        self.connect_button = QPushButton("Connect")
        self.layout.addWidget(self.connect_button)

        self.send_button = QPushButton("Send")
        self.send_button.setEnabled(False)
        self.layout.addWidget(self.send_button)

        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setEnabled(False)
        self.layout.addWidget(self.disconnect_button)

        self.setLayout(self.layout)

        # Signals
        self.connect_button.clicked.connect(self.connect_to_server)
        self.send_button.clicked.connect(self.send_message)
        self.disconnect_button.clicked.connect(self.disconnect_from_server)

    def connect_to_server(self):
        self.client.connect_to_server()

    def disconnect_from_server(self):
        self.client.disconnect_from_server()

    def send_message(self):
        message = self.message_input.text()
        if message:
            self.client.send_data(message)
            self.message_input.clear()

    def update_log(self, message):
        self.log_list.addItem(message)
        self.log_list.scrollToBottom()

    def update_connection_status(self, connected):
        self.status_label.setText(
            "Status: Connected" if connected else "Status: Disconnected"
        )
        self.connect_button.setEnabled(not connected)
        self.disconnect_button.setEnabled(connected)
        self.send_button.setEnabled(connected)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = ClientGUI()
    window.show()
    sys.exit(app.exec())
