import socket
import threading
from PyQt6.QtCore import pyqtSignal, QObject


class Server(QObject):
    logInfoSignal = pyqtSignal(str)
    onTriggerSignal = pyqtSignal(socket.socket)

    HOST = "127.0.0.1"
    PORT = 8080

    def __init__(self, parent=None):
        super().__init__(parent)
        self.server_socket = None
        self.server_running = False

    def run_server(self):
        self.server_running = False
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.HOST, self.PORT))
        self.server_socket.listen(
            1
        )  # Chỉ cho phép một kết nối client tại một thời điểm
        self.logInfoSignal.emit(f"Server is running on {self.HOST}:{self.PORT}...")
        self.server_running = True

        try:
            while True:
                self.logInfoSignal.emit("Waiting for client connection...")
                client_socket, client_address = self.server_socket.accept()
                self.logInfoSignal.emit(f"Connected from {client_address}")

                # Gọi hàm xử lý client
                # threading.Thread(
                #     target=self.handle_client,
                #     args=(client_socket, client_address),
                #     daemon=True,
                # ).start()

                self.handle_client(client_socket, client_address)

        except OSError:
            self.logInfoSignal.emit("Server is stopped.")
        finally:
            self.server_socket.close()
            self.server_running = False

    # Tắt server
    def stop_server(self):
        if self.server_running:
            self.server_socket.close()
            self.server_running = False
            self.logInfoSignal.emit("Server is stopped.")

    # Hàm xử lý client
    def handle_client(self, client_socket, client_address):
        try:
            data = client_socket.recv(1024)
            if not data:
                self.logInfoSignal.emit(f"Client {client_address} disconnected.")
            else:
                response = data.decode("utf-8")
                self.logInfoSignal.emit(f"Received from {client_address}: {response}")

                if response.lower() == "check":
                    self.onTriggerSignal.emit(client_socket)

        except ConnectionResetError:
            self.logInfoSignal.emit(f"Connection with {client_address} lost.")
        finally:
            # client_socket.close()
            self.logInfoSignal.emit(f"Connection closed for {client_address}.")

    # Hàm gửi lại output về phía client
    def send_message(self, client: socket.socket, msg: str):
        try:
            client.sendall(msg.encode("utf-8"))
            client.close()
            self.logInfoSignal.emit(f"Sent to client: {msg}")
        except BrokenPipeError:
            self.logInfoSignal.emit("Error: Unable to send. Client disconnected.")
