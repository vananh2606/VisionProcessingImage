import cv2
from PyQt6.QtCore import QThread, pyqtSignal


class CameraThread(QThread):
    frameCaptured = pyqtSignal(object)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.cap = None
        self.running = False

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.running = True
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                self.frameCaptured.emit(frame)
        self.cap.release()

    def stop(self):
        self.running = False
        self.wait()
