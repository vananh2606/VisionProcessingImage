from PyQt6.QtCore import QThread, pyqtSignal
import cv2 as cv
import time

from cameras.hik import HIK
from cameras.webcam import Webcam
from cameras.base_camera import NO_ERROR


class CameraThread(QThread):
    frameCaptured = pyqtSignal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = HIK(
            config={
                "id": "0",
                "feature": "",
                # "color": True
            }
        )
        self.b_open = None
        self.frame = None
        self.running = False

    def open_camera(self):
        self.b_open = self.camera.open()
        self.b_open &= self.camera.start_grabbing()

    def grab_camera(self):
        if self.b_open:
            err, self.frame = self.camera.grab()
            return self.frame
        else:
            return None

    def run(self):
        if self.b_open:
            self.running = True
            while self.running:
                err, self.frame = self.camera.grab()

                if err != NO_ERROR:
                    print("Camera error: ", err)
                    break
                else:
                    self.frameCaptured.emit(self.frame, time.time())

                time.sleep(0.04)

    def stop_camera(self):
        print("Stop Camera")
        self.running = False

    def close_camera(self):
        self.stop_camera()
        self.camera.stop_grabbing()
        self.camera.close()
