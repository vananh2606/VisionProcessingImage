from PyQt6.QtCore import QThread, pyqtSignal
import sys
import os

sys.path.append("control_camera/")
from control_camera.cameras.hik import HIK
from control_camera.cameras.base_camera import NO_ERROR


class CameraThread(QThread):
    frameCaptured = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = HIK(
            config={
                "id": "0",
                "feature": "",
                # "color": True
            }
        )
        self.cap = None
        self.running = False

    def run(self):
        self.cap = self.camera.open()
        self.cap &= self.camera.start_grabbing()
        if self.cap:
            self.running = True
            while self.running:
                err, frame = self.camera.grab()

                if err != NO_ERROR:
                    print("Camera error : ", err)
                    break
                else:
                    self.frameCaptured.emit(frame)

    def stop(self):
        self.running = False
        self.camera.stop_grabbing()
        self.wait()
