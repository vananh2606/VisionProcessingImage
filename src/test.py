from cameras.webcam import Webcam

config = {"id": 0, "feature": ""}
camera = Webcam(config)

camera.open()
camera.start_grabbing()
err, mat = camera.grab()
print(mat.shape)
camera.stop_grabbing()
camera.close()

camera = Webcam(config)
camera.open()
camera.start_grabbing()
err, mat = camera.grab()
print(mat.shape)
camera.stop_grabbing()
camera.close()
