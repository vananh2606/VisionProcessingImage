# import sys
# sys.path.append("src")

from cameras.webcam import Webcam
from cameras.base_camera import NO_ERROR

import cv2
import time
import math

from collections import deque

cv2.namedWindow("Camera", cv2.WINDOW_FREERATIO)

config = {
    "id": "0"
}

camera = Webcam(config=config)

print(camera.get_model_name())
print("Error: ", camera.get_error())

b = camera.open()
b &= camera.start_grabbing()

counter = deque()
N_FRAME = 100

if b:
    while True:
        t0 = time.time()
        err, mat = camera.grab()
        dt = time.time() - t0

        if err != NO_ERROR:
            print("Camera error : ", err)
            break
        else:
            counter.append(dt)
            if len(counter) == N_FRAME:
                t = sum(counter)
                fps = N_FRAME / max(t, 0.01)
                counter.popleft()

                text = "FPS : %d" % fps
                cv2.putText(mat, text, (200, 200), 1, 5, (0, 255, 0), 5)

            cv2.imshow("Camera", mat)
            if cv2.waitKey(5) == ord("q"):
                break