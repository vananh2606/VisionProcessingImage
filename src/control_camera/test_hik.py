# import sys
# sys.path.append("src")

from cameras.hik import HIK
from cameras.base_camera import NO_ERROR

import cv2 as cv
import numpy as np
import time
import math

from collections import deque

cv.namedWindow("Camera", cv.WINDOW_FREERATIO)

config = {
    "id": "0",
    "feature": "",
    # "color": True
}

camera = HIK(config=config)

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
            # FPS
            counter.append(dt)
            if len(counter) == N_FRAME:
                t = sum(counter)
                fps = N_FRAME / max(t, 0.01)
                counter.popleft()

                text = "FPS : %d" % fps
                # cv.putText(mat, text, (200, 200), 1, 5, (0, 255, 0), 5)

            # Chụp lưu lại nhiều ảnh ở thư mục resource
            # key = cv.waitKey(1)
            # if key == ord("c"):
            #     image_count += 1
            #     filename = f"resource/capture_{image_count}.png"  # Tạo tên file
            #     cv.imwrite(filename, mat)  # Lưu ảnh
            #     print(f"Đã chụp ảnh và lưu vào '{filename}'")

            cv.imshow("Camera", mat)

            if cv.waitKey(5) == ord("q"):
                break
