import cv2 as cv
import time
from collections import deque

camera = cv.VideoCapture(0)
counter = deque()
N_FRAME = 100
image_count = 0

while True:
    t0 = time.time()
    ret, mat = camera.read()
    dt = time.time() - t0

    # FPS
    counter.append(dt)
    if len(counter) == N_FRAME:
        t = sum(counter)
        fps = N_FRAME / max(t, 0.01)
        counter.popleft()

        text = "FPS : %d" % fps
        # cv.putText(mat, text, (200, 200), 1, 5, (0, 255, 0), 5)

    # Chụp lưu lại nhiều ảnh ở thư mục resource
    key = cv.waitKey(1)
    if key == ord("c"):
        image_count += 1
        filename = f"resource/test/capture_{image_count}.png"  # Tạo tên file
        cv.imwrite(filename, mat)  # Lưu ảnh
        print(f"Đã chụp ảnh và lưu vào '{filename}'")

    cv.imshow("Camera", mat)

    if cv.waitKey(5) == ord("q"):
        break
