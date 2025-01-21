import cv2 as cv
import numpy as np
from collections import namedtuple

BLOBS = namedtuple(
    "blobs",
    ["src", "dst", "mbin", "roi", "contours", "boxs", "circles", "vectors"],
    defaults=[None, None, None, None, [], [], [], []],
)

RESULT = namedtuple(
    "result",
    ["src", "dst", "mbin", "ret", "msg", "blobs"],
    defaults=[None, None, None, True, "", BLOBS()],
)


class ImageProcessor:
    def apply_blur(image, config: dict):
        """Apply blur based on selected parameters"""
        ksize = config["blur"]["ksize"]
        if ksize % 2 == 0:
            ksize += 1

        blur_type = config["blur"]["type"]
        if blur_type == "Gaussian Blur":
            return cv.GaussianBlur(image, (ksize, ksize), 0)
        elif blur_type == "Median Blur":
            return cv.medianBlur(image, ksize)
        else:  # Average Blur
            return cv.blur(image, (ksize, ksize))

    def apply_threshold(image, config: dict):
        """Apply threshold based on selected parameters"""
        block_size = config["threshold"]["block_size"]
        if block_size % 2 == 0:
            block_size += 1

        c = config["threshold"]["c_index"]
        adaptive_type = (
            cv.ADAPTIVE_THRESH_GAUSSIAN_C
            if config["threshold"]["adaptive_type"] == "Gaussian"
            else cv.ADAPTIVE_THRESH_MEAN_C
        )

        thresh_type_map = {
            "Binary": cv.THRESH_BINARY,
            "Binary Inverted": cv.THRESH_BINARY_INV,
            "Truncate": cv.THRESH_TRUNC,
            "To Zero": cv.THRESH_TOZERO,
            "To Zero Inverted": cv.THRESH_TOZERO_INV,
        }

        thresh_type = thresh_type_map[config["threshold"]["thresh_type"]]
        return cv.adaptiveThreshold(
            image, 255, adaptive_type, thresh_type, block_size, c
        )

    def apply_morphological(image, config: dict):
        """Apply morphological operation based on selected parameters"""
        k_size = config["morphological"]["kernel_size"]
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (k_size, k_size))

        morph_type = config["morphological"]["type"]
        if morph_type == "Erode":
            return cv.erode(image, kernel, iterations=5)
        elif morph_type == "Dilate":
            return cv.dilate(image, kernel, iterations=5)
        elif morph_type == "Open":
            return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
        else:  # Close
            return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

    @staticmethod
    def find_blobs(src, config: dict, b_debug=False):
        retrieval_mode_map = {
            "EXTERNAL": cv.RETR_EXTERNAL,
            "LIST": cv.RETR_LIST,
            "CCOMP": cv.RETR_CCOMP,
            "TREE": cv.RETR_TREE,
        }

        approximation_mode_map = {
            "NONE": cv.CHAIN_APPROX_NONE,
            "SIMPLE": cv.CHAIN_APPROX_SIMPLE,
            "TC89_L1": cv.CHAIN_APPROX_TC89_L1,
            "TC89_KCOS": cv.CHAIN_APPROX_TC89_KCOS,
        }

        retrieval_mode = retrieval_mode_map[config["contour"]["retrieval_mode"]]
        approximation_mode = approximation_mode_map[
            config["contour"]["approximation_mode"]
        ]

        # Convert image
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

        # Apply blur
        blur = ImageProcessor.apply_blur(gray, config)

        # Apply threshold
        mbin = ImageProcessor.apply_threshold(blur, config)

        # Apply morphological operations
        mbin = ImageProcessor.apply_morphological(mbin, config)

        cnts, _ = cv.findContours(mbin, retrieval_mode, approximation_mode)

        min_area = float(config["detection"]["area_min"])
        max_area = float(config["detection"]["area_max"])
        max_distance = config["detection"]["distance"]

        contours = []
        boxs = []

        dst = None
        if b_debug:
            dst = src.copy()

        for cnt in cnts:
            x, y, w, h = cv.boundingRect(cnt)
            area = w * h
            if min_area <= area <= max_area and abs(w - h) < max_distance:
                contours.append(cnt)
                boxs.append([x, y, w, h])

                if b_debug:
                    cv.rectangle(dst, (x, y), (x + w, y + h), (0, 255, 0), 5)

        return BLOBS(dst=dst, mbin=mbin, boxs=boxs, contours=contours)

    @staticmethod
    def find_circles(src, roi, config: dict, b_debug=False):
        """Detect circles using Hough Circle Transform"""
        cropped_image = src[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
        gray = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)

        # Áp dụng blur trước khi detect
        blur_type = config["hough_circle"]["type_blur_hough"]
        ksize = config["hough_circle"]["ksize_hough"]
        if ksize % 2 == 0:
            ksize += 1

        if blur_type == "Gaussian Blur":
            blurred = cv.GaussianBlur(gray, (ksize, ksize), 0)
        elif blur_type == "Median Blur":
            blurred = cv.medianBlur(gray, ksize)
        else:  # Average Blur
            blurred = cv.blur(gray, (ksize, ksize))

        # Map Hough method types
        hough_method_map = {
            "HOUGH_STANDARD": cv.HOUGH_STANDARD,
            "HOUGH_PROBABILISTIC": cv.HOUGH_PROBABILISTIC,
            "HOUGH_MULTI_SCALE": cv.HOUGH_MULTI_SCALE,
            "HOUGH_GRADIENT": cv.HOUGH_GRADIENT,
            "HOUGH_GRADIENT_ALT": cv.HOUGH_GRADIENT_ALT,
        }

        method = hough_method_map.get(
            config["hough_circle"]["type_hough"], cv.HOUGH_GRADIENT
        )

        # Lấy các tham số từ config
        dp = config["hough_circle"]["dp"]
        min_dist = config["hough_circle"]["min_dist"]
        param1 = config["hough_circle"]["param1"]
        param2 = config["hough_circle"]["param2"]
        min_radius = config["hough_circle"]["min_radius"]
        max_radius = config["hough_circle"]["max_radius"]

        # rows = gray.shape[0]

        circles = cv.HoughCircles(
            blurred,
            method,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        circles = [list(map(int, circle)) for circle in circles[0]]

        dst = None
        vector = None

        if b_debug:
            dst = cropped_image.copy()
            for circle in circles:
                x0, y0, r0 = circle
                cv.circle(dst, (x0, y0), r0, (0, 255, 0), 3)

        if circles is not None:
            ret = ImageProcessor.filter_circle(cropped_image.shape[:2][::-1], circles)

            if ret:
                x0, y0, _ = ret[0]
                x1, y1, _ = ret[1]

                vector = (x1 - x0, y1 - y0)
                if b_debug:
                    cv.arrowedLine(dst, (x0, y0), (x1, y1), (0, 255, 0), 3)

            # map to global image
            ret = [(x + roi[0], y + roi[1], r) for x, y, r in ret]
            vector = (vector[0] + roi[0], vector[1] + roi[1])

            return BLOBS(
                src=cropped_image, dst=dst, circles=ret, vectors=[vector], roi=roi
            )
        else:
            return BLOBS(src=cropped_image, dst=cropped_image)

    @staticmethod
    def find_result(src, config: dict):
        # find_blobs
        blobs: BLOBS = ImageProcessor.find_blobs(src, config)

        vectors = []
        circles = []

        for box in blobs.boxs:
            blob_circles: BLOBS = ImageProcessor.find_circles(src, box, config)
            circles.append(blob_circles.circles)

            if blob_circles.circles:
                vectors.append(blob_circles.vectors[0])
            else:
                vectors.append(None)

        blobs = BLOBS(
            circles=circles,
            vectors=vectors,
            contours=blobs.contours,
            boxs=blobs.boxs,
            mbin=blobs.mbin,
        )

        dst = src.copy()
        dst = ImageProcessor.draw_output(dst, blobs)

        return RESULT(src=src, dst=dst, mbin=blobs.mbin, blobs=blobs)

    def draw_output(mat, blobs: BLOBS, color=(0, 255, 0), lw=5):
        boxs = blobs.boxs
        circles = blobs.circles

        for box in boxs:
            x, y, w, h = box
            cv.rectangle(mat, (x, y), (x + w, y + h), color, lw)

        for pair_circles in circles:
            if pair_circles:
                c0, c1 = pair_circles
                cv.circle(mat, (c0[0], c0[1]), c0[2], color, lw)
                cv.circle(mat, (c1[0], c1[1]), c1[2], color, lw)
                cv.arrowedLine(mat, (c0[0], c0[1]), (c1[0], c1[1]), color, lw)
        return mat

    def decision():
        return True

    def cal_distance(pos_0, pos_1):
        x1, y1 = pos_0
        x2, y2 = pos_1
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def filter_circle(img_size, circles):
        w, h = img_size
        center = w / 2, h / 2
        min_distance = 0.0

        if len(circles) < 2:
            return None

        circle_0 = circles[0]
        c = (circles[0][0], circles[0][1])
        min_distance = ImageProcessor.cal_distance(c, center)

        for circle in circles[1:]:
            center_circle = (circle[0], circle[1])
            distance = ImageProcessor.cal_distance(center_circle, center)
            if distance < min_distance:
                min_distance = distance
                circle_0 = circle

        circles.remove(circle_0)

        center_circle_0 = (circle_0[0], circle_0[1])

        circle_1 = circles[0]
        c = (circles[0][0], circles[0][1])
        min_distance = ImageProcessor.cal_distance(c, center_circle_0)

        for circle in circles[1:]:
            center_circle = (circle[0], circle[1])
            distance = ImageProcessor.cal_distance(center_circle, center_circle_0)
            if distance < min_distance:
                min_distance = distance
                circle_1 = circle

        return tuple(map(int, circle_0)), tuple(map(int, circle_1))
