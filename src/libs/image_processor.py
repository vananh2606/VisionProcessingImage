import cv2 as cv
import numpy as np
from collections import namedtuple

BLOBS = namedtuple(
    "blobs",
    [
        "src",
        "dst",
        "mbin",
        "roi",
        "contours",
        "boxes",
        "circles",
        "vectors",
        "centers",
        "aligments",
    ],
    defaults=[None, None, None, None, [], [], [], [], [], []],
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

        rows = config.get("rows", 4)
        columns = config.get("columns", 5)

        anchor_rois = []
        # Lấy danh sách ROI từ config nếu có
        shapes: dict = config["shapes"]
        if shapes:
            anchor_rois = [shapes[i]["box"] for i in shapes]

        sorted_boxes = [None] * (
            rows * columns
        )  # List các boxes sau khi sorting dua vao anchor_rois

        sorted_contours = [None] * (rows * columns)
        # boxes = []
        dst = None

        for cnt in cnts:
            x, y, w, h = cv.boundingRect(cnt)
            area = w * h
            if min_area <= area <= max_area and abs(w - h) < max_distance:
                # Tính tâm của box
                center_x = x + w // 2
                center_y = y + h // 2

                # Kiểm tra box thuộc ROI nào
                for roi_index, roi in enumerate(anchor_rois):
                    rx, ry, rw, rh = roi
                    if (rx <= center_x <= rx + rw) and (ry <= center_y <= ry + rh):
                        sorted_boxes[roi_index] = [x, y, w, h]
                        sorted_contours[roi_index] = cnt
                        break

        if b_debug:
            dst = src.copy()
            for i, box in enumerate(sorted_boxes):
                if box is None:
                    continue
                x, y, w, h = box
                color = (0, 255, 0) if roi_index is None else (0, 0, 255)
                cv.rectangle(dst, (x, y), (x + w, y + h), color, 5)
                cv.putText(
                    dst,
                    f"BLOB: {i}",
                    (x, y - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )

        return BLOBS(dst=dst, mbin=mbin, boxes=sorted_boxes, contours=sorted_contours)

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

        dst = None

        if circles is not None:
            circles = [list(map(int, circle)) for circle in circles[0]]

            vector = None
            center = None

            if b_debug:
                dst = cropped_image.copy()
                for circle in circles:
                    x0, y0, r0 = circle
                    cv.circle(dst, (x0, y0), r0, (0, 255, 0), 3)

            ret = ImageProcessor.filter_circle(cropped_image.shape[:2][::-1], circles)

            if ret:
                x0, y0, _ = ret[0]
                x1, y1, _ = ret[1]

                vector = (x1 - x0, y1 - y0)
                center = (x0, y0)
                if b_debug:
                    cv.arrowedLine(dst, (x0, y0), (x1, y1), (0, 255, 0), 3)

                # map to global image
                ret = [(x + roi[0], y + roi[1], r) for x, y, r in ret]
                vector = (vector[0] + roi[0], vector[1] + roi[1])
                center = (center[0] + roi[0], center[1] + roi[1])

            return BLOBS(
                src=cropped_image,
                dst=dst,
                roi=roi,
                circles=ret,
                vectors=[vector],
                centers=[center],
            )
        else:
            return BLOBS(src=cropped_image, dst=cropped_image)

    def get_origin_from_config(config: dict):
        return config["blobs"]

    def cal_angle(vector: list):
        x, y = vector
        # Calculate angle using arctan2 (handles all quadrants correctly)
        angle_rad = np.arctan2(y, x)
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def cal_aligment(origin: dict, current: dict) -> tuple[int, int, float]:
        if current["center"] is None or current["vector"] is None:
            return None

        x0 = origin["center"][0]
        y0 = origin["center"][1]
        angle0 = ImageProcessor.cal_angle(origin["vector"])

        x = current["center"][0]
        y = current["center"][1]
        angle = ImageProcessor.cal_angle(current["vector"])

        dx = x - x0
        dy = y - y0
        da = angle - angle0

        if da > 180:
            da -= 360
        elif da < -180:
            da += 360

        return dx, dy, da

    @staticmethod
    def find_result(src, config: dict, b_origin=False):
        """
        Find Blobs, Find Circles, Calculate aligment
        """
        # find_blobs
        blobs: BLOBS = ImageProcessor.find_blobs(src, config)

        circles = [None] * len(blobs.boxes)
        vectors = [None] * len(blobs.boxes)
        centers = [None] * len(blobs.boxes)

        for i, box in enumerate(blobs.boxes):
            if box is None:
                continue
            blob_circles: BLOBS = ImageProcessor.find_circles(src, box, config)
            circles[i] = blob_circles.circles

            if blob_circles.circles:
                vectors[i] = blob_circles.vectors[0]
                centers[i] = blob_circles.centers[0]

        if b_origin:
            aligments = None
            msg = ""
        else:
            origins = ImageProcessor.get_origin_from_config(config)
            currents = {
                str(i): {
                    "center": centers[i],
                    "vector": vectors[i],
                }
                for i in range(len(centers))
            }

            aligments = {
                key: ImageProcessor.cal_aligment(origins[key], currents[key])
                for key in origins
            }

            msg = ImageProcessor.decision(aligments)

        blobs = BLOBS(
            mbin=blobs.mbin,
            contours=blobs.contours,
            boxes=blobs.boxes,
            circles=circles,
            vectors=vectors,
            centers=centers,
            aligments=aligments,
        )

        dst = src.copy()
        dst = ImageProcessor.draw_output(dst, blobs)

        return RESULT(src=src, dst=dst, mbin=blobs.mbin, msg=msg, blobs=blobs)

    def draw_output(mat, blobs: BLOBS, lw=5):
        boxes = blobs.boxes
        circles = blobs.circles
        vectors = blobs.vectors
        centers = blobs.centers
        aligments = blobs.aligments

        color_box, color_circles, color_aligments = (
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
        )

        for i, box in enumerate(boxes):
            if box is None:
                continue
            x, y, w, h = box
            cv.rectangle(mat, (x, y), (x + w, y + h), color_box, lw)
            cv.putText(
                mat, f"Boxes: {i}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, color_box, lw
            )

        for i, pair_circles in enumerate(circles):
            if pair_circles:
                c0, c1 = pair_circles
                cv.circle(mat, (c0[0], c0[1]), c0[2], color_circles, lw)
                cv.circle(mat, (c1[0], c1[1]), c1[2], color_circles, lw)
                cv.arrowedLine(mat, (c0[0], c0[1]), (c1[0], c1[1]), color_circles, lw)

                if aligments is not None:
                    dx, dy, da = aligments[str(i)]
                    cv.putText(
                        mat,
                        f"{dx},{dy},{da:.2f}",
                        (c0[0], c0[1]),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        color_aligments,
                        2,
                        cv.LINE_AA,
                    )

        return mat

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

    def decision(aligments: dict) -> str:
        # if aligments is None:
        #     return ""
        msg = []
        for aligment in aligments.values():
            if aligment is None:
                msg.append(f"None")
            else:
                dx, dy, da = aligment
                msg.append(f"{dx},{dy},{da:.2f}")
        return "_".join(msg)
