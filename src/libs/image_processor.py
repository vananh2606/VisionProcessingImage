import cv2 as cv
import numpy as np


class ImageProcessor:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
    def process_contours(original_img, processed_img, config: dict):
        """Process and draw contours"""
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

        contours, _ = cv.findContours(processed_img, retrieval_mode, approximation_mode)

        min_area = float(config["detection"]["area_min"])
        max_area = float(config["detection"]["area_max"])
        max_distance = config["detection"]["distance"]

        result_img = original_img.copy()

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            area = w * h
            if min_area <= area <= max_area and abs(w - h) < max_distance:
                cv.drawContours(result_img, [contour], -1, (0, 0, 255), 5)
                cv.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 5)

        return result_img
