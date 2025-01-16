from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
import cv2
import numpy as np
from canvas import Canvas
from utils import ndarray2pixmap


class ImageConverter:
    """
    A utility class for converting OpenCV images to QImage and QPixmap.\n
    Methods:
    - opencv_to_qimage(cv_img): Converts an OpenCV image to QImage.
    - opencv_to_qpixmap(cv_img, scale_to_size=None): Converts an OpenCV image to QPixmap.
        Converts an OpenCV image to QImage.
        Parameters:
        - cv_img: The OpenCV image to be converted.
        Returns:
        - QImage: The converted QImage.
        Converts an OpenCV image to QPixmap.
        Parameters:
        - cv_img: The OpenCV image to be converted.
        - scale_to_size: The size to scale the QPixmap to (optional).
        Returns:
        - QPixmap: The converted QPixmap.
    - qimage_to_opencv(qimage): Converts a QImage to an OpenCV image.
    - qpixmap_to_opencv(qpixmap): Converts a QPixmap to an OpenCV image.
    """

    @staticmethod
    def opencv_to_qimage(cv_img):
        """
        Convert an OpenCV image to a QImage.\n
        Parameters:
        - cv_img: numpy.ndarray
            The OpenCV image to be converted.
        Returns:
        QImage
            The converted QImage.
        """
        # Chuyển BGR sang RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # Lấy thông số
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # Tạo QImage
        return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

    @staticmethod
    def opencv_to_qpixmap(cv_img, scale_to_size=None):
        """
        Convert an OpenCV image to a QPixmap.\n
        Args:
            cv_img (numpy.ndarray): The OpenCV image to be converted.
            scale_to_size (Optional[QSize]): The size to which the QPixmap should be scaled. Default is None.
        Returns:
            QPixmap: The converted QPixmap.
        """
        # Chuyển sang QImage
        qimage = ImageConverter.opencv_to_qimage(cv_img)

        # Tạo QPixmap
        pixmap = QPixmap.fromImage(qimage)

        # Scale nếu cần
        if scale_to_size:
            pixmap = pixmap.scaled(
                scale_to_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        return pixmap

    def smooth_label(self, label: Canvas, image: np.ndarray):
        pixmap = ImageConverter.opencv_to_qpixmap(image, label.size())

        # Calculate position to center the image
        x = (label.size().width() - pixmap.width()) // 2
        y = (label.size().height() - pixmap.height()) // 2

        # Clear the label and set new pixmap
        label.clear()
        label.setPixmap(pixmap)
        # Adjust geometry to center the image
        label.setContentsMargins(x, y, x, y)
