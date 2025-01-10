from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
import cv2
import numpy as np


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

    @staticmethod
    def qimage_to_opencv(qimage):
        """
        Convert a QImage to an OpenCV image.\n
        Args:
            qimage (QImage): The QImage to be converted.
        Returns:
            numpy.ndarray: The converted OpenCV image.
        """
        # Chuyển sang numpy array
        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()
        image_format = qimage.format()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        cv_img = np.array(ptr).reshape(height, width, 4)

        return cv_img

    @staticmethod
    def qpixmap_to_opencv(qpixmap):
        """
        Convert a QPixmap to an OpenCV image.\n
        Args:
            qpixmap (QPixmap): The QPixmap to be converted.
        Returns:
            numpy.ndarray: The converted OpenCV image.
        """
        # Chuyển sang QImage
        qimage = qpixmap.toImage()

        # Chuyển sang OpenCV
        cv_img = ImageConverter.qimage_to_opencv(qimage)

        return cv_img
