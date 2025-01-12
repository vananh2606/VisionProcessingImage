from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QInputDialog,
    QMessageBox,
)
import cv2 as cv
import json
from utils.control_camera import CameraThread
from utils.image_converter import ImageConverter

from gui.MainWindowUI_ui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        """Kết nối các signals với slots"""
        self.ui.Camera.clicked.connect(self.toggle_camera)
        self.ui.LoadImage.clicked.connect(self.load_image)
        self.ui.Capture.clicked.connect(self.capture_image)
        self.ui.Save.clicked.connect(self._connect_signals)

        # Thêm khởi tạo các giá trị mặc định và các loại xử lý
        self._initialize_parameters()

        # Kết nối các signals
        self._connect_signals()

    def _initialize_parameters(self):
        """Khởi tạo các tham số mặc định và options"""
        # Blur parameters
        self.blur_types = ["Gaussian Blur", "Median Blur", "Average Blur"]
        self.ui.type_blur.addItems(self.blur_types)
        self.ui.ksize.setRange(1, 31)
        self.ui.ksize.setSingleStep(2)
        self.ui.ksize.setValue(9)

        # Threshold parameters
        self.adaptive_thresh_types = ["Gaussian", "Mean"]
        self.thresh_types = [
            "Binary",
            "Binary Inverted",
            "Truncate",
            "To Zero",
            "To Zero Inverted",
        ]

        self.ui.type_adaptive_thresh.addItems(self.adaptive_thresh_types)
        self.ui.type_thresh.addItems(self.thresh_types)

        self.ui.block_size.setRange(3, 255)
        self.ui.block_size.setSingleStep(2)
        self.ui.block_size.setValue(125)

        self.ui.c_index.setRange(-255, 255)
        self.ui.c_index.setValue(9)

        # Morphological parameters
        self.morph_types = ["Erode", "Dilate", "Open", "Close"]
        self.ui.morph.addItems(self.morph_types)
        self.ui.kernel.setRange(1, 31)
        self.ui.kernel.setSingleStep(2)
        self.ui.kernel.setValue(5)

        # Contour parameters
        self.retrieval_modes_options = ["EXTERNAL", "LIST", "CCOMP", "TREE"]
        self.approximation_modes_options = ["NONE", "SIMPLE", "TC89_L1", "TC89_KCOS"]

        self.ui.retrieval_modes.addItems(self.retrieval_modes_options)
        self.ui.contour_approximation_modes.addItems(self.approximation_modes_options)

        # Detection parameters
        self.ui.area_min.setText("4500")
        self.ui.area_max.setText("9000")
        self.ui.distance.setRange(0, 100)
        self.ui.distance.setValue(15)

    def get_config(self) -> dict:
        """Get current configuration from UI parameters"""
        try:
            config = {
                # Blur parameters
                "blur": {
                    "type": self.ui.type_blur.currentText(),
                    "ksize": self.ui.ksize.value()
                },
                # Threshold parameters
                "threshold": {
                    "adaptive_type": self.ui.type_adaptive_thresh.currentText(),
                    "thresh_type": self.ui.type_thresh.currentText(),
                    "block_size": self.ui.block_size.value(),
                    "c_index": self.ui.c_index.value()
                },
                # Morphological parameters
                "morphological": {
                    "type": self.ui.morph.currentText(),
                    "kernel_size": self.ui.kernel.value()
                },
                # Contour parameters
                "contour": {
                    "retrieval_mode": self.ui.retrieval_modes.currentText(),
                    "approximation_mode": self.ui.contour_approximation_modes.currentText()
                },
                # Detection parameters
                "detection": {
                    "area_min": self.ui.area_min.text(),
                    "area_max": self.ui.area_max.text(),
                    "distance": self.ui.distance.value()
                }
            }
            return config
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error getting configuration: {str(e)}")
            return {}

    def set_config(self, config: dict):
        """Apply configuration to UI parameters"""
        try:
            # Blur parameters
            if "blur" in config:
                blur_index = self.ui.type_blur.findText(config["blur"]["type"])
                if blur_index >= 0:
                    self.ui.type_blur.setCurrentIndex(blur_index)
                self.ui.ksize.setValue(config["blur"]["ksize"])

            # Threshold parameters
            if "threshold" in config:
                adaptive_index = self.ui.type_adaptive_thresh.findText(
                    config["threshold"]["adaptive_type"]
                )
                if adaptive_index >= 0:
                    self.ui.type_adaptive_thresh.setCurrentIndex(adaptive_index)
                
                thresh_index = self.ui.type_thresh.findText(
                    config["threshold"]["thresh_type"]
                )
                if thresh_index >= 0:
                    self.ui.type_thresh.setCurrentIndex(thresh_index)
                
                self.ui.block_size.setValue(config["threshold"]["block_size"])
                self.ui.c_index.setValue(config["threshold"]["c_index"])

            # Morphological parameters
            if "morphological" in config:
                morph_index = self.ui.morph.findText(config["morphological"]["type"])
                if morph_index >= 0:
                    self.ui.morph.setCurrentIndex(morph_index)
                self.ui.kernel.setValue(config["morphological"]["kernel_size"])

            # Contour parameters
            if "contour" in config:
                retrieval_index = self.ui.retrieval_modes.findText(
                    config["contour"]["retrieval_mode"]
                )
                if retrieval_index >= 0:
                    self.ui.retrieval_modes.setCurrentIndex(retrieval_index)
                
                approximation_index = self.ui.contour_approximation_modes.findText(
                    config["contour"]["approximation_mode"]
                )
                if approximation_index >= 0:
                    self.ui.contour_approximation_modes.setCurrentIndex(approximation_index)

            # Detection parameters
            if "detection" in config:
                self.ui.area_min.setText(str(config["detection"]["area_min"]))
                self.ui.area_max.setText(str(config["detection"]["area_max"]))
                self.ui.distance.setValue(config["detection"]["distance"])

            # Process image with new configuration
            self.process_image()

        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error setting configuration: {str(e)}")

    def save_config(self, config: dict, filename: str):
        """Save configuration to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error saving configuration: {str(e)}")

    # Add these methods to handle the Load and Save button clicks
    def load_model_config(self):
        """Load configuration from JSON file"""
        try:
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setNameFilter("JSON files (*.json)")

            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                filename = file_dialog.selectedFiles()[0]
                with open(filename, 'r') as f:
                    config = json.load(f)
                self.set_config(config)
                QMessageBox.information(None, "Success", "Configuration loaded successfully")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error loading configuration: {str(e)}")

    def save_model_config(self):
        """Save current configuration to JSON file"""
        try:
            file_dialog = QFileDialog()
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            file_dialog.setNameFilter("JSON files (*.json)")
            file_dialog.setDefaultSuffix("json")

            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                filename = file_dialog.selectedFiles()[0]
                config = self.get_config()
                self.save_config(config, filename)
                QMessageBox.information(None, "Success", "Configuration saved successfully")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error saving configuration: {str(e)}")

    def _connect_signals(self):
        # Kết nối các parameters với hàm xử lý
        self.ui.type_blur.currentIndexChanged.connect(self.process_image)
        self.ui.ksize.valueChanged.connect(self.process_image)
        self.ui.type_adaptive_thresh.currentIndexChanged.connect(self.process_image)
        self.ui.type_thresh.currentIndexChanged.connect(self.process_image)
        self.ui.block_size.valueChanged.connect(self.process_image)
        self.ui.c_index.valueChanged.connect(self.process_image)
        self.ui.morph.currentIndexChanged.connect(self.process_image)
        self.ui.kernel.valueChanged.connect(self.process_image)
        self.ui.retrieval_modes.currentIndexChanged.connect(self.process_image)
        self.ui.contour_approximation_modes.currentIndexChanged.connect(
            self.process_image
        )
        self.ui.area_min.textChanged.connect(self.process_image)
        self.ui.area_max.textChanged.connect(self.process_image)
        self.ui.distance.valueChanged.connect(self.process_image)

        self.ui.LoadModel.clicked.connect(self.load_model_config)
        self.ui.Save.clicked.connect(self.save_model_config)
        # self.process_image()

    def process_image(self):
        """Xử lý ảnh dựa trên các tham số hiện tại"""
        if not hasattr(self, "current_image"):
            return

        try:
            print(self.current_image.shape)
            # Convert to grayscale
            gray = cv.cvtColor(self.current_image, cv.COLOR_BGR2GRAY)

            # Apply blur
            blur = self._apply_blur(gray)

            # Apply threshold
            thresh = self._apply_threshold(blur)

            # Apply morphological operations
            morph = self._apply_morphological(thresh)

            # Find and draw contours
            result = self._process_contours(self.current_image, morph)

            # Update UI
            self._update_ui(result, morph)

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,
                "Error",
                f"Error processing image: {str(e)}",
            )

    def _apply_blur(self, image):
        """Áp dụng blur dựa trên tham số đã chọn"""
        ksize = self.ui.ksize.value()
        if ksize % 2 == 0:
            ksize += 1

        blur_type = self.ui.type_blur.currentText()
        if blur_type == "Gaussian Blur":
            return cv.GaussianBlur(image, (ksize, ksize), 0)
        elif blur_type == "Median Blur":
            return cv.medianBlur(image, ksize)
        else:  # Average Blur
            return cv.blur(image, (ksize, ksize))

    def _apply_threshold(self, image):
        """Áp dụng threshold dựa trên tham số đã chọn"""
        block_size = self.ui.block_size.value()
        if block_size % 2 == 0:
            block_size += 1

        c = self.ui.c_index.value()

        adaptive_type = (
            cv.ADAPTIVE_THRESH_GAUSSIAN_C
            if self.ui.type_adaptive_thresh.currentText() == "Gaussian"
            else cv.ADAPTIVE_THRESH_MEAN_C
        )

        thresh_type_map = {
            "Binary": cv.THRESH_BINARY,
            "Binary Inverted": cv.THRESH_BINARY_INV,
            "Truncate": cv.THRESH_TRUNC,
            "To Zero": cv.THRESH_TOZERO,
            "To Zero Inverted": cv.THRESH_TOZERO_INV,
        }
        thresh_type = thresh_type_map[self.ui.type_thresh.currentText()]

        return cv.adaptiveThreshold(
            image, 255, adaptive_type, thresh_type, block_size, c
        )

    def _apply_morphological(self, image):
        """Áp dụng phép toán morphological dựa trên tham số đã chọn"""
        k_size = self.ui.kernel.value()
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (k_size, k_size))

        morph_type = self.ui.morph.currentText()
        if morph_type == "Erode":
            return cv.erode(image, kernel, iterations=2)
        elif morph_type == "Dilate":
            return cv.dilate(image, kernel, iterations=2)
        elif morph_type == "Open":
            return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
        else:  # Close
            return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

    def _process_contours(self, original_img, processed_img):
        """Xử lý và vẽ contours"""
        # Lấy mode cho findContours
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

        retrieval_mode = retrieval_mode_map[self.ui.retrieval_modes.currentText()]
        approximation_mode = approximation_mode_map[
            self.ui.contour_approximation_modes.currentText()
        ]

        # Tìm contours
        contours, _ = cv.findContours(processed_img, retrieval_mode, approximation_mode)

        # Xử lý contours
        min_area = float(self.ui.area_min.text())
        max_area = float(self.ui.area_max.text())
        max_distance = self.ui.distance.value()

        result_img = original_img.copy()

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            area = w * h
            if min_area <= area <= max_area and abs(w - h) < max_distance:
                cv.drawContours(result_img, [contour], -1, (0, 0, 255), 2)
                cv.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return result_img

    def _update_ui(self, original, processed):
        """Cập nhật giao diện với ảnh đã xử lý"""
        original_pixmap = ImageConverter.opencv_to_qpixmap(
            original, self.ui.OriginalImage.size()
        )
        processed_pixmap = ImageConverter.opencv_to_qpixmap(
            processed, self.ui.ProcessingImage.size()
        )

        self.ui.OriginalImage.setPixmap(original_pixmap)
        self.ui.ProcessingImage.setPixmap(processed_pixmap)

    def toggle_camera(self):
        if not hasattr(self, "camera_thread") or not self.camera_thread.isRunning():
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frameCaptured.connect(self.update_frame)
        self.camera_thread.start()
        self.Camera.setText("Stop Camera")

    def stop_camera(self):
        self.camera_thread.stop()
        self.Camera.setText("Start Camera")

    def update_frame(self, frame):
        qpixmap = ImageConverter.opencv_to_qpixmap(frame, self.ui.OriginalImage.size())
        self.ui.OriginalImage.setPixmap(qpixmap)

    def capture_image(self):
        """Chụp ảnh từ camera"""
        if hasattr(self, "camera_thread") and self.camera_thread.isRunning():
            self.current_image = self.camera_thread.get_current_frame()
            if self.current_image is not None:
                self.process_image()

    def load_image(self):
        """Load and process an image from file"""
        try:
            file_dialog = QtWidgets.QFileDialog()
            file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
            file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")

            if file_dialog.exec() == QtWidgets.QFileDialog.DialogCode.Accepted:
                file_path = file_dialog.selectedFiles()[0]
                self.original_image = cv.imread(file_path)  # Store original image
                if self.original_image is not None:
                    self.current_image = (
                        self.original_image.copy()
                    )  # Make a copy for processing
                    self.process_image()
                else:
                    QtWidgets.QMessageBox.critical(
                        None,
                        "Error",
                        "Failed to load image. Please try another file.",
                    )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,
                "Error",
                f"An error occurred while loading the image: {str(e)}",
            )
