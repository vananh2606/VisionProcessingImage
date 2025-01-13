from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QLabel,
)
import cv2 as cv
import numpy as np
import json
import threading
import time
import os

from utils.control_camera import CameraThread
from utils.image_converter import ImageConverter

from gui.MainWindowUI_ui import Ui_MainWindow
from PyQt6.QtCore import pyqtSignal


class MainWindow(QMainWindow):
    showResultSignal = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        """Kết nối các signals với slots"""
        self.ui.Camera.clicked.connect(self.toggle_camera)
        self.ui.LoadImage.clicked.connect(self.load_image)
        self.ui.Capture.clicked.connect(self.capture_image)
        self.ui.AddModel.clicked.connect(self.add_model_config)
        self.ui.SaveModel.clicked.connect(self.save_model_config)
        self.ui.model.currentIndexChanged.connect(self.load_model_config)

        self.showResultSignal.connect(self._update_ui)
        self._update_model_list()  # Initialize model list

        # Thêm khởi tạo các giá trị mặc định và các loại xử lý
        self._initialize_parameters()
        self.b_stop = False
        self.camera_thread = None
        self.current_image = None

        # Kết nối các signals
        self.start_loop_process()

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
                    "ksize": self.ui.ksize.value(),
                },
                # Threshold parameters
                "threshold": {
                    "adaptive_type": self.ui.type_adaptive_thresh.currentText(),
                    "thresh_type": self.ui.type_thresh.currentText(),
                    "block_size": self.ui.block_size.value(),
                    "c_index": self.ui.c_index.value(),
                },
                # Morphological parameters
                "morphological": {
                    "type": self.ui.morph.currentText(),
                    "kernel_size": self.ui.kernel.value(),
                },
                # Contour parameters
                "contour": {
                    "retrieval_mode": self.ui.retrieval_modes.currentText(),
                    "approximation_mode": self.ui.contour_approximation_modes.currentText(),
                },
                # Detection parameters
                "detection": {
                    "area_min": self.ui.area_min.text(),
                    "area_max": self.ui.area_max.text(),
                    "distance": self.ui.distance.value(),
                },
            }
            return config
        except Exception as e:
            QMessageBox.critical(
                None, "Error", f"Error getting configuration: {str(e)}"
            )
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
                    self.ui.contour_approximation_modes.setCurrentIndex(
                        approximation_index
                    )

            # Detection parameters
            if "detection" in config:
                self.ui.area_min.setText(str(config["detection"]["area_min"]))
                self.ui.area_max.setText(str(config["detection"]["area_max"]))
                self.ui.distance.setValue(config["detection"]["distance"])
            # Process image with new configuration
        except Exception as e:
            QMessageBox.critical(
                None, "Error", f"Error setting configuration: {str(e)}"
            )

    def save_config(self, config: dict, filename: str):
        """Save configuration to JSON file"""
        try:
            with open(filename, "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error saving configuration: {str(e)}")

    def add_model_config(self):
        """Add new model configuration"""
        try:
            # Get model name from user
            model_name, ok = QInputDialog.getText(
                self, "Add New Model", "Enter model name:", text="new_model"
            )

            if ok and model_name:
                # Create models directory if it doesn't exist
                parent_dir = os.path.join("models")
                os.makedirs(parent_dir, exist_ok=True)

                # Create filename with .json extension
                filename = f"{model_name}.json"
                path = os.path.join(parent_dir, filename)

                # Check if file already exists
                if os.path.exists(path):
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"Model '{model_name}' already exists. Please choose a different name.",
                    )
                    return

                # Get current configuration
                config = self.get_config()

                # Save configuration to new file
                with open(path, "w") as f:
                    json.dump(config, f, indent=4)

                # Update model combobox
                self._update_model_list()

                # Select the newly added model
                index = self.ui.model.findText(model_name)
                if index >= 0:
                    self.ui.model.setCurrentIndex(index)

                QMessageBox.information(
                    self, "Success", f"New model '{model_name}' created successfully"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create new model: {str(e)}")

    def _update_model_list(self):
        """Update the model combobox with available model configurations"""
        try:
            # Store current selection
            current_model = self.ui.model.currentText()

            # Clear current items
            self.ui.model.clear()

            # Get list of model files
            parent_dir = os.path.join("models")
            if os.path.exists(parent_dir):
                model_files = [f for f in os.listdir(parent_dir) if f.endswith(".json")]
                model_names = [os.path.splitext(f)[0] for f in model_files]
                self.ui.model.addItems(sorted(model_names))

                # Restore previous selection if it still exists
                index = self.ui.model.findText(current_model)
                if index >= 0:
                    self.ui.model.setCurrentIndex(index)
        except Exception as e:
            print(f"Error updating model list: {str(e)}")

    def load_model_config(self):
        """Load configuration from selected model"""
        try:
            model_name = self.ui.model.currentText()
            if not model_name:
                QMessageBox.warning(self, "Warning", "Please select a model to load")
                return

            filename = os.path.join("models", f"{model_name}.json")

            if not os.path.exists(filename):
                QMessageBox.warning(
                    self, "Warning", f"Model file not found: {filename}"
                )
                return

            with open(filename, "r") as f:
                config = json.load(f)

            self.set_config(config)
            QMessageBox.information(
                self, "Success", f"Model '{model_name}' loaded successfully"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")

    def save_model_config(self):
        """Save configuration to current model"""
        try:
            model_name = self.ui.model.currentText()
            if not model_name:
                QMessageBox.warning(self, "Warning", "Please select a model to save to")
                return

            filename = os.path.join("models", f"{model_name}.json")
            config = self.get_config()

            # Confirm overwrite if file exists
            if os.path.exists(filename):
                reply = QMessageBox.question(
                    self,
                    "Confirm Save",
                    f"Model '{model_name}' already exists. Do you want to overwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )

                if reply == QMessageBox.StandardButton.No:
                    return

            with open(filename, "w") as f:
                json.dump(config, f, indent=4)

            QMessageBox.information(
                self, "Success", f"Model '{model_name}' saved successfully"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")

    def start_loop_process(self):
        threading.Thread(target=self.thread_loop_process, daemon=True).start()

    def stop_loop_process(self):
        self.b_stop = True

    def thread_loop_process(self):
        self.b_stop = False
        while True:
            config = self.get_config()
            ret = self.process_image(mat=self.current_image, config=config)
            if ret is not None:
                result, morph = ret
                self.showResultSignal.emit(result, morph)

            if self.b_stop:
                break

            time.sleep(0.05)

    def process_image(self, mat=None, config: dict = None):
        start_time = time.time()
        """Xử lý ảnh dựa trên các tham số hiện tại"""
        if mat is None or config is None:
            return None

        # Convert to grayscale
        gray = cv.cvtColor(mat, cv.COLOR_BGR2GRAY)

        # Apply blur
        blur = self._apply_blur(gray, config)

        # Apply threshold
        thresh = self._apply_threshold(blur, config)

        # Apply morphological operations
        morph = self._apply_morphological(thresh, config)

        # Find and draw contours
        result = self._process_contours(self.current_image, morph, config)

        # Update UI
        # self._update_ui(result, morph)
        print("Techtime Process Image: ", time.time() - start_time)
        return result, morph

    def _apply_blur(self, image, config: dict):
        """Áp dụng blur dựa trên tham số đã chọn"""
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

    def _apply_threshold(self, image, config: dict):
        """Áp dụng threshold dựa trên tham số đã chọn"""
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

    def _apply_morphological(self, image, config: dict):
        """Áp dụng phép toán morphological dựa trên tham số đã chọn"""
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

    def _process_contours(self, original_img, processed_img, config: dict):
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

        retrieval_mode = retrieval_mode_map[config["contour"]["retrieval_mode"]]
        approximation_mode = approximation_mode_map[
            config["contour"]["approximation_mode"]
        ]

        # Tìm contours
        contours, _ = cv.findContours(processed_img, retrieval_mode, approximation_mode)

        # Xử lý contours
        min_area = float(config["detection"]["area_min"])
        max_area = float(config["detection"]["area_max"])
        max_distance = config["detection"]["distance"]

        result_img = original_img.copy()

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            area = w * h
            if min_area <= area <= max_area and abs(w - h) < max_distance:
                cv.drawContours(result_img, [contour], -1, (0, 0, 255), 2)
                cv.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return result_img

    def _update_ui(self, original, processed):
        """
        Updates the UI with original and processed images while maintaining aspect ratio
        and proper scaling.

        Args:
            original (np.ndarray): Original image
            processed (np.ndarray): Processed image (binary/grayscale)
        """
        try:
            # Convert and scale the original image
            if original is not None:
                self.smooth_label(self.ui.OriginalImage, original)

            # Convert and scale the processed image
            if processed is not None:
                self.smooth_label(self.ui.ProcessingImage, processed)

        except Exception as e:
            print(f"Error updating UI: {str(e)}")

    def smooth_label(self, label: QLabel, image: np.ndarray):
        pixmap = ImageConverter.opencv_to_qpixmap(image, label.size())

        # Calculate position to center the image
        x = (label.size().width() - pixmap.width()) // 2
        y = (label.size().height() - pixmap.height()) // 2

        # Clear the label and set new pixmap
        label.clear()
        label.setPixmap(pixmap)
        # Adjust geometry to center the image
        label.setContentsMargins(x, y, x, y)

    def toggle_camera(self):
        """Toggle camera state between running and stopped"""
        if not self.camera_thread or not self.camera_thread.isRunning():
            self.start_camera()
            self.ui.Camera.setText("Stop Camera")
            # Disable Load Image when camera is running
            self.ui.LoadImage.setEnabled(False)
        else:
            self.stop_camera()
            self.ui.Camera.setText("Start Camera")
            # Re-enable Load Image when camera is stopped
            self.ui.LoadImage.setEnabled(True)

    def start_camera(self):
        """Start the camera and return success status"""
        try:
            self.camera_thread = CameraThread()
            self.camera_thread.frameCaptured.connect(self.update_frame)
            self.camera_thread.start()
        except Exception as e:
            QMessageBox.critical(
                self, "Camera Error", f"Failed to start camera: {str(e)}"
            )

    def stop_camera(self):
        """Stop the camera and cleanup"""
        try:
            if self.camera_thread and self.camera_thread.isRunning():
                self.camera_thread.stop()
                self.camera_thread.frameCaptured.disconnect()
        except Exception as e:
            QMessageBox.critical(
                self, "Camera Error", f"Failed to stop camera: {str(e)}"
            )

    def update_frame(self, frame):
        """Update the frame display and store current frame"""
        # self.cap_image = frame.copy()
        self.smooth_label(self.ui.OriginalImage, frame)

    def capture_image(self):
        """Capture current frame or loaded image"""
        try:
            if self.camera_thread.frame is None:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "No image to capture. Please start camera or load an image first.",
                )
                return

            # Create default filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            default_filename = f"captured_image_{timestamp}.jpg"

            # Open save file dialog
            file_dialog = QFileDialog()
            file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            file_dialog.setNameFilter("Images (*.jpg *.png *.bmp)")
            file_dialog.setDefaultSuffix("jpg")
            file_dialog.selectFile(default_filename)

            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                filename = file_dialog.selectedFiles()[0]
                # Save the image
                cv.imwrite(filename, self.camera_thread.frame)
                QMessageBox.information(
                    self, "Success", f"Image saved successfully to:\n{filename}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to capture image: {str(e)}")

    def load_image(self):
        """Load and process an image from file"""
        try:
            # Stop camera if running
            if self.camera_thread and self.camera_thread.isRunning():
                self.stop_camera()
                self.ui.Camera.setText("Open Camera")

            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")

            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                file_path = file_dialog.selectedFiles()[0]
                self.current_image = cv.imread(file_path)

                if self.current_image is None:
                    QMessageBox.critical(
                        self, "Error", "Failed to load image. Please try another file."
                    )
                    return

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred while loading the image: {str(e)}"
            )

    def closeEvent(self, a0):
        self.stop_loop_process()
        return super().closeEvent(a0)
