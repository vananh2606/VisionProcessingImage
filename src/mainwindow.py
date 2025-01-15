import shutil
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QLabel,
    QListWidgetItem,
)
from PyQt6.QtCore import QStringListModel, QPointF, QDir, pyqtSignal

import cv2 as cv
import numpy as np
import json
import threading
import time
import os

from libs.settings import Settings
from libs.camera_thread import CameraThread
from libs.image_converter import ImageConverter
from libs.image_processor import ImageProcessor
from gui.MainWindowUI_ui import Ui_MainWindow
from libs.canvas import Canvas, WindowCanvas
from libs.shape import Shape
from libs.utils import ndarray2pixmap


class MainWindow(QMainWindow):
    showResultSignal = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.showResultSignal.connect(self.update_ui)
        self.is_camera_active = False
        self.processing_lock = threading.Lock()

        self.setup_connections()
        self.initialize_parameters()

        self.canvasOriginalImage = Canvas()
        self.canvasProcessingImage = Canvas()
        self.canvasOutputImage = Canvas()

        self.ui.Screen.addWidget(WindowCanvas(self.canvasOriginalImage))
        self.ui.Screen.addWidget(WindowCanvas(self.canvasProcessingImage))
        self.ui.Screen.addWidget(WindowCanvas(self.canvasOutputImage))

        self.b_stop = False
        self.camera_thread = None
        self.current_image = None
        self.file_paths = []

        self.image_processor = ImageProcessor()
        self.image_converter = ImageConverter()
        self.settings = Settings()

        self.update_model_list()
        self.start_loop_process()

    def setup_connections(self):
        """Set up signal-slot connections"""
        self.ui.Camera.clicked.connect(self.toggle_camera)
        self.ui.Capture.clicked.connect(self.capture_image)
        self.ui.LoadImage.clicked.connect(self.load_image)
        self.ui.OpenFolder.clicked.connect(self.open_folder)
        self.ui.AddModel.clicked.connect(self.add_model_config)
        self.ui.SaveModel.clicked.connect(self.save_model_config)
        self.ui.DeleteModel.clicked.connect(self.delete_model_config)
        self.ui.model.currentIndexChanged.connect(self.load_model_config)
        self.ui.listWidgetFile.itemSelectionChanged.connect(self.display_image)

    def initialize_parameters(self):
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
        self.ui.area_min.setText("100000")
        self.ui.area_max.setText("150000")
        self.ui.distance.setRange(0, 100)
        self.ui.distance.setValue(15)

    def get_config(self) -> dict:
        """Get current configuration from UI parameters"""
        shapes: list[Shape] = self.canvasOriginalImage.shapes
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
                "shapes": {
                    i: {"label": shapes[i].label, "box": shapes[i].cvBox}
                    for i in range(len(shapes))
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

            self.canvasOriginalImage.shapes.clear()
            if "shapes" in config:
                shapes: dict = config["shapes"]
                for i in shapes:
                    label = shapes[i]["label"]
                    x, y, w, h = shapes[i]["box"]
                    s = Shape(label)
                    s.points = [
                        QPointF(x, y),
                        QPointF(x + w, y),
                        QPointF(x + w, y + h),
                        QPointF(x, y + h),
                    ]
                    self.canvasOriginalImage.shapes.append(s)

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

            if not ok or not model_name:
                return

            model_name = model_name.upper()

            # Check if model already exists
            if model_name in self.settings.get_model_names():
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"Model '{model_name}' already exists. Please choose a different name.",
                )
                return

            # Get current configuration and save
            config = self.get_config()
            if self.settings.save_model(model_name, config):
                self.ui.model.addItem(model_name)
                self.ui.model.setCurrentText(model_name)
                self.statusBar().showMessage(
                    f"New model '{model_name}' created successfully", 5000
                )
            else:
                QMessageBox.critical(self, "Error", "Failed to create new model")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create new model: {str(e)}")

    def delete_model_config(self):
        """Delete configuration of current model"""
        try:
            model_name = self.ui.model.currentText()
            if not model_name:
                QMessageBox.warning(self, "Warning", "Please select a model to delete")
                return

            # Confirm deletion
            reply = QMessageBox.question(
                self,
                "Confirm Delete",
                f"Are you sure you want to delete the model '{model_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.No:
                return

            # Delete model
            if self.settings.delete_model(model_name):
                self.ui.model.removeItem(self.ui.model.currentIndex())
                QMessageBox.information(
                    self, "Success", f"Model '{model_name}' deleted successfully"
                )
                self.load_model_config()
            else:
                QMessageBox.critical(
                    self, "Error", f"Failed to delete model '{model_name}'"
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete model: {str(e)}")

    def load_model_config(self):
        """Load configuration from selected model"""
        try:
            model_name = self.ui.model.currentText()
            if not model_name:
                return

            config = self.settings.load_model(model_name)
            if config and Settings.validate_config(config):
                self.set_config(config)
                self.statusBar().showMessage(
                    f"Model '{model_name}' loaded successfully", 5000
                )
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"Invalid or missing configuration for model: {model_name}",
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

            # Get current configuration
            config = self.get_config()

            # Confirm overwrite if model exists
            if model_name in self.settings.get_model_names():
                reply = QMessageBox.question(
                    self,
                    "Confirm Save",
                    f"Model '{model_name}' already exists. Do you want to overwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return

            # Save configuration
            if self.settings.save_model(model_name, config):
                QMessageBox.information(
                    self, "Success", f"Model '{model_name}' saved successfully"
                )
            else:
                QMessageBox.critical(
                    self, "Error", "Failed to save model configuration"
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")

    def update_model_list(self):
        """Update the model combobox with available model configurations"""
        try:
            self.ui.model.clear()
            model_names = self.settings.get_model_names()
            if model_names:
                self.ui.model.addItems(sorted(model_names))
                self.load_model_config()
        except Exception as e:
            print(f"Error updating model list: {str(e)}")

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
        """Process image with thread safety"""
        time_start = time.time()
        if mat is None or config is None:
            return None

        with self.processing_lock:
            try:
                # Convert to grayscale
                gray = cv.cvtColor(mat, cv.COLOR_BGR2GRAY)

                # Apply blur
                blur = self.image_processor.apply_blur(gray, config)

                # Apply threshold
                thresh = self.image_processor.apply_threshold(blur, config)

                # Apply morphological operations
                morph = self.image_processor.apply_morphological(thresh, config)

                # Find and draw contours
                result = self.image_processor.process_contours(mat, morph, config)

                print("Time Processing: ", time.time() - time_start)
                return result, morph
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                return None

    def update_ui(self, output, processed):
        """
        Updates the UI with original and processed images while maintaining aspect ratio
        and proper scaling.

        Args:
            original (np.ndarray): Original image
            processed (np.ndarray): Processed image (binary/grayscale)
        """
        try:
            # Convert and scale the original image
            if output is not None:
                self.canvasOutputImage.load_pixmap(ndarray2pixmap(output))

            # Convert and scale the processed image
            if processed is not None:
                self.canvasProcessingImage.load_pixmap(ndarray2pixmap(processed))

        except Exception as e:
            print(f"Error updating UI: {str(e)}")

    def toggle_camera(self):
        """Toggle camera state between running and stopped"""
        if not self.camera_thread or not self.camera_thread.isRunning():
            self.start_camera()
            self.ui.Camera.setText("Stop Camera")
            self.ui.LoadImage.setEnabled(False)
            self.ui.OpenFolder.setEnabled(False)
            self.ui.Capture.setEnabled(True)
            self.is_camera_active = True
        else:
            self.stop_camera()
            self.ui.Camera.setText("Start Camera")
            self.ui.LoadImage.setEnabled(True)
            self.ui.OpenFolder.setEnabled(True)
            self.ui.Capture.setEnabled(False)
            self.is_camera_active = False
            # Clear the current image when stopping camera
            with self.processing_lock:
                self.current_image = None
                # Clear the labels
                self.canvasOutputImage.clear_pixmap()
                self.canvasProcessingImage.clear_pixmap()

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
        with self.processing_lock:
            if self.is_camera_active:
                self.current_image = frame.copy()
                self.canvasOriginalImage.load_pixmap(ndarray2pixmap(frame))

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
            # Ensure camera is stopped
            if self.is_camera_active:
                self.toggle_camera()

            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")

            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                file_path = file_dialog.selectedFiles()[0]

                with self.processing_lock:
                    self.current_image = cv.imread(file_path)
                    if self.current_image is None:
                        QMessageBox.critical(
                            self,
                            "Error",
                            "Failed to load image. Please try another file.",
                        )
                        return
                    # Update the original image display
                    self.canvasOriginalImage.load_pixmap(
                        ndarray2pixmap(self.current_image), True
                    )

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred while loading the image: {str(e)}"
            )

    def open_folder(self):
        # Hộp thoại chọn thư mục
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            # Lấy danh sách file ảnh từ thư mục
            self.file_paths.clear()  # Xóa dữ liệu cũ
            self.ui.listWidgetFile.clear()  # Xóa mục cũ trong QListWidget
            image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
            for file_name in os.listdir(folder_path):
                if any(file_name.lower().endswith(ext) for ext in image_extensions):
                    file_path = os.path.join(folder_path, file_name)
                    self.file_paths.append(file_path)

                    # Thêm mục mới vào QListWidget
                    list_item = QListWidgetItem(file_name)
                    self.ui.listWidgetFile.addItem(list_item)

    def display_image(self):
        # Display the current image in the viewer
        selected_items = self.ui.listWidgetFile.selectedItems()
        if selected_items:
            item = selected_items[0]
            index = self.ui.listWidgetFile.row(item)
            file_path = self.file_paths[index]
            with self.processing_lock:
                self.current_image = cv.imread(file_path)
                if self.current_image is None:
                    QMessageBox.critical(
                        self,
                        "Error",
                        "Failed to load image. Please try another file.",
                    )
                    return
                # Update the original image display
                self.canvasOriginalImage.load_pixmap(
                    ndarray2pixmap(self.current_image), True
                )

    def closeEvent(self, event):
        """Clean up threads before closing"""
        self.stop_loop_process()
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_camera()
        return super().closeEvent(event)
