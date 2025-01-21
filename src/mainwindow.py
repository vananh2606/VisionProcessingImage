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
import socket
import time
import os
from collections import namedtuple

from libs.settings import Settings
from libs.camera_thread import CameraThread
from libs.image_converter import ImageConverter
from libs.image_processor import ImageProcessor, RESULT, BLOBS
from gui.MainWindowUI_ui import Ui_MainWindow
from libs.canvas import Canvas, WindowCanvas
from libs.shape import Shape
from libs.utils import ndarray2pixmap

from libs.tcp_server import Server


STEP_WAIT_TRIGGER = "STEP_WAIT_TRIGGER"
STEP_PREPROCESS = "STEP_PREPROCESS"
STEP_PROCESS = "STEP_PROCESS"
STEP_OUTPUT = "STEP_OUTPUT"
STEP_RELEASE = "STEP_RELEASE"


class MainWindow(QMainWindow):
    showResultTechingSignal = pyqtSignal(object)
    showResultAutoSignal = pyqtSignal(RESULT)
    logInfoSignal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.is_camera_active = False
        self.processing_lock = threading.Lock()

        self.server = Server()
        self.server_thread = None
        self.server_running = False

        self.setup_connections()
        self.initialize_parameters()

        self.canvasOriginalImage = Canvas()
        self.canvasProcessingImage = Canvas()
        self.canvasOutputImage = Canvas()

        self.ui.SRC.addWidget(WindowCanvas(self.canvasOriginalImage))
        self.ui.MBIN.addWidget(WindowCanvas(self.canvasProcessingImage))
        self.ui.DST.addWidget(WindowCanvas(self.canvasOutputImage))

        self.canvasOutputImageAuto = Canvas()
        self.ui.ScreenAuto.addWidget(WindowCanvas(self.canvasOutputImageAuto))

        self.b_stop = False
        self.camera_thread = None
        self.current_image = None
        self.file_paths = []
        self.current_socket = None
        self.b_trigger = False
        self.b_stop_auto = False

        self.image_processor = ImageProcessor()
        self.image_converter = ImageConverter()
        self.settings = Settings()

        self.update_model_list()
        self.start_loop_process()

    def setup_connections(self):
        """Set up signal-slot connections"""
        self.showResultTechingSignal.connect(self.show_result_teaching)
        self.showResultAutoSignal.connect(self.show_result_auto)

        self.ui.button_camera.clicked.connect(self.toggle_camera)
        self.ui.button_capture.clicked.connect(self.capture_image)
        self.ui.button_load_image.clicked.connect(self.load_image)
        self.ui.button_open_folder.clicked.connect(self.open_folder)
        self.ui.button_add_model.clicked.connect(self.add_model_config)
        self.ui.button_save_model.clicked.connect(self.save_model_config)
        self.ui.button_delete_model.clicked.connect(self.delete_model_config)
        self.ui.combo_box_model.currentIndexChanged.connect(self.load_model_config)
        self.ui.list_widget_file.itemSelectionChanged.connect(self.display_image)

        self.logInfoSignal.connect(self.view_log_info)
        self.server.logInfoSignal.connect(self.view_log_info)
        self.server.onTriggerSignal.connect(self.on_trigger)
        self.ui.button_start.clicked.connect(self.on_start_auto)
        self.ui.button_stop.clicked.connect(self.on_stop_auto)

    """
    Logic Layout Auto
    """

    def on_trigger(self, s: socket.socket):
        if self.b_trigger:
            return
        self.b_trigger = True
        self.current_socket = s

    def off_trigger(self):
        self.b_trigger = False
        self.current_socket = None

    def start_loop_auto(self):
        """Khởi động camera và bắt đầu vòng lặp"""
        try:
            self.ui.button_start.setEnabled(False)

            # Khởi động server nếu chưa chạy
            self.start_server()

            # Khởi động camera thread mới
            self.camera_thread = CameraThread()
            self.camera_thread.open_camera()

            threading.Thread(target=self.loop_auto, daemon=True).start()

        except Exception as e:
            self.logInfoSignal.emit(f"Error starting auto mode: {str(e)}")

    def stop_loop_auto(self):
        """Dừng vòng lặp xử lý và camera"""
        try:
            # Đặt cờ dừng vòng lặp
            self.b_stop_auto = True

            # Dừng camera
            if self.camera_thread:
                self.camera_thread.stop_camera()

            # Dừng server
            self.stop_server()
        except Exception as e:
            self.logInfoSignal.emit(f"Error stopping auto mode: {str(e)}")

        self.ui.button_start.setEnabled(True)

    def loop_auto(self):
        self.b_stop_auto = False
        step = STEP_WAIT_TRIGGER
        before_step = ""
        config: dict = None
        mat: np.ndarray = None
        result: RESULT = None

        self.logInfoSignal.emit("Auto processing started")

        while True:
            if before_step != step:
                self.logInfoSignal.emit(step)
                before_step = step

            if step == STEP_WAIT_TRIGGER:
                if self.b_trigger:
                    step = STEP_PREPROCESS

            elif step == STEP_PREPROCESS:
                config = self.get_config()
                _, mat = self.camera_thread.camera.grab()

                step = STEP_PROCESS

            elif step == STEP_PROCESS:
                if mat is None:
                    result = RESULT()
                else:
                    result = self.process_image(mat=mat, config=config)

                step = STEP_OUTPUT
                pass

            elif step == STEP_OUTPUT:
                self.server.send_message(self.current_socket, result.msg)
                self.showResultAutoSignal.emit(result)
                step = STEP_RELEASE

            elif step == STEP_RELEASE:
                self.off_trigger()
                config = None
                mat = None
                result = None
                step = STEP_WAIT_TRIGGER

            time.sleep(0.005)

            if self.b_stop_auto:
                break

        self.logInfoSignal.emit("Auto processing stopped")

    def on_start_auto(self):
        self.start_loop_auto()

    def on_stop_auto(self):
        self.stop_loop_auto()

    def start_server(self):
        threading.Thread(target=self.server.run_server, daemon=True).start()
        self.logInfoSignal.emit("Started server")

    def stop_server(self):
        self.server_running = False
        self.server.stop_server()
        self.logInfoSignal.emit("Stopped server")

    def view_log_info(self, mess):
        self.ui.list_view_log.addItem(mess)

    """
    Logic Layout Teaching
    """

    def initialize_parameters(self):
        """Khởi tạo các tham số mặc định và options"""

        """
        Detection Roi
        """
        # Blur parameters
        self.blur_types = ["Gaussian Blur", "Median Blur", "Average Blur"]
        self.ui.combo_box_type_blur.addItems(self.blur_types)
        self.ui.spin_box_ksize.setRange(1, 31)
        self.ui.spin_box_ksize.setSingleStep(2)
        self.ui.spin_box_ksize.setValue(9)

        # Threshold parameters
        self.adaptive_thresh_types = ["Gaussian", "Mean"]
        self.thresh_types = [
            "Binary",
            "Binary Inverted",
            "Truncate",
            "To Zero",
            "To Zero Inverted",
        ]

        self.ui.combo_box_type_adaptive_thresh.addItems(self.adaptive_thresh_types)
        self.ui.combo_box_type_thresh.addItems(self.thresh_types)

        self.ui.spin_box_block_size.setRange(3, 255)
        self.ui.spin_box_block_size.setSingleStep(2)
        self.ui.spin_box_block_size.setValue(125)

        self.ui.spin_box_c_index.setRange(-255, 255)
        self.ui.spin_box_c_index.setValue(9)

        # Morphological parameters
        self.morph_types = ["Erode", "Dilate", "Open", "Close"]
        self.ui.combo_box_morph.addItems(self.morph_types)
        self.ui.spin_box_kernel.setRange(1, 31)
        self.ui.spin_box_kernel.setSingleStep(2)
        self.ui.spin_box_kernel.setValue(5)

        # Contour parameters
        self.retrieval_modes_options = ["EXTERNAL", "LIST", "CCOMP", "TREE"]
        self.approximation_modes_options = ["NONE", "SIMPLE", "TC89_L1", "TC89_KCOS"]

        self.ui.combo_box_retrieval_modes.addItems(self.retrieval_modes_options)
        self.ui.combo_box_contour_approximation_modes.addItems(
            self.approximation_modes_options
        )

        # Detection parameters
        self.ui.line_edit_area_min.setText("100000")
        self.ui.line_edit_area_max.setText("150000")
        self.ui.distance.setRange(0, 100)
        self.ui.distance.setValue(15)

        """
        Hough Circle
        """
        # Blur parameters hough
        self.blur_types = ["Gaussian Blur", "Median Blur", "Average Blur"]
        self.ui.combo_box_type_blur_hough.addItems(self.blur_types)
        self.ui.spin_box_ksize_hough.setRange(1, 31)
        self.ui.spin_box_ksize_hough.setSingleStep(2)
        self.ui.spin_box_ksize_hough.setValue(9)

        # Hough parameters
        self.hough_type = [
            "HOUGH_STANDARD",
            "HOUGH_PROBABILISTIC",
            "HOUGH_MULTI_SCALE",
            "HOUGH_GRADIENT",
            "HOUGH_GRADIENT_ALT",
        ]
        self.ui.combo_box_type_hough.addItems(self.hough_type)

        self.ui.spin_box_dp.setRange(1, 3)
        self.ui.spin_box_dp.setValue(1)

        self.ui.spin_box_min_dist.setRange(0, 100)
        self.ui.spin_box_min_dist.setValue(8)

        self.ui.spin_box_param1.setRange(0, 255)
        self.ui.spin_box_param1.setValue(50)
        self.ui.spin_box_param2.setRange(0, 200)
        self.ui.spin_box_param2.setValue(20)

        self.ui.spin_box_min_radius.setValue(1)
        self.ui.spin_box_max_radius.setValue(20)

    def get_config(self) -> dict:
        """Get current configuration from UI parameters"""
        shapes: list[Shape] = self.canvasOriginalImage.shapes
        try:
            config = {
                # Blur parameters
                "blur": {
                    "type": self.ui.combo_box_type_blur.currentText(),
                    "ksize": self.ui.spin_box_ksize.value(),
                },
                # Threshold parameters
                "threshold": {
                    "adaptive_type": self.ui.combo_box_type_adaptive_thresh.currentText(),
                    "thresh_type": self.ui.combo_box_type_thresh.currentText(),
                    "block_size": self.ui.spin_box_block_size.value(),
                    "c_index": self.ui.spin_box_c_index.value(),
                },
                # Morphological parameters
                "morphological": {
                    "type": self.ui.combo_box_morph.currentText(),
                    "kernel_size": self.ui.spin_box_kernel.value(),
                },
                # Contour parameters
                "contour": {
                    "retrieval_mode": self.ui.combo_box_retrieval_modes.currentText(),
                    "approximation_mode": self.ui.combo_box_contour_approximation_modes.currentText(),
                },
                # Detection parameters
                "detection": {
                    "area_min": self.ui.line_edit_area_min.text(),
                    "area_max": self.ui.line_edit_area_max.text(),
                    "distance": self.ui.distance.value(),
                },
                "shapes": {
                    i: {"label": shapes[i].label, "box": shapes[i].cvBox}
                    for i in range(len(shapes))
                },
                "hough_circle": {
                    "type_blur_hough": self.ui.combo_box_type_blur_hough.currentText(),
                    "ksize_hough": self.ui.spin_box_ksize_hough.value(),
                    "type_hough": self.ui.combo_box_type_hough.currentText(),
                    "dp": self.ui.spin_box_dp.value(),
                    "min_dist": self.ui.spin_box_min_dist.value(),
                    "param1": self.ui.spin_box_param1.value(),
                    "param2": self.ui.spin_box_param2.value(),
                    "min_radius": self.ui.spin_box_min_radius.value(),
                    "max_radius": self.ui.spin_box_max_radius.value(),
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
                blur_index = self.ui.combo_box_type_blur.findText(
                    config["blur"]["type"]
                )
                if blur_index >= 0:
                    self.ui.combo_box_type_blur.setCurrentIndex(blur_index)
                self.ui.spin_box_ksize.setValue(config["blur"]["ksize"])

            # Threshold parameters
            if "threshold" in config:
                adaptive_index = self.ui.combo_box_type_adaptive_thresh.findText(
                    config["threshold"]["adaptive_type"]
                )
                if adaptive_index >= 0:
                    self.ui.combo_box_type_adaptive_thresh.setCurrentIndex(
                        adaptive_index
                    )

                thresh_index = self.ui.combo_box_type_thresh.findText(
                    config["threshold"]["thresh_type"]
                )
                if thresh_index >= 0:
                    self.ui.combo_box_type_thresh.setCurrentIndex(thresh_index)

                self.ui.spin_box_block_size.setValue(config["threshold"]["block_size"])
                self.ui.spin_box_c_index.setValue(config["threshold"]["c_index"])

            # Morphological parameters
            if "morphological" in config:
                morph_index = self.ui.combo_box_morph.findText(
                    config["morphological"]["type"]
                )
                if morph_index >= 0:
                    self.ui.combo_box_morph.setCurrentIndex(morph_index)
                self.ui.spin_box_kernel.setValue(config["morphological"]["kernel_size"])

            # Contour parameters
            if "contour" in config:
                retrieval_index = self.ui.combo_box_retrieval_modes.findText(
                    config["contour"]["retrieval_mode"]
                )
                if retrieval_index >= 0:
                    self.ui.combo_box_retrieval_modes.setCurrentIndex(retrieval_index)

                approximation_index = (
                    self.ui.combo_box_contour_approximation_modes.findText(
                        config["contour"]["approximation_mode"]
                    )
                )
                if approximation_index >= 0:
                    self.ui.combo_box_contour_approximation_modes.setCurrentIndex(
                        approximation_index
                    )

            # Detection parameters
            if "detection" in config:
                self.ui.line_edit_area_min.setText(str(config["detection"]["area_min"]))
                self.ui.line_edit_area_max.setText(str(config["detection"]["area_max"]))
                self.ui.distance.setValue(config["detection"]["distance"])

            # Shape
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

            # Hough Circle
            if "hough_circle" in config:
                blur_hough_index = self.ui.combo_box_type_blur_hough.findText(
                    config["hough_circle"]["type_blur_hough"]
                )
                if blur_hough_index >= 0:
                    self.ui.combo_box_type_blur_hough.setCurrentIndex(blur_index)
                self.ui.spin_box_ksize_hough.setValue(
                    config["hough_circle"]["ksize_hough"]
                )

                hough_index = self.ui.combo_box_type_hough.findText(
                    config["hough_circle"]["type_hough"]
                )
                if hough_index >= 0:
                    self.ui.combo_box_type_hough.setCurrentIndex(hough_index)

                hough_config = config.get("hough_circle", {})

                self.ui.spin_box_dp.setValue(hough_config.get("dp", 1))
                self.ui.spin_box_min_dist.setValue(hough_config.get("min_dist", 8))
                self.ui.spin_box_param1.setValue(hough_config.get("param1", 50))
                self.ui.spin_box_param2.setValue(hough_config.get("param2", 20))
                self.ui.spin_box_min_radius.setValue(hough_config.get("min_radius", 1))
                self.ui.spin_box_max_radius.setValue(hough_config.get("max_radius", 20))

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
                self.ui.combo_box_model.addItem(model_name)
                self.ui.combo_box_model.setCurrentText(model_name)
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
            model_name = self.ui.combo_box_model.currentText()
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
                self.ui.combo_box_model.removeItem(
                    self.ui.combo_box_model.currentIndex()
                )
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
            model_name = self.ui.combo_box_model.currentText()
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
            model_name = self.ui.combo_box_model.currentText()
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
            self.ui.combo_box_model.clear()
            model_names = self.settings.get_model_names()
            if model_names:
                self.ui.combo_box_model.addItems(sorted(model_names))
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
            ret: RESULT = self.process_image(mat=self.current_image, config=config)
            if ret is not None:
                self.showResultTechingSignal.emit(ret)

            if self.b_stop:
                break

            time.sleep(0.005)

    def process_image(self, mat=None, config: dict = None):
        """Process image with thread safety"""
        time_start = time.time()
        if mat is None or config is None:
            return None

        with self.processing_lock:
            try:
                process_name = self.ui.combo_box_process_name.currentText()

                if process_name == "ProcessAll":
                    # Find and draw contours
                    result: RESULT = self.image_processor.find_result(mat, config)

                    # print("Time Processing: ", time.time() - time_start)
                    return result

                if process_name == "FindBlobs":
                    # Find and draw contours
                    result: BLOBS = self.image_processor.find_blobs(
                        mat, config, b_debug=True
                    )

                    # print("Time Processing: ", time.time() - time_start)
                    return result

                if process_name == "FindCircles":
                    # Find and draw contours
                    index = self.canvasOriginalImage.idSelected
                    if index is not None:
                        current_shape: Shape = self.canvasOriginalImage[index]
                        roi = current_shape.cvBox

                        result: BLOBS = self.image_processor.find_circles(
                            mat, roi, config, b_debug=True
                        )

                    # print("Time Processing: ", time.time() - time_start)
                    return result

            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}][process_image][ERROR]: {str(e)}")
                return None

    def show_result_auto(self, result: RESULT):
        try:
            # Convert and scale the original image
            if result.dst is not None:
                self.canvasOutputImageAuto.load_pixmap(ndarray2pixmap(result.dst))

        except Exception as e:
            print(f"Error updating UI: {str(e)}")

    def show_result_teaching(self, result: RESULT):
        try:
            # Convert and scale the original image
            if result.dst is not None:
                self.canvasOutputImage.load_pixmap(ndarray2pixmap(result.dst))

            # Convert and scale the processed image
            if result.mbin is not None:
                self.canvasProcessingImage.load_pixmap(ndarray2pixmap(result.mbin))

        except Exception as e:
            print(f"Error updating UI: {str(e)}")

    def toggle_camera(self):
        """Toggle camera state between running and stopped"""
        if not self.camera_thread or not self.camera_thread.isRunning():
            self.start_camera()
            self.ui.button_camera.setText("Stop Camera")
            self.ui.button_load_image.setEnabled(False)
            self.ui.button_open_folder.setEnabled(False)
            self.ui.button_capture.setEnabled(True)
            self.is_camera_active = True
        else:
            self.stop_camera()
            self.ui.button_camera.setText("Start Camera")
            self.ui.button_load_image.setEnabled(True)
            self.ui.button_open_folder.setEnabled(True)
            self.ui.button_capture.setEnabled(False)
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
            self.ui.list_widget_file.clear()
            self.camera_thread = CameraThread()
            self.camera_thread.frameCaptured.connect(self.update_frame)
            self.camera_thread.open_camera()
            self.camera_thread.start()

        except Exception as e:
            QMessageBox.critical(
                self, "Camera Error", f"Failed to start camera: {str(e)}"
            )

    def stop_camera(self):
        """Stop the camera and cleanup"""
        try:
            if self.camera_thread and self.camera_thread.isRunning():
                self.camera_thread.stop_camera()
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

            self.ui.list_widget_file.clear()

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
            self.ui.list_widget_file.clear()  # Xóa mục cũ trong QListWidget
            image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
            for file_name in os.listdir(folder_path):
                if any(file_name.lower().endswith(ext) for ext in image_extensions):
                    file_path = os.path.join(folder_path, file_name)
                    self.file_paths.append(file_path)

                    # Thêm mục mới vào QListWidget
                    list_item = QListWidgetItem(file_name)
                    self.ui.list_widget_file.addItem(list_item)

    def display_image(self):
        # Display the current image in the viewer
        selected_items = self.ui.list_widget_file.selectedItems()
        if selected_items:
            item = selected_items[0]
            index = self.ui.list_widget_file.row(item)
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
