import logging
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from PyQt5.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QMouseEvent, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yaya_tools.detection.detector_yolov4_darknet import DetectorDarknet


def logging_terminal_setup() -> None:
    """
    Setup logging for the application.

    Parameters
    ----------
    path_field : str
        Field in the config file that contains the path to the log file.
        Default is "path".
    is_terminal : bool
        If True, logs will be printed to the terminal.
        Default is True.
    """
    logging.getLogger().setLevel(logging.DEBUG)  # Ensure log level is set to DEBUG
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logging.info("\n\n###### Logging start of terminal session ######\n")


# Extended label to handle mouse clicks
class VideoLabel(QLabel):
    clicked = pyqtSignal(QPoint)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Mouse left button click event."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.pos())
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video Processing Application")
        self.videoCapture: Optional[cv2.VideoCapture] = None
        self.timer: QTimer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.currentFrame: Optional[np.ndarray] = None
        self.homography: Optional[np.ndarray] = None  # Transformation matrix
        self.dst_size: Optional[Tuple[int, int]] = None  # Target size (width, height)
        self.drawing_mode: bool = False  # Drawing mode flag
        self.points: List[List[int]] = []  # List of clicked points (source)
        self.playing: bool = False
        self.speedMultiplier: int = 1
        self.fps: float = 25.0
        self.objects_buffer: list = []  # Buffer for the last 3 seconds of detections
        self.yolo_detector: Optional[DetectorDarknet] = None  # YOLO model attribute
        self.box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.CLASS)
        self.label_annotator = sv.LabelAnnotator(text_padding=5, color_lookup=sv.ColorLookup.CLASS)

        # Main Video label
        self.main_video_label: VideoLabel = VideoLabel()
        self.main_video_label.setAlignment(Qt.AlignCenter)
        self.main_video_label.clicked.connect(self.getPoint)

        # Homography Video label
        self.homography_video_label: VideoLabel = VideoLabel()
        self.homography_video_label.setAlignment(Qt.AlignCenter)
        self.homography_video_label.clicked.connect(self.getPoint)

        # Buttons + horizontal container
        self.openButton: QPushButton = QPushButton("Open")
        self.openButton.clicked.connect(self.openFile)

        self.openYOLOButton: QPushButton = QPushButton("Open YOLO")
        self.openYOLOButton.clicked.connect(self.openYOLO)

        self.startStopButton: QPushButton = QPushButton("Start")
        self.startStopButton.clicked.connect(self.togglePlayback)

        self.speedComboBox: QComboBox = QComboBox()
        self.speedComboBox.addItems(["1x", "2x", "4x"])
        self.speedComboBox.currentIndexChanged.connect(self.changeSpeed)

        self.transformButton: QPushButton = QPushButton("Add Transformation")
        self.transformButton.clicked.connect(self.activateDrawingMode)

        self.resetButton: QPushButton = QPushButton("Reset Transformation")
        self.resetButton.clicked.connect(self.resetTransformation)

        buttonsLayout: QHBoxLayout = QHBoxLayout()
        buttonsLayout.addWidget(self.openButton)
        buttonsLayout.addWidget(self.openYOLOButton)  # Added YOLO button
        buttonsLayout.addWidget(self.startStopButton)
        buttonsLayout.addWidget(self.speedComboBox)
        buttonsLayout.addWidget(self.transformButton)
        buttonsLayout.addWidget(self.resetButton)
        buttonsLayout.addStretch()  # Spacer at the end

        # Main layout – first the button container, then the video label
        mainLayout: QVBoxLayout = QVBoxLayout()
        mainLayout.addLayout(buttonsLayout)
        # Horizontal Layout inside mainLayout for two video labels
        horizontalLayout: QHBoxLayout = QHBoxLayout()
        horizontalLayout.addWidget(self.main_video_label)
        horizontalLayout.addWidget(self.homography_video_label)
        mainLayout.addLayout(horizontalLayout)

        container: QWidget = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to a maximum width of 1280 pixels while maintaining aspect ratio."""
        max_width = 800
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        return frame

    def openFile(self) -> None:
        """Open a video file."""
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Select video file", "", "Video files (*.avi *.mp4 *.mov *.mkv)"
        )
        if fileName:
            self.homography = None
            self.dst_size = None
            self.videoCapture = cv2.VideoCapture(fileName)
            ret, frame = self.videoCapture.read()
            if ret:
                frame = self.resize_frame(frame)
                self.currentFrame = frame
                height, width, _ = frame.shape
                # Set label size according to the resized frame
                self.main_video_label.setFixedSize(width, height)
                self.displayFrame(frame)
                self.fps = self.videoCapture.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0:
                    self.fps = 25
                # Timer auto start
                self.timer.setInterval(int(1000 / self.fps))
                self.timer.start()
                self.playing = True
                self.startStopButton.setText("Stop")

    def nextFrameSlot(self) -> None:
        """Process the next frame in the video."""
        if self.videoCapture is not None:
            ret, frame = self.videoCapture.read()
            if ret:
                frame = self.resize_frame(frame)
                self.currentFrame = frame
                self.displayFrame(frame)
                # Skip additional frames according to the selected multiplier
                for _ in range(self.speedMultiplier - 1):
                    ret_skip, _ = self.videoCapture.read()
                    if not ret_skip:
                        self.videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        break
            else:
                # After finishing video playback, return to the beginning
                self.videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def displayFrame(self, frame: np.ndarray) -> None:
        """Display the current frame."""
        # If homography matrix exists, transform the frame
        homography_frame = frame
        if self.homography is not None and self.dst_size is not None:
            homography_frame = cv2.warpPerspective(frame, self.homography, self.dst_size)

        # If in drawing mode, overlay points and lines to show the forming polygon
        if self.drawing_mode and self.points:
            # Draw points
            for pt in self.points:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

            # Draw lines between consecutive points
            if len(self.points) > 1:
                for i in range(len(self.points) - 1):
                    pt1 = (int(self.points[i][0]), int(self.points[i][1]))
                    pt2 = (int(self.points[i + 1][0]), int(self.points[i + 1][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                # If the user clicked 4 points, close the polygon
                if len(self.points) == 4:
                    pt1 = (int(self.points[3][0]), int(self.points[3][1]))
                    pt2 = (int(self.points[0][0]), int(self.points[0][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Additional step: if YOLO model is loaded, perform detection and annotation
        detections: sv.Detections = sv.Detections.empty()
        if self.yolo_detector is not None:
            detections = self.yolo_detector.detect(frame_number=0, frame=frame)

        # Detections: Get anchor point of center bottom
        objects_xy: np.ndarray = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(np.int32)

        # Update buffer: buffer length equals fps*3 (3 seconds)
        max_buffer_length = int(self.fps) * 3
        self.objects_buffer.append(objects_xy)
        if len(self.objects_buffer) > max_buffer_length:
            self.objects_buffer.pop(0)

        # Use default color_lookup
        annotated = self.box_annotator.annotate(frame.copy(), detections)
        annotated = self.label_annotator.annotate(annotated, detections)
        # Annotate: Draw points
        for obj_xy in objects_xy:
            cv2.circle(annotated, (int(obj_xy[0]), int(obj_xy[1])), 5, (0, 255, 0), -1)

        frame = annotated

        # Base frame: Convert and display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        bytesPerLine = 3 * width
        qImg = QImage(rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.main_video_label.setPixmap(QPixmap.fromImage(qImg))

        # Draw points from buffer on the homography image:
        if self.homography is not None and self.dst_size is not None:
            for buffered_points in self.objects_buffer:
                for point in buffered_points:
                    transformed = cv2.perspectiveTransform(np.array([[point]], dtype="float32"), self.homography)
                    pt = transformed[0][0]
                    cv2.circle(homography_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

        # Homography frame: Convert and display
        rgb_homography = cv2.cvtColor(homography_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_homography.shape
        bytesPerLine = 3 * width
        qImg_homography = QImage(rgb_homography.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.homography_video_label.setPixmap(QPixmap.fromImage(qImg_homography))

    def activateDrawingMode(self) -> None:
        """Activate drawing mode – reset the list of points."""
        self.drawing_mode = True
        self.points = []
        logging.info("Click 4 points in order: Bottom left, Bottom right, Top right, Top left.")

    def getPoint(self, pos: QPoint) -> None:
        """Register a point when clicked."""
        if self.drawing_mode:
            self.points.append([pos.x(), pos.y()])
            logging.info(f"Registered point: {pos.x()}, {pos.y()}.")
            if len(self.points) == 4:
                self.drawing_mode = False
                self.computeHomography()
                # After computing homography, clear points to avoid drawing on the processed image
                self.points = []
                logging.info("Transformation matrix computed.")

    def computeHomography(self) -> None:
        """Compute the homography matrix."""
        pts = np.array(self.points, dtype="float32")
        # Assume points are in order:
        # [0] Bottom left, [1] Bottom right, [2] Top right, [3] Top left

        width_bottom = float(np.linalg.norm(pts[1] - pts[0]))
        width_top = float(np.linalg.norm(pts[2] - pts[3]))
        maxWidth = int(max(width_bottom, width_top))

        height_left = float(np.linalg.norm(pts[0] - pts[3]))
        height_right = float(np.linalg.norm(pts[1] - pts[2]))
        maxHeight = int(max(height_left, height_right))

        self.dst_size = (maxWidth, maxHeight)
        dst = np.array(
            [
                [0, maxHeight],  # Bottom left
                [maxWidth, maxHeight],  # Bottom right
                [maxWidth, 0],  # Top right
                [0, 0],  # Top left
            ],
            dtype="float32",
        )
        self.homography = cv2.getPerspectiveTransform(pts, dst)

    def resetTransformation(self) -> None:
        """Reset the transformation matrix."""
        self.homography = None
        self.dst_size = None
        logging.info("Transformation reset.")

    def togglePlayback(self) -> None:
        """Toggle video playback."""
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.startStopButton.setText("Start")
        else:
            self.timer.start(int(1000 / self.fps))
            self.playing = True
            self.startStopButton.setText("Stop")

    def changeSpeed(self) -> None:
        """Update the speed multiplier based on the selected value in the combobox."""
        speed_text = self.speedComboBox.currentText()
        self.speedMultiplier = int(speed_text.replace("x", ""))
        logging.info(f"Selected speed: {self.speedMultiplier}x")

    def openYOLO(self) -> None:
        """Load the YOLO model from cfg and weights files."""
        cfgPath, _ = QFileDialog.getOpenFileName(self, "Select .cfg file", "", "CFG files (*.cfg)")
        if not cfgPath:
            return

        weightsPath, _ = QFileDialog.getOpenFileName(self, "Select .weights file", "", "Weights files (*.weights)")
        if not weightsPath:
            return

        namesPath, _ = QFileDialog.getOpenFileName(self, "Select .names file", "", "Names files (*.names)")
        if not namesPath:
            return

        # Initialize YOLO model - names file will be generated (e.g., as a placeholder)
        self.yolo_detector = DetectorDarknet(
            config={
                "cfg_path": cfgPath,
                "weights_path": weightsPath,
                "data_path": "",
                "names_path": namesPath,
                "confidence": 0.50,
                "nms_threshold": 0.30,
                "force_cpu": True,
            }
        )
        self.yolo_detector.init()
        logging.info("YOLO model loaded.")


def main() -> None:
    """Run the PyQt5 application."""
    logging_terminal_setup()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
