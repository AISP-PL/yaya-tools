"""
YOLOv4 detector class using OpenCV DNN module.
"""

import logging
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import supervision as sv

from yaya_tools.detection.detector import Detector  # type: ignore

logger = logging.getLogger(__name__)


def xlylwh_to_xyxy(bbox: Sequence[int]) -> tuple[int, int, int, int]:
    """
    Xtopleft, Ytopleft, W, H -> XMIN, YMIN, XMAX, YMAX
    """
    xtl, ytl, w, h = bbox

    xmin = xtl
    ymin = ytl
    xmax = xtl + w
    ymax = ytl + h

    return (xmin, ymin, xmax, ymax)


class DetectorCVDNN(Detector):
    """YOLOv4 detector class."""

    def __init__(
        self,
        config: dict[str, Any],
        gpuID: int = 0,
        netWidth: int = 0,
        netHeight: int = 0,
    ):
        """
        Constructor
        """
        super().__init__(config)
        # GPU used
        self.gpuid = gpuID
        # Configuration dictionary
        self.cfg_path = config["cfg_path"]
        # Weights file path
        self.weights_path = config["weights_path"]
        # Data file path
        self.data_path = config["data_path"]
        # Names file path
        self.names_path = config["names_path"]
        # Confidence threshold
        self.confidence = config["confidence"]
        # NMS threshold
        self.nms_thresh = config["nms_threshold"]
        # Force CPU
        self.force_cpu = config["force_cpu"]

        # Network configuration
        # ---------------------
        # Network pointer
        self.net: Optional[cv2.dnn.Net] = None
        # Model pointer
        self.model: Optional[cv2.dnn.DetectionModel] = None
        # Network width
        self.netWidth = netWidth
        # Network height
        self.netHeight = netHeight
        # Network layers list
        self.netLayers: Optional[tuple[str, ...]] = None

        # Pre-Read and strip all labels
        self.classes: list[str] = open(self.names_path, "r").read().splitlines()
        self.classes = list(map(str.strip, self.classes))  # strip names

        # Reused darknet image
        self.image = None
        # Reused darknet image properties
        self.imwidth = 0
        self.imheight = 0

    def __del__(self):
        """Destructor."""

    def is_initialized(self) -> bool:
        """Return True if detector is initialized."""
        return self.net is not None

    def __init_from_darknet(self) -> None:
        """Initialize network."""
        # Darknet : Load from darknet .cfg i .weights
        self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)

        # Get network input size
        with open(self.cfg_path, "r") as cfg_file:
            for line in cfg_file.readlines():
                if line.startswith("width"):
                    self.netWidth = int(line.strip().split("=")[1])
                if line.startswith("height"):
                    self.netHeight = int(line.strip().split("=")[1])

    def init(self) -> None:
        """Init call with other arguments."""
        # Check : OpenCV using optimized version
        if not cv2.useOptimized():
            logger.warning("CV2 is not CPU optimized!")

        # Initialize network according to network type
        self.__init_from_darknet()

        # Check if network was loaded
        if self.net is None:
            raise ValueError("Failed to load network!")

        # GPU : Use cuda
        if not self.force_cpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            logger.info("GPU used : ID%d", self.gpuid)
        # Otherwise : Use CPU
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("CPU used.")

        # Logging network informations
        logger.info("Created %ux%u network with %u classes.", self.netWidth, self.netHeight, len(self.classes))

        # Get all network layers names
        self.netLayers = tuple(self.net.getLayerNames())
        # Get all network unconnected layers
        self.netOutLayers = self.net.getUnconnectedOutLayersNames()
        # Get detection model reference for network
        self.model = cv2.dnn_DetectionModel(self.net)  # type: ignore
        if self.model is None:
            raise ValueError("Failed to create detection model!")

        # Set input parameters for model
        self.model.setInputParams(size=(self.netWidth, self.netHeight), scale=1 / 255, swapRB=True)

        # Logging network output layers names.
        logger.info("Network output layers: %s", ",".join(self.netOutLayers))

        logger.info(
            "Created %ux%u network with %u classes.",
            self.netWidth,
            self.netHeight,
            len(self.classes),
        )

    def open_stream(self, width: int, height: int, fps: float) -> None:
        """Open stream with stream details."""

    def detect(
        self,
        frame_number: int,
        frame: np.ndarray,
    ) -> sv.Detections:
        """Detect objects in given frame"""
        # Image : Check
        if frame is None:
            logger.error("(Detector) Image is None!")
            return sv.Detections.empty()

        # Network : Check
        if self.net is None or self.model is None:
            logger.error("(Detector) Network is not initialized!")
            return sv.Detections.empty()

        # Image : Dimensions
        imwidth = frame.shape[1]
        imheight = frame.shape[0]

        # Check : Image is valid
        if (imwidth == 0) or (imheight == 0):
            return sv.Detections.empty()

        yolo_classids, yolo_confidences, yolo_bboxes = self.model.detect(
            frame=frame, confThreshold=self.confidence, nmsThreshold=self.nms_thresh
        )

        # Check : Empty
        if len(yolo_bboxes) == 0:
            return sv.Detections.empty()

        # As supervision : convert to supervision format
        # [xyxy array, classes array, and confidences array]
        yolo_xyxy = np.array([xlylwh_to_xyxy(bbox) for bbox in yolo_bboxes])

        detections_sv = sv.Detections(
            xyxy=yolo_xyxy,
            confidence=np.array(yolo_confidences),
            class_id=np.array(yolo_classids),
        )

        return detections_sv
