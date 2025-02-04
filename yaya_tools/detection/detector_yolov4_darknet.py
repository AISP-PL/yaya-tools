""" """

import logging
import os
from typing import Any

import cv2
import numpy as np
import supervision as sv  # type: ignore

from yaya_tools.detection import darknet
from yaya_tools.detection.detector import Detector


class DetectorDarknet(Detector):
    """YOLOv4 detector class."""

    def __init__(
        self,
        config: dict[str, Any],
        gpuID: int = 0,
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

        # Network configuration
        # ---------------------
        # Network pointer
        self.net = None
        # Network width
        self.netWidth = 0
        # Network height
        self.netHeight = 0

        # Confidence threshold
        self.confidence = config["confidence"]
        # NMS threshold
        self.nms_thresh = config["nms_threshold"]

        # Reused darknet image ptr
        self.image = None
        # Reused darknet image properties
        self.imwidth = 0
        self.imheight = 0
        # List of colors matched with labels
        self.colors: list[tuple] = []
        # Pre-Read and strip all labels
        self.classes = open(self.names_path).read().splitlines()
        self.classes = list(map(str.strip, self.classes))  # strip names

        # Info : Logging
        logging.info(
            "YOLOv4 Detector created with cfg=%s, weights=%s, data=%s, names=%s",
            self.cfg_path,
            self.weights_path,
            self.data_path,
            self.names_path,
        )
        logging.info("YOLOv4 Detector created with confidence=%f, nms=%f", self.confidence, self.nms_thresh)

    def __del__(self):
        """Destructor."""
        # GPU : Select mine before freeing
        darknet.set_gpu(self.gpuid)

        # Free network image
        if self.image is not None:
            darknet.free_image(self.image)

        # Unload network from memory
        if self.net is not None:
            darknet.free_network_ptr(self.net)

    def is_initialized(self) -> bool:
        """Return True if detector is initialized."""
        return self.net is not None

    def init(self):
        """Init call with other arguments."""
        # Check : Network weights should exists
        if self.weights_path is None:
            raise Exception("Network weights not specified!")

        # Check : Network weights should be larger > 1MiB
        if os.path.getsize(self.weights_path) < 1024 * 1024:
            raise Exception("Network weights are too small! Have you installed git-LFS?")

        # Choose GPU to use for YOLO
        darknet.set_gpu(self.gpuid)

        # YOLO net, labels, cfg
        try:
            self.net, self.classes, self.colors = darknet.load_network(
                self.cfg_path, self.names_path, self.weights_path
            )

        except Exception as e:
            raise Exception("Failed to load network!", e)

        # Get network  input (width and height)
        self.netWidth = darknet.network_width(self.net)
        self.netHeight = darknet.network_height(self.net)

        # Create frame object we will use each time, w
        # with dimensions of network width,height.
        if self.image is None:
            self.image = darknet.make_image(self.netWidth, self.netHeight, 3)

        logging.info(
            "Created %ux%u network with %u classes.",
            self.netWidth,
            self.netHeight,
            len(self.classes),
        )
        logging.info("GPU used : ID%d", self.gpuid)

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
            logging.error("(Detector) Image is None!")
            return sv.Detections.empty()

        # Always swap BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Image : Dimensions
        imwidth = frame_rgb.shape[1]
        imheight = frame_rgb.shape[0]

        # Check : Image is valid
        if (imwidth == 0) or (imheight == 0):
            return sv.Detections.empty()

        # Image : Resize to network dimensions.
        resized = cv2.resize(frame_rgb, (self.netWidth, self.netHeight), interpolation=cv2.INTER_NEAREST)

        # Copy image to darknet image
        darknet.copy_image_from_bytes(self.image, resized.tobytes())

        # YOLO detections format as np.ndarray
        # [[x1, y1, x2, y2, confidence, class_no], [...] ...]
        detections = darknet.detect_image(
            self.net,
            self.classes,
            self.image,
            imwidth,
            imheight,
            thresh=self.confidence,
            nms=self.nms_thresh,
        )

        # Check : Empty
        if detections is None or len(detections) == 0:
            return sv.Detections.empty()

        # As supervision : convert to supervision format
        # [xyxy array, classes array, and confidences array]
        detections_sv = sv.Detections(
            xyxy=detections[:, 0:4].astype(int),
            confidence=detections[:, 4],
            class_id=detections[:, 5].astype(int),
        )

        return detections_sv
