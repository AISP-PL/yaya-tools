"""
Detector abstract class for all detectors
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import supervision as sv  # type: ignore


class Detector(ABC):
    """
    Detector abstract class for all AI detectors and motion detectors.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str = "detector",
    ):
        """Constructor"""
        self.config = config
        self.name = name

    @abstractmethod
    def is_initialized(self) -> bool:
        """Return True if detector is initialized."""
        pass

    @abstractmethod
    def init(self):
        """Init call with other arguments."""
        pass

    @abstractmethod
    def open_stream(self, width: int, height: int, fps: float) -> None:
        """Open stream with stream details."""
        pass

    @abstractmethod
    def detect(
        self,
        frame_number: int,
        frame: np.ndarray,
    ) -> sv.Detections:
        """Detect objects in given frame"""
        pass
