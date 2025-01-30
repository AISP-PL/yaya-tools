"""
This module contains helper functions for dataset management
"""

import supervision as sv  # type: ignore


def dataset_to_validation(annotations: sv.Detections, ratio: float = 0.20) -> list[str]:
    """
    Create a validation files list from basing on all annotations
    """
