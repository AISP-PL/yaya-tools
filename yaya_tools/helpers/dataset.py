"""
This module contains helper functions for dataset management
"""

import os
from typing import Optional

import supervision as sv  # type: ignore
from supervision.dataset.formats.yolo import yolo_annotations_to_detections
from supervision.utils.file import read_txt_file


def dataset_load_as_sv(images_annotations: dict[str, Optional[str]], dataset_path: str) -> sv.Detections:
    """
    Load the annotations from the dataset folder

    Arguments
    ----------
    images_annotations : dict[str, Optional[str]]
        Dictionary of images and their annotations
    dataset_path : str
        Path to the dataset folder

    Returns
    -------
    dict[str, Optional[str]]
        Dictionary of images and their annotations
    """
    detections: sv.Detections = sv.Detections.empty()

    for image_path, annotations_path in images_annotations.items():
        # Skip images without annotations
        if annotations_path is None:
            continue

        # Annotations : Load the annotations
        lines = read_txt_file(file_path=os.path.join(dataset_path, annotations_path), skip_empty=True)
        # sv.Detections : Create
        file_detections = yolo_annotations_to_detections(
            lines=lines,
            resolution_wh=(1, 1),
            with_masks=False,
            is_obb=False,
        )

        # Merge
        detections = sv.Detections.merge([detections, file_detections])

    return detections


def dataset_to_validation() -> None:
    """Create a validation.txt file from the dataset folder"""
