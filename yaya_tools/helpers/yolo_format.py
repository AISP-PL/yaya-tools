from typing import Tuple

import numpy as np
from supervision import Detections


def object_to_yolo(
    xyxy: np.ndarray,
    class_id: int,
    image_shape: Tuple[int, int, int],
    precision: int = 7,
) -> str:
    """Convert bounding box coordinates to YOLO format."""
    h, w, _ = image_shape
    xyxy_relative = xyxy / np.array([w, h, w, h], dtype=np.float32)
    x_min, y_min, x_max, y_max = xyxy_relative
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return f"{int(class_id)} {x_center:.{precision}f} {y_center:.{precision}f} {width:.{precision}f} {height:.{precision}f}"


def detections_to_yolo_annotations(
    detections: Detections,
    image_shape: tuple[int, int, int],
    precision: int = 7,
) -> list[str]:
    """
    Convert detections to YOLO format annotations.

    Parameters:
    -----------
        detections (Detections): Detections object containing bounding boxes and class IDs.
        image_shape (tuple): Shape of the image in the format (height, width, channels).
        precision (int): Decimal precision for the coordinates in the YOLO format.

    Returns:
    --------
        list[str]: List of YOLO format annotations.

    """
    annotation = []
    for xyxy, mask, _, class_id, _, _ in detections:
        if class_id is None:
            raise ValueError("Class ID is required for YOLO annotations.")

        next_object = object_to_yolo(xyxy=xyxy, class_id=class_id, image_shape=image_shape)
        annotation.append(next_object)

    return annotation
