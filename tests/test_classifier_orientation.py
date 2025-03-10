from typing import Optional

import numpy as np
import pytest
import supervision as sv  # type: ignore

from yaya_tools.classifiers.classifier_orientation import DetectionsOrientation, get_detections_orientation
from yaya_tools.helpers.annotations import annotations_load_as_sv
from yaya_tools.helpers.dataset import load_directory_images_annotatations

# Example points (range 0..1000)
data_examples = {
    "horizontal": [(100, 200), (200, 210), (300, 190), (400, 205)],
    "vertical": [(200, 100), (210, 200), (190, 300), (205, 400)],
    "diagonal_right": [(100, 100), (200, 200), (300, 300), (400, 400)],
    "diagonal_left": [(100, 400), (200, 300), (300, 200), (400, 100)],
    "scattered": [(100, 300), (200, 100), (300, 400), (400, 200)],
    "clustered": [(100, 100), (150, 140), (130, 120), (160, 150)],
}


# Helper method creatin xyxy boxes from points and random 100-200 width and height.
def sv_detections_from_points(points: list[tuple[int, int]]) -> sv.Detections:
    """Create detections from points."""
    boxes = []
    for point in points:
        x, y = point
        width = np.random.randint(100, 200)
        height = np.random.randint(100, 200)
        boxes.append([x - width, y - height, x + width, y + height])

    return sv.Detections(xyxy=np.array(boxes), class_id=np.zeros(len(boxes)), confidence=np.ones(len(boxes)))


@pytest.mark.parametrize(
    "data, expected",
    [
        (data_examples["horizontal"], DetectionsOrientation.HORIZONTAL),
        (data_examples["vertical"], DetectionsOrientation.VERTICAL),
        (data_examples["diagonal_right"], DetectionsOrientation.DIAGONAL_RIGHT),
        (data_examples["diagonal_left"], DetectionsOrientation.DIAGONAL_LEFT),
        (data_examples["scattered"], DetectionsOrientation.SCATTERED),
        (data_examples["clustered"], DetectionsOrientation.CLUSTERED),
    ],
)
def test_get_detections_orientation(data, expected) -> None:
    """Test the get_detections_orientation function with various point orientations."""
    detections = sv_detections_from_points(data)
    assert get_detections_orientation(detections) == expected


def test_dir_horizontal() -> None:
    """Test the horizontal direction."""
    path = "tests/test_orientation/horizontal"

    # All images : with optional annotation filename
    all_images_annotations: dict[str, Optional[str]] = load_directory_images_annotatations(path)

    # All annotations as SV : Get
    annotations, all_negatives = annotations_load_as_sv(all_images_annotations, path)

    # Orientations : Get
    annotations_files = annotations.data.get("filepaths", np.array([]))
    unique_files = np.unique(annotations_files)
    files_orientation: np.ndarray = np.array(
        [get_detections_orientation(annotations[annotations_files == filename]) for filename in unique_files], dtype=int  # type: ignore
    )

    # Check if all orientations are horizontal
    assert np.all(files_orientation == DetectionsOrientation.HORIZONTAL)
