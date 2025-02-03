"""
Test function that loads all annotated images from directory path,
then it runs the detector on each image, finally it calculates the mAP.
"""

import tempfile

import numpy as np
import supervision as sv

from yaya_tools.detection.detector_yolov4_cvdnn import DetectorCVDNN
from yaya_tools.helpers.yolo_yaml import create_yolo_yaml_str  # type: ignore


def callback_detect(model: DetectorCVDNN, image: np.ndarray) -> sv.Detections:
    """Callback function that runs the detector on the image."""
    return model.detect(1, image)


def test_detector(dataset_path: str, detector: DetectorCVDNN) -> sv.ConfusionMatrix:
    """Test function that loads all annotated images from directory path,
    then it runs the detector on each image, finally it calculates the mAP.
    """
    # YAML : Create temporary yaml file in temporary location, get classes from detector.
    classes = detector.classes
    data_yaml_path = ""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
        yaml_str = create_yolo_yaml_str(
            train_path="", val_path="", test_path="", classes=classes, annotation_format="coco"
        )
        file.write(yaml_str)
        data_yaml_path = file.name

    # Sv: Load as dataset
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=dataset_path, annotations_directory_path=dataset_path, data_yaml_path=data_yaml_path
    )

    # Confusion matrix : Calculate
    confusion_matrix = sv.ConfusionMatrix.benchmark(
        dataset=dataset, callback=lambda image: callback_detect(detector, image)
    )
    return confusion_matrix
