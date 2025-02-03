"""
Test function that loads all annotated images from directory path,
then it runs the detector on each image, finally it calculates the mAP.
"""

import numpy as np
import supervision as sv

from yaya_tools.detection.detector_yolov4_cvdnn import DetectorCVDNN  # type: ignore


def callback_detect(model: DetectorCVDNN, image: np.ndarray) -> sv.Detections:
    """Callback function that runs the detector on the image."""
    return model.detect(1, image)


def test_detector(dataset_path: str, detector: DetectorCVDNN) -> sv.ConfusionMatrix:
    """Test function that loads all annotated images from directory path,
    then it runs the detector on each image, finally it calculates the mAP.
    """
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=dataset_path, annotations_directory_path=dataset_path, data_yaml_path=""
    )

    # Confusion matrix : Calculate
    confusion_matrix = sv.ConfusionMatrix.benchmark(
        dataset=dataset, callback=lambda image: callback_detect(detector, image)
    )
    return confusion_matrix
