"""
Test function that loads all annotated images from directory path,
then it runs the detector on each image, finally it calculates the mAP.
"""

import logging
import tempfile

import numpy as np
import supervision as sv

from yaya_tools.detection.detector_yolov4_cvdnn import DetectorCVDNN
from yaya_tools.helpers.yolo_yaml import create_yolo_yaml_str  # type: ignore

logger = logging.getLogger(__name__)


def callback_detect(model: DetectorCVDNN, image: np.ndarray) -> sv.Detections:
    """Callback function that runs the detector on the image."""
    return model.detect(1, image)


def dataset_benchmark(dataset_path: str, detector: DetectorCVDNN) -> tuple[sv.MeanAveragePrecision, float]:
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

    # mAP : Calculate
    mAP = sv.MeanAveragePrecision.benchmark(dataset=dataset, callback=lambda image: callback_detect(detector, image))

    # Negatives score : Calculate accuracy of False Positives in empty images
    negatives_accuracy: list[float] = []
    negatives_filenames: list[str] = [
        filepath for filepath, annotations_sv in dataset.annotations.items() if annotations_sv == sv.Detections.empty()
    ]

    # Check : No negatives, return mAP and 1.0
    if len(negatives_filenames) == 0:
        return mAP, 1.0

    for filepath in negatives_filenames:
        # Detect, add 1.0 if ok
        detections = callback_detect(detector, dataset.images.get(filepath, np.ndarray([0, 0, 3], dtype=np.uint8)))
        if detections == sv.Detections.empty():
            negatives_accuracy.append(1.0)
            continue

        # Any detections, add 0.0 accuracy
        negatives_accuracy.append(0.0)

    return mAP, np.mean(np.array(negatives_accuracy))


def log_map(mAP: sv.MeanAveragePrecision, negatives_ap: float) -> None:
    """Log most important mAP metrics"""
    logger.info(f"mAP: {mAP.map50_95}")
    logger.info(f"mAP 50: {mAP.map50}")
    logger.info(f"mAP 75: {mAP.map75}")
    logger.info(f"AP on negative images: {negatives_ap}")

    # Log per class AP
    for index in range(len(mAP.per_class_ap50_95)):
        logger.info(f"AP {index}: {mAP.per_class_ap50_95[index][4]}")
