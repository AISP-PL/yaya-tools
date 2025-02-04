"""
Test function that loads all annotated images from directory path,
then it runs the detector on each image, finally it calculates the mAP.
"""

import logging
import tempfile
from pathlib import Path

import cv2  # type: ignore
import numpy as np
import supervision as sv
import tqdm

from yaya_tools.detection.detector_yolov4_cvdnn import DetectorCVDNN
from yaya_tools.helpers.yolo_yaml import create_yolo_yaml_str  # type: ignore

logger = logging.getLogger(__name__)


def annotate_image_results(
    filepath: str, image: np.ndarray, annotations: sv.Detections, detections: sv.Detections
) -> None:
    """Annotate image with detections and annotations"""

    # Subdirectory: Create if not exists
    images_directory = Path(filepath).parent / ".results"
    Path(images_directory).mkdir(parents=True, exist_ok=True)

    # Ground truth : Anntoate as green boxes
    box_annotator_gt = sv.BoxAnnotator(color=sv.Color.GREEN)
    # Predictions : Annotate as red boxes
    box_annotator_pred = sv.BoxAnnotator(color=sv.Color.RED)
    # Labels : Annotate as text
    label_annotator_gt = sv.LabelAnnotator(color=sv.Color.GREEN, text_padding=5)
    label_annotator_pred = sv.LabelAnnotator(color=sv.Color.RED, text_padding=5, text_position=sv.Position.TOP_RIGHT)

    # Annotate image : Boxes and labels
    annotated_image = box_annotator_gt.annotate(image.copy(), annotations)
    annotated_image = label_annotator_gt.annotate(annotated_image, annotations)
    annotated_image = box_annotator_pred.annotate(annotated_image, detections)
    annotated_image = label_annotator_pred.annotate(annotated_image, detections)

    # Save image :
    image_path = images_directory / Path(filepath).name
    cv2.imwrite(str(image_path), annotated_image)


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

    # Dataset : Predict all
    predictions: list[sv.Detections] = []
    targets: list[sv.Detections] = []
    negatives_detections: list[sv.Detections] = []
    for imagename, image, annotation in tqdm.tqdm(dataset, desc="Detecting images", total=len(dataset)):
        predictions_batch = detector.detect(frame_number=1, frame=image)

        # Append to lists
        predictions.append(predictions_batch)
        targets.append(annotation)
        annotate_image_results(filepath=imagename, image=image, annotations=annotation, detections=predictions_batch)

        # Check : No annotations, negative
        if annotation == sv.Detections.empty():
            negatives_detections.append(predictions_batch)

    # mAP : Calculate
    mAP = sv.MeanAveragePrecision.from_detections(
        predictions=predictions,
        targets=targets,
    )

    # No negatives, return 1.0
    if len(negatives_detections) == 0:
        return mAP, 1.0

    # Negatives : Calculate accuracy
    negatives_accuracy: list[float] = [
        1.0 if detections == sv.Detections.empty() else 0.0 for detections in negatives_detections
    ]

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
