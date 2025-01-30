import logging
import os
from typing import Optional

import numpy as np
import supervision as sv  # type: ignore
import tqdm
from supervision.dataset.formats.yolo import yolo_annotations_to_detections
from supervision.utils.file import read_txt_file

logger = logging.getLogger(__name__)


def annotations_load_as_sv(
    images_annotations: dict[str, Optional[str]],
    dataset_path: str,
    filter_filenames: set[str],
) -> tuple[sv.Detections, list[str]]:
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
    sv_detections_list: list[sv.Detections] = []
    negative_samples: list[str] = []

    # Filter : Filter only images_annotations with the filter_filenames
    filtered_images_annotations = {
        image_path: annotation_path
        for image_path, annotation_path in images_annotations.items()
        if image_path in filter_filenames
    }

    for image_path, annotations_path in tqdm.tqdm(filtered_images_annotations.items(), desc="Loading annotations"):
        # Skip images without annotations
        if annotations_path is None:
            continue

        # Annotations : Load the annotations
        annotations_path = os.path.join(dataset_path, annotations_path)
        lines = read_txt_file(file_path=annotations_path, skip_empty=True)
        # sv.Detections : Create
        file_detections = yolo_annotations_to_detections(
            lines=lines,
            resolution_wh=(1, 1),
            with_masks=False,
            is_obb=False,
        )

        # Negative samples : Check if empty
        if len(file_detections.xyxy) == 0:
            negative_samples.append(image_path)
            continue

        # Annotated sample : Add the file path
        file_detections.data["filepaths"] = np.array([annotations_path] * len(file_detections.xyxy))
        sv_detections_list.append(file_detections)

    detections = sv.Detections.merge(sv_detections_list)
    return detections, negative_samples


def annotations_log_summary(annotations_sv: sv.Detections, negative_samples: list[str]) -> None:
    """
    Log a summary of the annotations, how many classes, how many annotations,
    For each class log count/all and %% of the total annotations, add horizontal bar
    """
    # Check : Empty
    if annotations_sv.class_id is None:
        logger.info("Annotations dataset is empty.")
        return

    # Data : parse
    unique_classes = np.unique(annotations_sv.class_id)
    total_annotations = len(annotations_sv.xyxy)
    total_files = np.unique(annotations_sv.data.get("filepaths", np.ndarray([]))).shape[0] + len(negative_samples)

    logger.info("Annotations: Found %u different annotations.", total_annotations)
    logger.info("Annotations dataset has %u classes.", len(unique_classes))
    for class_id in unique_classes:
        class_count = (annotations_sv.class_id == class_id).sum()
        class_ratio = class_count / total_annotations
        logger.info(
            " - Class %02u : %u/%u (%.2f%%) annotations",
            class_id,
            class_count,
            total_annotations,
            class_ratio * 100,
        )

    # Negative samples : Logging
    logger.info(
        " - Negative : %u/%u (%.2f%%) files",
        len(negative_samples),
        total_files,
        len(negative_samples) / total_files * 100,
    )
