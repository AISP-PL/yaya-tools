import logging
import os
from typing import Optional

import numpy as np

from yaya_tools.helpers.files import is_image_file

logger = logging.getLogger(__name__)


def main_test() -> None:
    """Test function for package installation tests"""
    print("yaya_tools package installed successfully!")


def main_dataset(dataset_path: str, validation_recreate: bool = False, ratio: float = 0.2) -> None:
    """
    Main function for dataset management

    Arguments
    ----------
    dataset_path : str
        Path to the dataset folder
    validation_recreate : bool
        If True, recreate validation.txt file
    ratio : float
        Validation ratio (default=0.2)

    Returns
    -------
    None
    """
    # Training dataset file : Load the list of training images
    train_list: list[str] = []
    try:
        with open(os.path.join(dataset_path, "train.txt"), "r") as f:
            train_list = [p.strip() for p in f if p.strip()]
    except FileNotFoundError:
        logger.error("train.txt file not found!")

    # Validation dataset file : Load the list of validation images
    validation_list: list[str] = []
    try:
        with open(os.path.join(dataset_path, "validation.txt"), "r") as f:
            validation_list = [p.strip() for p in f if p.strip()]
    except FileNotFoundError:
        logger.error("validation.txt file not found!")

    # Images : List all images in the dataset folder
    images_annotations: dict[str, Optional[str]] = {}
    for file_name in os.listdir(dataset_path):
        # Skip non-image files
        if not is_image_file(file_name):
            continue

        # Image : Exists, annotation to check
        images_annotations[file_name] = None

        # Annotation : Exists, overwrite the annotation file
        annotation_file = os.path.splitext(file_name)[0] + ".txt"
        if os.path.exists(os.path.join(dataset_path, annotation_file)):
            images_annotations[file_name] = annotation_file

    # Images annotated : Create list
    images_annotated: list[str] = [
        img_path for img_path, annotation_path in images_annotations.items() if annotation_path is not None
    ]
    # Training list : Set to all annotated images without validation images
    train_list_orig = train_list.copy()
    train_list = [img_path for img_path in images_annotated if img_path not in validation_list]
    train_diff = len(train_list_orig) - len(train_list)

    # Training list : Logging
    logger.info("Training images : Found %u", len(train_list))
    if train_diff > 0:
        logger.warning("Training images : Removed %u images in update.", train_diff)
    elif train_diff < 0:
        logger.warning("Training images : Added %u images in update.", -train_diff)

    # Training file : Save the list of training images
    with open(os.path.join(dataset_path, "train.txt"), "w") as f:
        f.write("\n".join(train_list))

    # Validation list : Check error
    if not validation_list:
        logger.error("validation.txt is empty!")

    # Validation list : Check ratio too low
    val_ratio = len(validation_list) / max(1, len(train_list) + len(validation_list))
    if val_ratio < 0.1:
        logger.warning("Validation ratio <10%%, use --validation_recreate and --ratio (default=20%%)")

    all_annotations = []
    for file_name in images_existing:
        annotation_path = os.path.join(dataset_path, os.path.splitext(file_name)[0] + ".txt")
        with open(annotation_path, "r") as ann_file:
            for line in ann_file:
                class_id, *coords = line.split()
                all_annotations.append((class_id, coords, file_name))

    annotations_np = np.array(all_annotations, dtype=object)
    unique_classes, counts = np.unique(annotations_np[:, 0], return_counts=True)
    total_ann = counts.sum()
    for cls, cnt in zip(unique_classes, counts):
        pct = cnt / total_ann * 100 if total_ann else 0
        bar = "#" * int(pct // 2)
        logger.info("Class %s: %d (%.1f%%) %s", cls, cnt, pct, bar)

    if validation_recreate:
        dataset_to_validation()
