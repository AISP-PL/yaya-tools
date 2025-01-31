"""
This module contains helper functions for dataset management
"""

import logging
import os
from shutil import copyfile
from typing import Optional

import numpy as np
import supervision as sv
import tqdm

from yaya_tools.helpers.files import is_image_file  # type: ignore

logger = logging.getLogger(__name__)


def dataset_create_validation(
    annotations_sv: sv.Detections, negative_samples: list[str], ratio: float = 0.20
) -> list[str]:
    """
    Create a validation files list from basing on all annotations
    """
    # Check : No annotations
    if annotations_sv.class_id is None:
        logger.error("No annotations found!")
        return []

    # Data : Get from the annotations
    annotations_filepaths: np.ndarray = np.array(annotations_sv.data.get("filepaths", []))
    unique_classes = np.unique(annotations_sv.class_id)
    unique_files = set(annotations_filepaths.tolist())
    total_files = len(unique_files) + len(negative_samples)

    # Split : number of samples for validation
    validation_size = max(1, round(ratio * total_files))

    # Samples : Select same % of samples from each class_id and negative samples also
    # to have equal distribution of classes. If not enough samples, then do second round
    # with random file selection.
    class_size = max(1, round(validation_size / (len(unique_classes) + 1)))

    # Samples : Add every class_id samples
    validation_files: set[str] = set()
    for class_id in unique_classes:
        class_files = annotations_filepaths[annotations_sv.class_id == class_id]
        np.random.shuffle(class_files)
        validation_files.update(class_files[:class_size].tolist())

    # Samples : Add negative samples
    np.random.shuffle(negative_samples)
    validation_files.update(negative_samples[:class_size])

    # Remaining : Add random samples
    files_missing = validation_size - len(validation_files)
    if files_missing > 0:
        remaining_files = list(unique_files - validation_files)
        np.random.shuffle(remaining_files)
        validation_files.update(remaining_files[:files_missing])

    # Truncate : To the validation size
    validation_files_list = list(validation_files)[:validation_size]

    # Logging : Summary
    logger.info(
        "Calculated validation dataset size is %u. Resulted size is %u.", validation_size, len(validation_files_list)
    )
    logger.info("Calculated equal class representation size is %u.", class_size)
    logger.info("For equal representation missed %u files and filled randomly.", max(0, files_missing))

    # Return : List
    return validation_files_list


def dataset_copy_to(dataset_path: str, files_to_copy: list[str], destination_path: str) -> None:
    """
    Copy files from the dataset to the destination folder
    """
    os.makedirs(destination_path, exist_ok=True)

    success_copies: int = 0
    for file_name in tqdm.tqdm(files_to_copy, desc="Copying files"):
        source_file = os.path.join(dataset_path, file_name)
        destination_file = os.path.join(destination_path, file_name)
        try:
            copyfile(source_file, destination_file)
            success_copies += 1
        except Exception as e:
            logger.error(f"Error copying {source_file} to {destination_file}: {e}")
            continue

    logger.info("Copied %u files to %s.", success_copies, destination_path)


def load_file_to_list(file_path: str) -> list[str]:
    """
    Load a text file to a list of strings
    """
    train_list: list[str] = []
    try:
        with open(file_path, "r") as f:
            train_list = [p.strip() for p in f if p.strip()]
    except FileNotFoundError:
        logger.error(f"{file_path} not found!")

    # Sort : List
    train_list = sorted(train_list)
    return train_list


def save_list_to_file(file_path: str, file_list: list[str]) -> None:
    """
    Save a list of strings to a text file
    """
    files_sorted = sorted(file_list)
    with open(file_path, "w") as f:
        f.write("\n".join(files_sorted))


def load_directory_images_annotatations(dataset_path: str) -> dict[str, Optional[str]]:
    """Load all images and their annotations from the dataset directory"""

    # Images : List all images in the dataset folder
    all_images_annotations: dict[str, Optional[str]] = {}
    for file_name in os.listdir(dataset_path):
        # Skip non-image files
        if not is_image_file(file_name):
            continue

        # Image : Exists, annotation to check
        all_images_annotations[file_name] = None

        # Annotation : Exists, overwrite the annotation file
        annotation_file = os.path.splitext(file_name)[0] + ".txt"
        if os.path.exists(os.path.join(dataset_path, annotation_file)):
            all_images_annotations[file_name] = annotation_file

    return all_images_annotations


def get_images_annotated(all_images_annotations: dict[str, Optional[str]]) -> list[str]:
    """Get a list of images that have annotations"""
    return [img_path for img_path, annotation_path in all_images_annotations.items() if annotation_path is not None]


def get_images_not_annotated(all_images_annotations: dict[str, Optional[str]]) -> list[str]:
    """Get a list of images that do not have annotations"""
    return [img_path for img_path, annotation_path in all_images_annotations.items() if annotation_path is None]


def dataset_log_summary(
    all_images: int,
    all_images_annotated: int,
    train_list_size: int,
    valid_list_size: int,
    train_added: int,
    train_deleted: int,
) -> None:
    """Log dataset information"""
    annotated_ratio = all_images_annotated / all_images if all_images > 0 else 0
    logger.info(
        "Directory has annotated %u/%u images (%.2f%%).", all_images_annotated, all_images, annotated_ratio * 100
    )
    logger.info(
        "Training [%u images] + Validation [%u images] = Total [%u images].",
        train_list_size,
        valid_list_size,
        train_list_size + valid_list_size,
    )
    if train_added > 0:
        logger.warning("Training dataset added %u new images.", train_added)

    if train_deleted > 0:
        logger.warning("Training dataset removed %u missing images.", train_deleted)
