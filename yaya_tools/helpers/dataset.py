"""
This module contains helper functions for dataset management
"""

import logging

import numpy as np
import supervision as sv  # type: ignore

logger = logging.getLogger(__name__)


def dataset_to_validation(annotations_sv: sv.Detections, negative_samples: list[str], ratio: float = 0.20) -> list[str]:
    """
    Create a validation files list from basing on all annotations
    """
    # Check : No annotations
    if annotations_sv.class_id is None:
        logger.error("No annotations found!")
        return []

    # Data : Get from the annotations
    annotations_filepaths: np.ndarray = annotations_sv.data.get("filepaths", np.array([]))
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
        validation_files.update(class_files[:class_size])

    # Samples : Add negative samples
    np.random.shuffle(negative_samples)
    validation_files.update(negative_samples[:class_size])

    # Remaining : Add random samples
    files_missing = validation_size - len(validation_files)
    if files_missing > 0:
        remaining_files = list(unique_files - validation_files)
        np.random.shuffle(remaining_files)
        validation_files.update(remaining_files[:files_missing])

    validation_files_list = list(validation_files)
    return validation_files_list[:validation_size]
