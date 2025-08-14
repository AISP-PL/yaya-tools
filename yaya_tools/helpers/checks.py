import logging

import numpy as np
import supervision as sv  # type: ignore

logger = logging.getLogger(__name__)


def get_missing_class_ids(training_sv: sv.Detections, validation_sv: sv.Detections) -> tuple[set[int], set[int]]:
    """Check unique classes in training and validation sets are same."""
    classes_training = np.unique(training_sv.class_id)
    classes_validation = np.unique(validation_sv.class_id)

    # Get difference as set
    missing_in_validation = set(classes_training) - set(classes_validation)
    missing_in_training = set(classes_validation) - set(classes_training)
    return missing_in_training, missing_in_validation
