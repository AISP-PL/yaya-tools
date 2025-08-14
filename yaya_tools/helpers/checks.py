import logging

import numpy as np
import supervision as sv  # type: ignore

logger = logging.getLogger(__name__)


def get_missing_class_ids(training_sv: sv.Detections, validation_sv: sv.Detections) -> tuple[set[int], set[int]]:
    """Check unique classes in training and validation sets are same.

    Zwraca brakujące identyfikatory klas jako set[int] z natywnymi Python int,
    a nie numpy int.*
    """
    classes_training: set[int] = {int(c) for c in np.unique(training_sv.class_id).tolist()}
    classes_validation: set[int] = {int(c) for c in np.unique(validation_sv.class_id).tolist()}

    missing_in_validation = classes_training - classes_validation
    missing_in_training = classes_validation - classes_training
    return missing_in_training, missing_in_validation
