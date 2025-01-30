import argparse
import logging
import os
from typing import Optional

from yaya_tools import __version__
from yaya_tools.helpers.annotations import annotations_filter_filenames, annotations_load_as_sv, annotations_log_summary
from yaya_tools.helpers.dataset import (
    dataset_to_validation,
    get_images_annotated,
    load_directory_images_annotatations,
    load_file_to_list,
)

logger = logging.getLogger(__name__)


def logging_terminal_setup() -> None:
    """
    Setup logging for the application.

    Parameters
    ----------
    path_field : str
        Field in the config file that contains the path to the log file.
        Default is "path".
    is_terminal : bool
        If True, logs will be printed to the terminal.
        Default is True.
    """
    logging.getLogger().setLevel(logging.DEBUG)  # Ensure log level is set to DEBUG
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    logging.info("\n\n###### Logging start of terminal session ######\n")


def main_test() -> None:
    """Test function for package installation tests"""
    print("yaya_tools package installed successfully!")


def main_dataset() -> None:
    """
    Main function for dataset management

    Arguments
    ----------
    dataset_path : str
        Path to the dataset folder
    validation_force_create : bool
        If True, recreate validation.txt file
    ratio : float
        Validation ratio (default=0.2)

    Returns
    -------
    None
    """
    # Argument parser
    parser = argparse.ArgumentParser(add_help=False, description="YAYa dataset management tool")
    parser.add_argument("-i", "--dataset_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument(
        "--validation_force_create",
        action="store_true",
        help="Recreate the validation.txt file from the dataset folder",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Validation ratio (default=0.2)",
    )
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("-v", action="version", version=__version__, help="Show version and exit.")
    args = parser.parse_args()

    dataset_path = args.dataset_path

    # Training : List of filenames
    train_list: list[str] = load_file_to_list(os.path.join(dataset_path, "train.txt"))
    # Validation : List of filenames
    validation_list: list[str] = load_file_to_list(os.path.join(dataset_path, "validation.txt"))
    # All images : with optional annotation filename
    all_images_annotations: dict[str, Optional[str]] = load_directory_images_annotatations(dataset_path)
    # Images annotated : Filter only
    images_annotated: list[str] = get_images_annotated(all_images_annotations)

    # All annotations as SV : Get
    all_annotations_sv, all_negatives = annotations_load_as_sv(all_images_annotations, dataset_path)

    # Validation : Recreate
    if args.validation_force_create:
        # Validation dataset  : Create as list of files
        validation_list = dataset_to_validation(all_annotations_sv, all_negatives, ratio=args.ratio)

    # Train and val annotations : Filter
    training_annotations_sv, training_negatives = annotations_filter_filenames(
        all_annotations_sv, all_negatives, train_list
    )
    validations_sv, validation_negative = annotations_filter_filenames(
        all_annotations_sv, all_negatives, validation_list
    )

    # Training list : Set to all annotated images without validation images
    train_list_orig = train_list.copy()
    train_list = [img_path for img_path in images_annotated if img_path not in validation_list]
    train_diff = len(train_list_orig) - len(train_list)

    # Training list : Logging
    logger.info("Directory has annotated %u of %u total images.", len(images_annotated), len(all_images_annotations))
    logger.info("Training dataset has %u images.", len(train_list))
    if train_diff > 0:
        logger.warning("Training dataset deleted %u images in update.", train_diff)
    elif train_diff < 0:
        logger.warning("Training dataset added %u images in update.", -train_diff)

    # Annotations : Logging
    annotations_log_summary(training_annotations_sv, training_negatives)

    # Training file : Save the list of training images
    with open(os.path.join(dataset_path, "train.txt"), "w") as f:
        f.write("\n".join(train_list))

    # Validation file : Save the list of validation images
    with open(os.path.join(dataset_path, "validation.txt"), "w") as f:
        f.write("\n".join(validation_list))

    # Validation list : Check error
    if not validation_list:
        logger.fatal("Validation dataset is empty!")

    # Validation list : Check ratio too low
    total_train_valid = len(images_annotated)
    val_ratio = len(validation_list) / max(1, total_train_valid)
    if val_ratio < 0.1:
        logger.warning(
            "Validation dataset to training ratio is lower <10%%! Please use --validation_force_create and --ratio (default=20%%)"
        )

    # Validation annotations : Get
    validations_sv, validation_negative = annotations_load_as_sv(
        all_images_annotations, dataset_path, filter_filenames=set(validation_list)
    )

    # Validation list : Logging
    logger.info("Validation dataset: Found %u/%u (%.2f%%).", len(validation_list), total_train_valid, val_ratio * 100)
    annotations_log_summary(validations_sv, validation_negative)


if __name__ == "__main__":
    logging_terminal_setup()
    main_dataset()
