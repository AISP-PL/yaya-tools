import argparse
import logging
import os
from typing import Optional

from yaya_tools import __version__
from yaya_tools.helpers.annotations import annotations_load_as_sv, annotations_log_summary
from yaya_tools.helpers.dataset import dataset_to_validation
from yaya_tools.helpers.files import is_image_file

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

    # Training annotations : Get
    annotations_sv, negative_samples = annotations_load_as_sv(images_annotations, dataset_path)

    # Training list : Logging
    logger.info("Training dataset: Found %u images.", len(train_list))
    if train_diff > 0:
        logger.warning("Training dataset : Removed %u images in update.", train_diff)
    elif train_diff < 0:
        logger.warning("Training dataset : Added %u images in update.", -train_diff)

    # Annotations : Logging
    annotations_log_summary(annotations_sv, negative_samples)

    # Training file : Save the list of training images
    with open(os.path.join(dataset_path, "train.txt"), "w") as f:
        f.write("\n".join(train_list))

    # Validation list : Check error
    if not validation_list:
        logger.fatal("Validataion dataset is empty!")

    # Validation list : Check ratio too low
    val_ratio = len(validation_list) / max(1, len(train_list) + len(validation_list))
    if val_ratio < 0.1:
        logger.warning(
            "Validation dataset to training ratio is lower <10%%! Please use --validation_force_create and --ratio (default=20%%)"
        )

    # Validation : Recreate
    if args.validation_force_create:
        validation_list = dataset_to_validation(annotations_sv, negative_samples, ratio=args.ratio)


if __name__ == "__main__":
    logging_terminal_setup()
    main_dataset()
