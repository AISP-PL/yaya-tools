import argparse
import logging
import os
from typing import Optional

from yaya_tools import __version__
from yaya_tools.helpers.annotations import (
    annotations_filter_filenames,
    annotations_filter_warnings,
    annotations_load_as_sv,
    annotations_log_summary,
)
from yaya_tools.helpers.dataset import (
    dataset_copy_to,
    dataset_create_validation,
    dataset_log_summary,
    get_images_annotated,
    load_directory_images_annotatations,
    load_file_to_list,
    save_list_to_file,
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


def main() -> None:
    """
    Main function for dataset management

    Returns
    -------
    None
    """
    logging_terminal_setup()
    # Argument parser
    parser = argparse.ArgumentParser(add_help=False, description="YAYa dataset management tool")
    parser.add_argument("-i", "--dataset_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--fix_annotations", action="store_true", help="Fix annotations warnings in the dataset folder")
    parser.add_argument("--copy_negatives_to", type=str, help="Path to copy the negative samples only")
    parser.add_argument("--train_all", action="store_true", help="Use all images for training dataset")
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

    # Warnings : Check and get
    warnings_annotations = annotations_filter_warnings(all_annotations_sv)

    # Negatives : Extract
    if args.copy_negatives_to:
        dataset_copy_to(args.dataset_path, all_negatives, args.copy_negatives_to)

    # Validation : Recreate
    if args.validation_force_create:
        # Validation dataset  : Create as list of files
        validation_list = dataset_create_validation(all_annotations_sv, all_negatives, ratio=args.ratio)

    # Training list : Set to all annotated images when --train_all is True, else remove validation images
    train_list_orig = train_list.copy()
    if args.train_all:
        train_list = images_annotated
    else:
        train_list = [img_path for img_path in images_annotated if img_path not in validation_list]
    train_diff = len(train_list_orig) - len(train_list)

    # Train and val annotations : Filter
    training_annotations_sv, training_negatives = annotations_filter_filenames(
        all_annotations_sv, all_negatives, train_list
    )
    validations_sv, validation_negative = annotations_filter_filenames(
        all_annotations_sv, all_negatives, validation_list
    )

    # Dataset : Logging
    dataset_log_summary(
        all_images=len(all_images_annotations),
        all_images_annotated=len(images_annotated),
        train_list_size=len(train_list),
        valid_list_size=len(validation_list),
        train_added=max(0, train_diff),
        train_deleted=max(0, -train_diff),
    )

    # Train and validation list : Save
    save_list_to_file(os.path.join(dataset_path, "train.txt"), train_list)
    save_list_to_file(os.path.join(dataset_path, "validation.txt"), validation_list)

    # Training : Logging summary
    annotations_log_summary("Training", training_annotations_sv, training_negatives)

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

    # Validation list : Logging
    annotations_log_summary("Valid", validations_sv, validation_negative)


if __name__ == "__main__":
    main()
