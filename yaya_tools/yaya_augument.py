import argparse
import logging
from typing import Optional

import supervision as sv  # types: ignore

from yaya_tools import __version__
from yaya_tools.classifiers.classifier_orientation import DetectionsOrientation
from yaya_tools.helpers.annotations import (
    annotations_filter_crowded,
    annotations_filter_equalize,
    annotations_filter_large,
    annotations_filter_orientation,
    annotations_filter_spacious,
    annotations_filter_tiny,
    annotations_load_as_sv,
    annotations_log_summary,
)
from yaya_tools.helpers.augmentations import Augumentation, augmentation_select
from yaya_tools.helpers.dataset import (
    load_directory_images_annotatations,
)
from yaya_tools.helpers.image_multiprocessing import multiprocess_augment

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
    parser.add_argument("--select_negatives", action="store_true", default=False, help="Select only  negative images")
    parser.add_argument(
        "--select_class_id", type=int, nargs="?", const=-1, default=-1, required=False, help="Select class id"
    )
    parser.add_argument(
        "--select_equalize",
        action="store_true",
        default=False,
        help="Select images to equalize dataset class representation",
    )
    parser.add_argument(
        "--select_horizontal", action="store_true", default=False, help="Select horizontal line of detections"
    )
    parser.add_argument(
        "--select_vertical", action="store_true", default=False, help="Select vertical line of detections"
    )
    parser.add_argument(
        "--select_diagonal_right", action="store_true", default=False, help="Select diagonal right line of detections"
    )
    parser.add_argument(
        "--select_diagonal_left", action="store_true", default=False, help="Select diagonal left line of detections"
    )
    parser.add_argument("--select_large", action="store_true", default=False, help="Select large annotations")
    parser.add_argument("--select_tiny", action="store_true", default=False, help="Select small annotations")
    parser.add_argument("--select_crowded", action="store_true", default=False, help="Select crowded scenes")
    parser.add_argument("--select_spacious", action="store_true", default=False, help="Select spacious scenes")
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        nargs="?",
        const=300,
        default=300,
        required=False,
        help="Maximum number of created images",
    )
    parser.add_argument(
        "--flip_horizontal",
        action="store_true",
        required=False,
        help="Flip horizontal image.",
    )
    parser.add_argument(
        "--flip_vertical",
        action="store_true",
        required=False,
        help="Flip vertical image.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs="?",
        const=-1,
        default=-1,
        required=False,
        help="Augument by random Crop image (for ex 640).",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        nargs="?",
        const=-1,
        default=-1,
        required=False,
        help="Augument by direct degrees rotation (for ex 90).",
    )
    parser.add_argument(
        "--randrotate",
        type=int,
        nargs="?",
        const=0,
        default=0,
        required=False,
        help="Random rotation from -degrees to degrees.",
    )
    parser.add_argument(
        "--brighten",
        action="store_true",
        required=False,
        help="Random make image brighten and adjust contrast.",
    )
    parser.add_argument(
        "--sharpen",
        action="store_true",
        required=False,
        help="Random make image sharpen.",
    )
    parser.add_argument(
        "--darken",
        action="store_true",
        required=False,
        help="Random make image darkne and adjust contrast.",
    )
    parser.add_argument("--clahe", action="store_true", required=False, help="Apply CLAHE to image.")
    parser.add_argument(
        "--equalize",
        action="store_true",
        required=False,
        help="Equalize image.",
    )
    parser.add_argument(
        "--colorshift",
        action="store_true",
        required=False,
        help="Random color shift in image ",
    )
    parser.add_argument(
        "--isonoise",
        action="store_true",
        required=False,
        help="Random add iso noise to image. Caution!! Hard!",
    )
    parser.add_argument(
        "--gaussnoise",
        action="store_true",
        required=False,
        help="Random add gauss noise.",
    )
    parser.add_argument(
        "--multi_noise",
        action="store_true",
        required=False,
        help="Random multi gauss noise.",
    )
    parser.add_argument(
        "--downsize_padding",
        action="store_true",
        required=False,
        help="Downsize with black padding.",
    )
    parser.add_argument(
        "--compression",
        action="store_true",
        required=False,
        help="compression image quality.",
    )
    parser.add_argument(
        "--degrade",
        action="store_true",
        required=False,
        help="Degrade image quality.",
    )
    parser.add_argument("--spatter", action="store_true", required=False, help="Spatter add.")
    parser.add_argument("--spatter_big", action="store_true", required=False, help="Spatter add.")
    parser.add_argument("--spatter_small", action="store_true", required=False, help="Spatter add.")
    parser.add_argument(
        "--blackboxing",
        required=False,
        type=int,
        nargs="?",
        const=-1,
        default=-1,
        help="Blackboxing HxH parts of image.",
    )
    parser.add_argument("--snow", action="store_true", required=False, help="Snow add.")
    parser.add_argument("--rain", action="store_true", required=False, help="Rain add.")
    parser.add_argument("--fog", action="store_true", required=False, help="Fog add.")
    parser.add_argument("--sunflare", action="store_true", required=False, help="Sunflare add.")
    parser.add_argument("--blur", action="store_true", required=False, help="Blur image.")
    parser.add_argument(
        "--blur_delicate",
        action="store_true",
        required=False,
        help="Blur delicate image.",
    )
    parser.add_argument("--flip", action="store_true", required=False, help="Flip randomly image.")
    parser.add_argument(
        "-mb",
        "--medianblur",
        action="store_true",
        required=False,
        help="Median blur image.",
    )
    parser.add_argument("--grayscale", action="store_true", required=False, help="Convert image to grayscale.")
    parser.add_argument("--sepia", action="store_true", required=False, help="Convert image to sepia.")
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("-v", action="version", version=__version__, help="Show version and exit.")
    args = parser.parse_args()

    # All images : with optional annotation filename
    all_images_annotations: dict[str, Optional[str]] = load_directory_images_annotatations(args.dataset_path)

    # All annotations as SV : Get
    all_annotations_sv, all_negatives = annotations_load_as_sv(all_images_annotations, args.dataset_path)

    # Dataset : Logging
    annotations_log_summary("Dataset", all_annotations_sv, all_negatives)

    selected_annotations = all_annotations_sv
    selected_negatives = all_negatives
    # Select : Only negatives
    if args.select_negatives:
        selected_annotations = sv.Detections.empty()
        selected_negatives = selected_negatives
        logger.info("Selection : Only negatives.")

    # Select : Only single class id
    if args.select_class_id != -1:
        selected_annotations = selected_annotations[selected_annotations.class_id == args.select_class_id]  # type: ignore
        selected_negatives = []
        logger.info(f"Selection : Only class id {args.select_class_id}.")

    # Select : Equalize class representation
    if args.select_equalize:
        selected_negatives = []
        selected_annotations = annotations_filter_equalize(selected_annotations)
        logger.info("Selection : Equalize class representation.")

    # Select : Horizontal line of detections
    if args.select_horizontal:
        selected_negatives = []
        selected_annotations = annotations_filter_orientation(selected_annotations, DetectionsOrientation.HORIZONTAL)
        logger.info("Selection : Horizontal line of detections.")

    # Select : Vertical line of detections
    if args.select_vertical:
        selected_negatives = []
        selected_annotations = annotations_filter_orientation(selected_annotations, DetectionsOrientation.VERTICAL)
        logger.info("Selection : Vertical line of detections.")

    # Select : Diagonal right line of detections
    if args.select_diagonal_right:
        selected_negatives = []
        selected_annotations = annotations_filter_orientation(
            selected_annotations, DetectionsOrientation.DIAGONAL_RIGHT
        )
        logger.info("Selection : Diagonal right line of detections.")

    # Select : Diagonal left line of detections
    if args.select_diagonal_left:
        selected_negatives = []
        selected_annotations = annotations_filter_orientation(selected_annotations, DetectionsOrientation.DIAGONAL_LEFT)
        logger.info("Selection : Diagonal left line of detections.")

    # Select : Large annotations
    if args.select_large:
        selected_annotations = annotations_filter_large(selected_annotations)
        selected_negatives = []
        logger.info("Selection : Large annotations.")

    # Select : Tiny annotations
    if args.select_tiny:
        selected_annotations = annotations_filter_tiny(selected_annotations)
        selected_negatives = []
        logger.info("Selection : Tiny annotations.")

    # Select : Crowded scenes
    if args.select_crowded:
        selected_annotations = annotations_filter_crowded(selected_annotations)
        selected_negatives = []
        logger.info("Selection : Crowded scenes.")

    # Select : Spacious scenes
    if args.select_spacious:
        selected_annotations = annotations_filter_spacious(selected_annotations)
        selected_negatives = []
        logger.info("Selection : Spacious scenes.")

    # Log : How many annotations are selected
    annotations_log_summary("Selected", selected_annotations, selected_negatives)

    # Augmentation : Select
    augmentation: Optional[Augumentation] = augmentation_select(args)
    if augmentation is None:
        logger.error("No augmentation selected.")
        return

    # Augmentation : Multiprocess
    multiprocess_augment(
        args.dataset_path,
        selected_annotations,
        selected_negatives,
        all_annotations_sv,
        args.iterations,
        augmentation,
    )


if __name__ == "__main__":
    main()
