import argparse
import logging
from typing import Optional

import supervision as sv  # types: ignore

from yaya_tools import __version__
from yaya_tools.helpers.annotations import annotations_load_as_sv, annotations_log_summary
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
    parser.add_argument("--select_all", action="store_true", default=True, help="Select all images (default: True)")
    parser.add_argument("--select_negatives", action="store_true", default=False, help="Select only  negative images")
    parser.add_argument(
        "--select_equalize",
        action="store_true",
        default=False,
        help="Select images to equalize dataset class representation",
    )
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
        const=0,
        default=0,
        required=False,
        help="Augument by random Crop image (for ex 640).",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        nargs="?",
        const=0,
        default=0,
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
        help="Random add iso noise to image.",
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
        const=0,
        default=0,
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
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("-v", action="version", version=__version__, help="Show version and exit.")
    args = parser.parse_args()

    # All images : with optional annotation filename
    all_images_annotations: dict[str, Optional[str]] = load_directory_images_annotatations(args.dataset_path)

    # All annotations as SV : Get
    all_annotations_sv, all_negatives = annotations_load_as_sv(all_images_annotations, args.dataset_path)

    # Dataset : Logging
    annotations_log_summary("Dataset", all_annotations_sv, all_negatives)

    # Selection :
    if args.select_all:
        selected_annotations = all_annotations_sv
        selected_negatives = all_negatives
    elif args.select_negatives:
        selected_annotations = sv.Detections.empty()
        selected_negatives = all_negatives

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
        args.iterations,
        augmentation,
    )


if __name__ == "__main__":
    main()
