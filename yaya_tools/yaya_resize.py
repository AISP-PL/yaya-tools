import argparse
import logging
from typing import Optional

from yaya_tools import __version__
from yaya_tools.helpers.dataset import (
    get_images_annotated,
    load_directory_images_annotatations,
)
from yaya_tools.helpers.image_multiprocessing import multiprocess_resize
from yaya_tools.helpers.terminal_logging import logging_terminal_setup

logger = logging.getLogger(__name__)


def main() -> None:
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
    logging_terminal_setup()
    # Argument parser
    parser = argparse.ArgumentParser(add_help=False, description="YAYa dataset management tool")
    parser.add_argument("-i", "--dataset_path", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--width", type=int, default=640, help="Width of the resized image")
    parser.add_argument("--height", type=int, default=640, help="Height of the resized image")
    parser.add_argument(
        "--keep_aspect_ratio", action="store_true", help="Keep aspect ratio of the resized image (default: False)"
    )
    parser.add_argument(
        "--copy_annotations", action="store_true", help="Copy annotations if found to the output folder"
    )
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("-v", action="version", version=__version__, help="Show version and exit.")
    args = parser.parse_args()

    # All images : with optional annotation filename
    all_images_annotations: dict[str, Optional[str]] = load_directory_images_annotatations(args.dataset_path)
    # Images annotated : Filter only
    images_annotated: list[str] = get_images_annotated(all_images_annotations)

    succes_files, failed_files = multiprocess_resize(
        source_directory=args.dataset_path,
        target_directory=args.output_path,
        images_names=images_annotated,
        new_width=args.width,
        new_height=args.height,
        keep_aspect_ratio=args.keep_aspect_ratio,
        copy_annotations=args.copy_annotations,
    )

    logger.info(f"Successfully processed {len(succes_files)} files")
    if failed_files:
        logger.error(f"Failed to process {','.join(failed_files)}")


if __name__ == "__main__":
    main()
