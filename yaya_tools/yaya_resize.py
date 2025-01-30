import argparse
import logging
from typing import Optional

from yaya_tools import __version__
from yaya_tools.helpers.dataset import (
    get_images_annotated,
    load_directory_images_annotatations,
)
from yaya_tools.helpers.image_multiprocessing import multiprocess_resize

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
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to the output folder")
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("-v", action="version", version=__version__, help="Show version and exit.")
    args = parser.parse_args()

    # All images : with optional annotation filename
    all_images_annotations: dict[str, Optional[str]] = load_directory_images_annotatations(args.dataset_path)
    # Images annotated : Filter only
    images_annotated: list[str] = get_images_annotated(all_images_annotations)

    succes_files, failed_files = multiprocess_resize(
        source_directory=args.dataset_path, target_directory=args.output_path, images_names=images_annotated
    )


if __name__ == "__main__":
    logging_terminal_setup()
    main()
