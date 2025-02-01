import argparse
import logging
import os
from typing import Optional

from yaya_tools import __version__
from yaya_tools.helpers.annotations import annotations_filter_filenames, annotations_load_as_sv, annotations_log_summary
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
    # Argument parser
    parser = argparse.ArgumentParser(add_help=False, description="YAYa dataset management tool")
    parser.add_argument("-s", "--source", type=str, required=True, help="Path to the source dataset folder")
    parser.add_argument("-d", "--dest", type=str, required=True, help="Path to the destination dataset folder")
    parser.add_argument(
        "--copy_new_annotations", action="store_true", help="Copy only new annotations to the destination"
    )
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("-v", action="version", version=__version__, help="Show version and exit.")
    args = parser.parse_args()

    # Source images : with optional annotation filename, filter annotated
    source_images_annotations: dict[str, Optional[str]] = load_directory_images_annotatations(args.source)
    source_images_annotated: list[str] = get_images_annotated(source_images_annotations)
    source_annotations_sv, source_negatives = annotations_load_as_sv(source_images_annotations, args.source)

    # Destination images : with optional annotation filename
    destination_images_annotations: dict[str, Optional[str]] = load_directory_images_annotatations(args.dest)
    destination_images_annotated: list[str] = get_images_annotated(destination_images_annotations)
    destination_annotations_sv, destination_negatives = annotations_load_as_sv(
        destination_images_annotations, args.dest
    )


if __name__ == "__main__":
    logging_terminal_setup()
    main()
