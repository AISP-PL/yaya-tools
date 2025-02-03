import argparse
import logging
from typing import Optional

from yaya_tools import __version__
from yaya_tools.helpers.annotations import (
    annotations_append,
    annotations_diff,
    annotations_load_as_sv,
    annotations_log_summary,
)
from yaya_tools.helpers.dataset import (
    images_annotations_log,
    load_directory_images_annotatations,
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
    parser.add_argument("-s", "--source", type=str, required=True, help="Path to the source dataset folder")
    parser.add_argument("-d", "--dest", type=str, required=True, help="Path to the destination dataset folder")
    parser.add_argument("--add_new", action="store_true", help="Add only new annotations to the destination")
    parser.add_argument("--remove_old", action="store_true", help="Remove only old annotations from the destination")
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("-v", action="version", version=__version__, help="Show version and exit.")
    args = parser.parse_args()

    # Source images : with optional annotation filename, filter annotated
    source_images_annotations: dict[str, Optional[str]] = load_directory_images_annotatations(args.source)
    source_annotations_sv, source_negatives = annotations_load_as_sv(source_images_annotations, args.source)
    images_annotations_log(dataset_path=args.source, all_images_annotations=source_images_annotations)
    annotations_log_summary("Source", source_annotations_sv, source_negatives)

    # Destination images : with optional annotation filename
    destination_images_annotations: dict[str, Optional[str]] = load_directory_images_annotatations(args.dest)
    destination_annotations_sv, destination_negatives = annotations_load_as_sv(
        destination_images_annotations, args.dest
    )
    images_annotations_log(dataset_path=args.dest, all_images_annotations=destination_images_annotations)
    annotations_log_summary("Destination", destination_annotations_sv, destination_negatives)

    # Diff : Create
    source_added, source_removed, source_fitting_bboxes = annotations_diff(
        source_annotations=source_annotations_sv, dest_annotations=destination_annotations_sv
    )

    # Logging : Summary
    annotations_log_summary("Source +new", source_added, [])
    annotations_log_summary("Source -removed", source_removed, [])

    # Action : Copy only new annotations
    if args.add_new:
        annotations_append(dataset_path=args.dest, new_annotations=source_added)

    # Action : Remove only old annotations
    if args.remove_old:
        pass


if __name__ == "__main__":
    main()
