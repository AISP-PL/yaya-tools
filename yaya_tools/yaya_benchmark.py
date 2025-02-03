import argparse
import logging
from typing import Optional

from yaya_tools import __version__
from yaya_tools.detection.detection_test import log_map, test_detector
from yaya_tools.detection.detector_yolov4_cvdnn import DetectorCVDNN
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
    # Argument parser
    parser = argparse.ArgumentParser(add_help=False, description="YAYa dataset management tool")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the weights file")
    parser.add_argument("--names_path", type=str, required=True, help="Path to the names file")
    parser.add_argument("--confidence", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.30, help="NMS threshold")

    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("-v", action="version", version=__version__, help="Show version and exit.")
    args = parser.parse_args()

    detector = DetectorCVDNN(
        config={
            "cfg_path": args.cfg_path,
            "weights_path": args.weights_path,
            "data_path": "",
            "names_path": args.names_path,
            "confidence": args.confidence,
            "nms_threshold": args.nms_threshold,
            "force_cpu": True,
        }
    )

    # Benchmark detector on dataset
    mAP = test_detector(args.dataset, detector)

    # Logging : Summary
    log_map(mAP)


if __name__ == "__main__":
    logging_terminal_setup()
    main()
