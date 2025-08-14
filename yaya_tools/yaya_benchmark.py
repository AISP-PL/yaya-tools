import argparse
import logging

from yaya_tools import __version__
from yaya_tools.detection.detection_test import dataset_benchmark, log_map
from yaya_tools.detection.detector_yolov4_cvdnn import DetectorCVDNN
from yaya_tools.helpers.terminal_logging import logging_terminal_setup

logger = logging.getLogger(__name__)


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
    detector.init()

    # Benchmark detector on dataset
    mAP, negatives_ap = dataset_benchmark(args.dataset, detector)

    # Logging : Summary
    log_map(mAP, negatives_ap)


if __name__ == "__main__":
    main()
