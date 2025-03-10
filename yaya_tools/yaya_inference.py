import argparse
import logging
from typing import Optional

import supervision as sv
from tqdm import tqdm

from yaya_tools import __version__
from yaya_tools.detection.detector import Detector
from yaya_tools.detection.detector_yolov4_cvdnn import DetectorCVDNN
from yaya_tools.detection.detector_yolov4_darknet import DetectorDarknet

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
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output video file")
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the weights file")
    parser.add_argument("--names_path", type=str, required=True, help="Path to the names file")
    parser.add_argument("--confidence", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.30, help="NMS threshold")
    parser.add_argument(
        "--tracking", action="store_true", required=False, help="If set, the detector will use tracking with ByteSORT."
    )
    parser.add_argument(
        "--gpu", action="store_true", required=False, help="If set, the detector will use the GPU for inference."
    )

    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("-v", action="version", version=__version__, help="Show version and exit.")
    args = parser.parse_args()

    # Detector : Create and initialize the detector
    if args.gpu:
        detector: Detector = DetectorDarknet(
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
    else:
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

    # Annotators : Create Box and Label
    color_lookup = sv.ColorLookup.CLASS
    if args.tracking:
        color_lookup = sv.ColorLookup.TRACK
    box_annotator = sv.BoxAnnotator(color_lookup=color_lookup)
    label_annotator = sv.LabelAnnotator(text_padding=5, color_lookup=color_lookup)

    # Tracker : sv.ByteTrack
    tracker = sv.ByteTrack()

    source_video_info = sv.VideoInfo.from_video_path(video_path=args.video)
    with sv.VideoSink(target_path=args.output, video_info=source_video_info) as sink:
        for frame in tqdm(
            sv.get_video_frames_generator(source_path=args.video), desc="Processing video", unit="frames"
        ):
            # Detection
            detections = detector.detect(frame_number=0, frame=frame)

            # Tracking
            if args.tracking:
                detections = tracker.update_with_detections(detections)

            # Annotate : Labels
            labels: Optional[list[str]] = None
            if detections.tracker_id is not None:
                labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

            # Annotate
            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels)

            # Write frame
            sink.write_frame(annotated)


if __name__ == "__main__":
    main()
