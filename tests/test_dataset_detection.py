from yaya_tools.detection.detection_test import test_detector
from yaya_tools.detection.detector_yolov4_cvdnn import DetectorCVDNN


def test_detector_with_test_dataset() -> None:
    """Test the detector using data from tests/test_dataset/."""
    dataset_path = "tests/test_dataset/"
    detector = DetectorCVDNN(
        config={
            "cfg_path": "tests/test_model/yolov4-tiny.cfg",
            "weights_path": "tests/test_model/yolov4-tiny.weights",
            "data_path": "tests/test_model/coco.data",
            "names_path": "tests/test_model/coco.names",
            "confidence": 0.50,
            "nms_threshold": 0.30,
            "force_cpu": True,
        }
    )
    detector.init()
    confusion_matrix = test_detector(dataset_path, detector)
    print(confusion_matrix)
