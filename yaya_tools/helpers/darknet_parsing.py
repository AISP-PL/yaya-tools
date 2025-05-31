import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ClassMetrics:
    """
    Dataclass representing per-class metrics extracted from a Darknet log.
    """

    class_id: int
    name: str
    ap: float  # Average Precision as a percentage (e.g., 98.44)
    TP: int  # True Positives
    FP: int  # False Positives


@dataclass
class DarknetLogResult:
    """
    Dataclass representing the overall parsed result of a Darknet log.
    """

    classes: List[ClassMetrics]
    average_iou: Optional[float]  # e.g., 79.60
    mAP_raw: Optional[float]  # e.g., 0.990564
    mAP_percent: Optional[float]  # e.g., 99.06


def parse_darknet_log(file_path: str) -> DarknetLogResult:
    """
    Reads a Darknet log file from `file_path`, parses per-class metrics,
    and also extracts the total mean Average Precision (mAP@0.50) and
    the average IoU. Returns a DarknetLogResult.

    Expected relevant log lines (examples):
      class_id = 0, name = a1.rowery, ap = 98.65%      (TP = 824, FP = 79)
      ...
      for conf_thresh = 0.25, TP = 23413, FP = 2111, FN = 499, average IoU = 79.60 %
      ...
      mean average precision (mAP@0.50) = 0.990564, or 99.06 %
    """
    # Regex to match per-class metrics:
    class_pattern = (
        r"class_id\s*=\s*(\d+),\s*"
        r"name\s*=\s*([\w\.\-]+),\s*"
        r"ap\s*=\s*([\d\.]+)%\s*"
        r"\(TP\s*=\s*(\d+),\s*FP\s*=\s*(\d+)\)"
    )
    # Regex to match average IoU line:
    iou_pattern = r"average IoU\s*=\s*([\d\.]+)\s*%"

    # Regex to match mAP line:
    map_pattern = r"mean average precision\s*\(mAP@0\.50\)\s*=\s*([\d\.]+),\s*or\s*([\d\.]+)\s*%"

    class_list: List[ClassMetrics] = []
    avg_iou: Optional[float] = None
    map_raw: Optional[float] = None
    map_pct: Optional[float] = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # Attempt to parse a class-specific line
            m_cls = re.search(class_pattern, line)
            if m_cls:
                class_id = int(m_cls.group(1))
                name = m_cls.group(2)
                ap = float(m_cls.group(3))
                tp = int(m_cls.group(4))
                fp = int(m_cls.group(5))

                class_list.append(ClassMetrics(class_id=class_id, name=name, ap=ap, TP=tp, FP=fp))
                continue  # proceed to next line

            # Attempt to parse the average IoU line
            m_iou = re.search(iou_pattern, line)
            if m_iou:
                avg_iou = float(m_iou.group(1))
                continue

            # Attempt to parse the mAP line
            m_map = re.search(map_pattern, line)
            if m_map:
                map_raw = float(m_map.group(1))
                map_pct = float(m_map.group(2))
                continue

    # Sort classes by class_id
    class_list.sort(key=lambda cm: cm.class_id)

    return DarknetLogResult(classes=class_list, average_iou=avg_iou, mAP_raw=map_raw, mAP_percent=map_pct)
