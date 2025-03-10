from enum import Enum

import numpy as np
import supervision as sv  # type: ignore
from scipy.stats import linregress


class detections_orientation(Enum):
    """Classify the orientation of a set of points."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    DIAGONAL_RIGHT = "diagonal_right"
    DIAGONAL_LEFT = "diagonal_left"
    SCATTERED = "scattered"
    CLUSTERED = "clustered"


def get_detections_orientation(detections: sv.Detections) -> detections_orientation:
    """Classify the orientation of a set of points."""

    # Points centers : Get from xyxy
    points = detections.get_anchors_coordinates(sv.Position.CENTER)
    if len(points) == 0:
        return detections_orientation.SCATTERED

    x = points[:, 0]
    y = points[:, 1]

    # Check for clustered points (small spread)
    if np.std(x) < 30 and np.std(y) < 30:
        return detections_orientation.CLUSTERED

    # Check for horizontal (small vertical range compared to horizontal range)
    if np.ptp(y) < 0.1 * np.ptp(x):
        return detections_orientation.HORIZONTAL

    # Check for vertical (small horizontal range compared to vertical range)
    if np.ptp(x) < 0.1 * np.ptp(y):
        return detections_orientation.VERTICAL

    # For non-horizontal/vertical sets, use linear regression to decide further
    slope, intercept, r_value, _, _ = linregress(x, y)
    r_squared = r_value**2

    # If the points don't align well, classify as scattered
    if r_squared < 0.3:
        return detections_orientation.SCATTERED

    angle = np.degrees(np.arctan(slope))

    # Diagonal classifications based on the angle of the regression line
    if 25 < angle < 65:
        return detections_orientation.DIAGONAL_RIGHT
    elif -65 < angle < -25:
        return detections_orientation.DIAGONAL_LEFT
    else:
        return detections_orientation.SCATTERED
