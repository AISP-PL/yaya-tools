import numpy as np


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """
    Converts bounding box coordinates from `(xc, yc, width, height)`
    format to `(x_min, y_min, x_max, y_max)` format.

    Args:
        xywh (np.ndarray): A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box in the format `(xc, yc, width, height)`.

    Returns:
        np.ndarray: A numpy array of shape `(N, 4)` where each row corresponds
            to a bounding box in the format `(x_min, y_min, x_max, y_max)`.

    """
    half_width = xywh[:, 2] / 2
    half_height = xywh[:, 3] / 2
    xyxy = xywh.copy()
    xyxy[:, 0] = xywh[:, 0] - half_width
    xyxy[:, 1] = xywh[:, 1] - half_height
    xyxy[:, 2] = xywh[:, 0] + half_width
    xyxy[:, 3] = xywh[:, 1] + half_height
    return xyxy


def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    """
    Converts bounding box coordinates from `(x_min, y_min, x_max, y_max)`
    format to `(xc, yc, width, height)` format.

    Args:
        xyxy (np.ndarray): A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box in the format `(x_min, y_min, x_max, y_max)`.

    Returns:
        np.ndarray: A numpy array of shape `(N, 4)` where each row corresponds
            to a bounding box in the format `(x, y, width, height)`.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        xyxy = np.array([
            [10, 20, 40, 60],
            [15, 25, 50, 70]
        ])

        sv.xyxy_to_xywh(xyxy=xyxy)
        # array([
        #     [10, 20, 30, 40],
        #     [15, 25, 35, 45]
        # ])
        ```
    """
    xywh = xyxy.copy()
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh
