import numpy as np

from yaya_tools.helpers.np_boxes import xywh_to_xyxy, xyxy_to_xywh


def test_xywh_to_xyxy() -> None:
    """Example data: [[xc, yc, width, height]]"""
    xywh = np.array([[50, 50, 20, 10], [100, 100, 40, 20]])
    expected = np.array([[40, 45, 60, 55], [80, 90, 120, 110]])
    result = xywh_to_xyxy(xywh)
    np.testing.assert_allclose(result, expected)


def test_xyxy_to_xywh() -> None:
    """Example data: [[x_min, y_min, x_max, y_max]]"""
    xyxy = np.array([[40, 45, 60, 55], [80, 90, 120, 110]])
    expected = np.array([[50, 50, 20, 10], [100, 100, 40, 20]])
    result = xyxy_to_xywh(xyxy)
    np.testing.assert_allclose(result, expected)
