from yaya_tools.helpers.darknet_parsing import (
    DarknetLogResult,
    parse_darknet_log,
)


def test_darknet_log() -> None:
    """
    Test parsing a Darknet log file and verify the results.
    """
    log = parse_darknet_log("tests/test_darknet/darknet_map.log")
    assert isinstance(log, DarknetLogResult)
    assert log.average_iou == 79.60
    assert log.mAP_raw == 0.990564
    assert log.mAP_percent == 99.06
