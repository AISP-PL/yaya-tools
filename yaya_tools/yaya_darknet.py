#!/usr/bin/env python3
import argparse
import logging
import sys

from yaya_tools.helpers.darknet_parsing import parse_darknet_log


def main() -> None:
    """
    Main function for reading darknet logs and displaying them
    """
    parser = argparse.ArgumentParser(description="Reading darknet log and receiving results")
    parser.add_argument("--log", type=str, default="", required=True, help="Path to log file")
    parser.add_argument("--return_iou", action="store_true", help="Return average IoU value")
    parser.add_argument("--return_map", action="store_true", help="Return mAP value  printing it")
    parser.add_argument("--return_map_float", action="store_true", help="Return mAP value printing it")

    args = parser.parse_args()

    log = parse_darknet_log(args.log)
    if log is None:
        logging.error("Failed to parse the log file.")
        sys.exit(1)

    # mAP : Return this values as text
    if args.return_map:
        print(f"mAP: {log.mAP_percent:2.2f}")

    # mAP : Return this values as float
    if args.return_map_float:
        print(f"{log.mAP_raw:.6f}")

    # Average IoU : Return this values as text
    if args.return_iou:
        print(f"Average IoU: {log.average_iou:.2f}")


if __name__ == "__main__":
    main()
