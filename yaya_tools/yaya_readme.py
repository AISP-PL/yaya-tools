#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def logging_terminal_setup() -> None:
    """
    Setup logging for the application. Logs will be sent to the terminal
    at DEBUG level with a simple timestamped format.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    # Avoid adding multiple handlers if this is called twice
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.addHandler(console)

    logging.info("\n\n###### Logging start of terminal session ######\n")


def update_key_in_file(file_path: Path, key: str, value: str, encoding: str = "utf-8") -> None:
    """
    Open the given text file, look for any line containing `key`.
    If found: replace that entire line with "key: value" (only once).
    If not found: log an error and append a new line "key: value" at EOF.

    Parameters
    ----------
    file_path : Path
        Path to the text file (e.g. readme.kmd).
    key : str
        The field name / key to search for.
    value : str
        The new value to set for that key.
    encoding : str, optional
        File encoding to use when reading/writing (default: "utf-8").
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path.resolve()}")
        raise FileNotFoundError(f"Cannot update: '{file_path}' does not exist.")

    # Read all lines
    try:
        with file_path.open("r", encoding=encoding) as f:
            lines: List[str] = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read '{file_path}': {e}")
        raise

    # Find : Single line containing the key
    found_index = -1
    for index, line in enumerate(lines):
        if key in line:
            found_index = index
            logger.info(f"Found key '{key}' in line [{found_index}]'{line.strip()}'.")
            break

    # Update : Only single line
    new_line_text = f"{key}: {value}\n"
    if found_index != -1:
        lines[found_index] = new_line_text
        logger.info(f"Updated line [{found_index}] to '{new_line_text.strip()}'.")
    # Append : Key not found, add at EOF
    else:
        lines.append(new_line_text)
        logger.warning(f"Key '{key}' not found. Appending new line: '{new_line_text.strip()}'.")

    # File : Write updated lines back to the file
    try:
        with file_path.open("w", encoding=encoding) as f:
            f.writelines(lines)
    except Exception as e:
        logger.error(f"Failed to write updates to '{file_path}': {e}")
        raise


def main() -> None:
    logging_terminal_setup()

    parser = argparse.ArgumentParser(
        prog="yaya-readme",
        description="yaya-readme: Command‐line tool to update or append a key/value in a text file (e.g. readme.kmd).",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument(
        "-v", "--version", action="version", version="yaya-readme 1.0.0", help="Show program's version number and exit."
    )
    parser.add_argument(
        "--update",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Update the given KEY with VALUE. If KEY exists, replaces its entire line. Otherwise, appends at the end.",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="README.md",
        help="Path to the file to update (default: README.md).",
    )
    args = parser.parse_args()

    if args.update:
        key, value = args.update
        file_to_edit = Path(args.file)
        try:
            update_key_in_file(file_to_edit, key, value)
        except Exception as e:
            logger.error(f"Aborting due to error: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
