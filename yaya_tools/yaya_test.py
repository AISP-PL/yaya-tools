import logging

from yaya_tools.helpers.terminal_logging import logging_terminal_setup

logger = logging.getLogger(__name__)


def main() -> None:
    """Test function for package installation tests"""
    logging_terminal_setup()
    print("yaya_tools package installed successfully!")


if __name__ == "__main__":
    main()
