import logging
import os

logger = logging.getLogger(__name__)


def logging_terminal_setup() -> None:
    """Setup kolorowego logowania na terminal (ANSI). Ustaw YAYA_COLOR=0 aby wyłączyć kolory."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    class ColorFormatter(logging.Formatter):
        RESET = "\033[0m"
        COLORS = {
            logging.DEBUG: "\033[36m",  # cyan
            logging.INFO: "",  # default no color
            logging.WARNING: "\033[33m",  # yellow
            logging.ERROR: "\033[31m",  # red
            logging.CRITICAL: "\033[1;41m",  # bold + red background
        }

        def __init__(self, *args, use_color: bool = True, **kwargs):
            super().__init__(*args, **kwargs)
            self.use_color = use_color

        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            msg = super().format(record)
            if self.use_color:
                color = self.COLORS.get(record.levelno)
                if color:
                    return f"{color}{msg}{self.RESET}"
            return msg

    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()

    force_disable = os.getenv("YAYA_COLOR", "1") in ("0", "false", "False")
    use_color = getattr(console.stream, "isatty", lambda: False)() and not force_disable

    formatter = ColorFormatter("%(asctime)s %(levelname)s: %(message)s", use_color=use_color)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    root.addHandler(console)
    logging.info("\n\n###### Logging start of terminal session ######\n")
