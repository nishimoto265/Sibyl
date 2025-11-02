"""Script that prints HELLO to stdout with verbose logging for debugging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


LOG_FILENAME = "25-11-02-09_hello.log"


def _configure_logger() -> logging.Logger:
    """Configure a file logger that records script execution steps."""

    logs_dir = Path(__file__).resolve().parents[2] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("parallel_developer.hello_script")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        log_path = logs_dir / LOG_FILENAME
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def main() -> int:
    logger = _configure_logger()
    logger.debug("Starting HELLO script run")

    message = "HELLO"
    print(message)

    logger.debug("Completed HELLO script run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
