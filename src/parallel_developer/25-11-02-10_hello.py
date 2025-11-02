"""Script that prints HELLO to stdout with verbose logging for diagnostics."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


LOG_FILENAME = "25-11-02-10_hello.log"
LOGGER_NAME = "parallel_developer.hello_script.25_11_02_10"


def _configure_logger() -> logging.Logger:
    """Configure a file logger that records each execution step."""

    logs_dir = Path(__file__).resolve().parents[2] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        log_path = logs_dir / LOG_FILENAME
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def main() -> int:
    logger = _configure_logger()
    logger.debug("hello_script_start")

    message = "HELLO"
    logger.debug("hello_script_message_prepared message=%s", message)
    print(message)

    logger.debug("hello_script_complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
