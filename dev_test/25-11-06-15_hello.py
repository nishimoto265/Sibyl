"""Hello script with detailed logging for troubleshooting."""
from __future__ import annotations

import logging
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "25-11-06-15_hello.log"


def configure_logger() -> logging.Logger:
    """Configure a module-level logger with file output."""
    logger = logging.getLogger("dev_test.hello")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(LOG_PATH)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.debug("logger_configured")
    return logger


def run() -> str:
    """Return the greeting message while logging intermediate states."""
    logger = configure_logger()
    logger.info("hello_script_start")
    message = "Hello, world!"
    logger.debug("message_prepared: %s", message)
    logger.info("hello_script_complete")
    return message


def main() -> None:
    """CLI entry point for the hello script."""
    logger = configure_logger()
    logger.debug("main_invoked")
    message = run()
    print(message)
    logger.debug("main_completed")


if __name__ == "__main__":
    main()
