"""Utility script that prints an evening greeting with detailed logging."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
IMPLEMENTATION_NAME = Path(__file__).stem
LOG_FILE = LOG_DIR / f"{IMPLEMENTATION_NAME}.log"


def _configure_logger() -> logging.Logger:
    """Configure a logger that writes verbose details to the logs directory."""

    logger = logging.getLogger("dev_test.evening")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


LOGGER = _configure_logger()


def build_evening_message(name: str | None = None) -> str:
    """Return a friendly evening greeting and emit debug-level breadcrumbs."""

    LOGGER.debug("build_evening_message_start name=%s", name or "anonymous")
    base = "Good evening"
    message = f"{base}, {name}" if name else base
    message = f"{message}! Take a deep breath and enjoy the night."
    LOGGER.debug("build_evening_message_complete message=%s", message)
    return message


def main(name: str | None = None) -> str:
    """Generate and print the evening greeting while logging high-level steps."""

    LOGGER.info("evening_script_start name=%s", name or "anonymous")
    message = build_evening_message(name)
    LOGGER.info("evening_script_complete message=%s", message)
    return message


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, allowing callers to override for testing."""

    parser = argparse.ArgumentParser(
        prog="evening",
        description="Print a detailed evening greeting and emit verbose logs.",
    )
    parser.add_argument(
        "--name",
        metavar="NAME",
        help="Optional recipient name to personalize the greeting.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    print(main(args.name))
