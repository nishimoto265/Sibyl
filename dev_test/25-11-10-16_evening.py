"""CLI utility that prints a Good evening greeting with detailed logging."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / f"{Path(__file__).stem}.log"
LOGGER = logging.getLogger("dev_test.25-11-10-16_evening")


def _configure_logging() -> None:
    if LOGGER.handlers:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)

    LOGGER.setLevel(logging.DEBUG)
    LOGGER.addHandler(handler)
    LOGGER.propagate = False


_configure_logging()


@dataclass(slots=True)
class Greeting:
    """Represents a formatted evening greeting message."""

    recipient: str = "there"

    def render(self) -> str:
        return f"Good evening, {self.recipient}."


def greet(name: str | None = None) -> str:
    """Return a polite evening greeting for the provided name."""

    cleaned = (name or "").strip()
    LOGGER.debug("greet_called name=%r cleaned=%r", name, cleaned)

    if not cleaned:
        message = Greeting().render()
        LOGGER.info("greet_default_used message=%s", message)
        return message

    message = Greeting(recipient=cleaned).render()
    LOGGER.info("greet_prepared message=%s", message)
    return message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print an evening greeting")
    parser.add_argument("name", nargs="?", default="", help="Optional name to greet")
    args = parser.parse_args(argv)

    LOGGER.debug("cli_start raw_args=%s", args)
    output = greet(args.name)
    print(output)
    LOGGER.info("cli_complete output=%s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
