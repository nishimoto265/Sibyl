"""Compatibility wrapper for the timestamped evening implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


IMPLEMENTATION_FILENAME = "25-11-10-16_evening.py"
_MODULE_NAME = f"dev_test.{IMPLEMENTATION_FILENAME.removesuffix('.py')}"


def _implementation_path() -> Path:
    path = Path(__file__).with_name(IMPLEMENTATION_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"Implementation file '{IMPLEMENTATION_FILENAME}' is missing at {path}."
        )
    return path


def _load_implementation() -> ModuleType:
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _implementation_path())
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {_MODULE_NAME}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


def main(argv: list[str] | None = None) -> str:
    impl = _load_implementation()
    args = impl._parse_args(argv)  # type: ignore[attr-defined]
    return impl.main(args.name)  # type: ignore[attr-defined]


if __name__ == "__main__":
    print(main())
