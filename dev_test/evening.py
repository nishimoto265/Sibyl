#!/usr/bin/env python3
"""Compatibility wrapper that delegates to the timestamped evening script."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Sequence

IMPLEMENTATION_FILENAME = "25-11-10-12_evening.py"
IMPLEMENTATION_MODULE = "dev_test.evening_impl"


def _load_implementation() -> ModuleType:
    """Load the actual implementation module that contains the logic."""

    impl_path = Path(__file__).with_name(IMPLEMENTATION_FILENAME)
    if not impl_path.exists():  # pragma: no cover - ensures explicit failure when missing
        raise FileNotFoundError(f"Missing implementation script: {impl_path}")

    spec = spec_from_file_location(IMPLEMENTATION_MODULE, impl_path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib sanity check
        raise ImportError(f"Could not load spec for {impl_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> str:
    """Delegate CLI handling to the implementation module."""

    module = _load_implementation()
    if not hasattr(module, "main"):
        raise AttributeError("Implementation module must expose a main() function")
    return module.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
