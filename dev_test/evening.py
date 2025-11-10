"""Compatibility wrapper exposing the timestamped evening implementation."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

IMPLEMENTATION_FILENAME = "25-11-10-16_evening.py"
_MODULE_NAME = "dev_test.evening_impl_25_11_10_16"


def _load_impl():
    impl_path = Path(__file__).with_name(IMPLEMENTATION_FILENAME)
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, impl_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Failed to create module spec"
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()
Greeting = _impl.Greeting
LOGGER = getattr(_impl, "LOGGER", None)
greet = _impl.greet

__all__ = ["Greeting", "greet", "LOGGER"]


if __name__ == "__main__":
    runpy.run_path(Path(__file__).with_name(IMPLEMENTATION_FILENAME), run_name="__main__")
