import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture(autouse=True)
def _isolate_parallel_dev_config(tmp_path, monkeypatch):
    config_dir = tmp_path / ".parallel-dev"
    config_path = config_dir / "config.yaml"
    monkeypatch.setenv("PARALLEL_DEV_CONFIG_PATH", str(config_path))
    yield
