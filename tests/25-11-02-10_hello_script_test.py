import subprocess
import sys
from pathlib import Path


def test_hello_script_prints_and_logs():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "src" / "parallel_developer" / "25-11-02-10_hello.py"
    log_path = project_root / "logs" / "25-11-02-10_hello.log"

    if log_path.exists():
        log_path.unlink()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert result.stdout == "HELLO\n"

    assert log_path.exists(), "Expected log file was not created."
    log_content = log_path.read_text(encoding="utf-8")
    assert "hello_script_start" in log_content
    assert "hello_script_complete" in log_content
