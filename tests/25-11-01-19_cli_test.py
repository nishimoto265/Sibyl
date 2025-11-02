from unittest.mock import Mock

from typer.testing import CliRunner

from parallel_developer.cli import app


def test_cli_prompt_invokes_orchestrator(monkeypatch):
    runner = CliRunner()

    orchestrator_mock = Mock(name="orchestrator")
    orchestrator_mock.run_cycle.return_value = Mock(
        selected_session="session-worker-2",
        sessions_summary={
            "worker-1": {"score": 75.0, "comment": "solid", "selected": False},
            "boss": {"score": 92.0, "comment": "merged", "selected": True},
        },
    )

    monkeypatch.setattr(
        "parallel_developer.cli.build_orchestrator",
        lambda worker_count, log_dir: orchestrator_mock,
    )

    result = runner.invoke(
        app,
        ["--workers", "4", "--instruction", "Ship-it"],
        input="1\n",
    )

    assert result.exit_code == 0
    orchestrator_mock.run_cycle.assert_called_once()
    args, kwargs = orchestrator_mock.run_cycle.call_args
    assert args[0] == "Ship-it"
    assert "selector" in kwargs
    stdout = result.stdout
    assert "Scoreboard" in stdout
    assert "boss" in stdout and "[selected]" in stdout
