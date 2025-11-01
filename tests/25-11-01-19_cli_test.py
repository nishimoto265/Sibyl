from unittest.mock import Mock

from typer.testing import CliRunner

from parallel_developer.cli import app


def test_cli_prompt_invokes_orchestrator(monkeypatch):
    runner = CliRunner()

    orchestrator_mock = Mock(name="orchestrator")
    orchestrator_mock.run_cycle.return_value = Mock(
        selected_session="session-worker-2",
        sessions_summary={"worker-1": {"score": 70.0}, "boss": {"score": 90.0}},
    )

    selector_mock = Mock(name="selector")
    selector_mock.return_value = Mock(
        selected_key="boss",
        scores={"worker-1": 70.0, "boss": 90.0},
        comments={},
    )

    monkeypatch.setattr(
        "parallel_developer.cli.build_orchestrator",
        lambda worker_count, log_dir: orchestrator_mock,
    )
    monkeypatch.setattr(
        "parallel_developer.cli.build_interactive_selector",
        lambda: selector_mock,
    )

    result = runner.invoke(
        app,
        ["--workers", "4", "--instruction", "Ship-it"],
    )

    assert result.exit_code == 0
    orchestrator_mock.run_cycle.assert_called_once()
    args, kwargs = orchestrator_mock.run_cycle.call_args
    assert args[0] == "Ship-it"
    assert kwargs["selector"] is selector_mock
