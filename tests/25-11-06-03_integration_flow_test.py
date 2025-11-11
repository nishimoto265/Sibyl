import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock

import pytest

from parallel_developer.controller import CLIController
from parallel_developer.orchestrator import CandidateInfo, CycleArtifact, CycleLayout, OrchestrationResult
from parallel_developer.stores import ManifestStore


def _run(coro):
    return asyncio.run(coro)


class _IntegrationOrchestrator:
    def __init__(
        self,
        *,
        log_dir: Path,
        session_name: str,
        worker_count: int,
        boss_mode,
        **_,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._session_name = session_name
        self._main_session_id = "session-main-001"
        self._worker_session_id = "session-worker-001"
        self._boss_session_id = "session-boss-001"
        self._main_hook = None
        self._worker_decider = None
        self._tmux = Mock()
        self._worker_path = self._log_dir / "worker-1"
        self._worker_path.mkdir(parents=True, exist_ok=True)
        self._boss_path = self._log_dir / "boss"
        self._boss_path.mkdir(parents=True, exist_ok=True)
        self._artifact_log = self._log_dir / "instruction.jsonl"
        self._artifact_log.write_text('{"type": "instruction"}\n', encoding="utf-8")
        self._session_map = {
            "main": self._main_session_id,
            "worker-1": self._worker_session_id,
            "boss": self._boss_session_id,
        }
        self._scoreboard: Dict[str, Dict[str, object]] = {
            "main": {
                "score": 70.0,
                "comment": "baseline",
                "session_id": self._main_session_id,
            },
            "worker-1": {
                "score": 95.0,
                "comment": "best answer",
                "session_id": self._worker_session_id,
            },
        }
        self._candidates: List[CandidateInfo] = [
            CandidateInfo(
                key="main",
                label="Main Session",
                session_id=self._main_session_id,
                branch="main",
                worktree=self._log_dir,
            ),
            CandidateInfo(
                key="worker-1",
                label="Worker 1",
                session_id=self._worker_session_id,
                branch="feature/worker-1",
                worktree=self._worker_path,
            ),
        ]

    def set_main_session_hook(self, hook) -> None:
        self._main_hook = hook

    def set_worker_decider(self, decider) -> None:
        self._worker_decider = decider

    def run_cycle(self, instruction: str, selector, resume_session_id: Optional[str] = None) -> OrchestrationResult:
        if self._main_hook:
            self._main_hook(self._main_session_id)

        if self._worker_decider:
            layout = CycleLayout(
                main_pane="main",
                boss_pane="boss",
                worker_panes=["pane-worker-1"],
                worker_names=["worker-1"],
                pane_to_worker={"pane-worker-1": "worker-1"},
                pane_to_path={"pane-worker-1": self._worker_path},
            )
            self._worker_decider({"pane-worker-1": self._worker_session_id}, {}, layout)

        decision = selector(self._candidates, scoreboard=self._scoreboard)
        selected_session = self._session_map[decision.selected_key]

        artifact = CycleArtifact(
            main_session_id=self._main_session_id,
            worker_sessions={"worker-1": self._worker_session_id},
            boss_session_id=self._boss_session_id,
            worker_paths={"worker-1": self._worker_path},
            boss_path=self._boss_path,
            instruction=instruction,
            tmux_session=self._session_name,
        )
        artifact.log_paths["jsonl"] = self._artifact_log
        artifact.selected_session_id = selected_session

        return OrchestrationResult(
            selected_session=selected_session,
            sessions_summary=self._scoreboard,
            artifact=artifact,
        )


@pytest.mark.asyncio
async def test_full_auto_flow_runs_to_completion(tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    manifest_store = ManifestStore(tmp_path / "manifests")
    logs_root = tmp_path / "logs"

    controller = CLIController(
        event_handler=handler,
        orchestrator_builder=lambda **kwargs: _IntegrationOrchestrator(**kwargs),
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )
    controller._config.logs_root = logs_root

    await controller.handle_input("/flow full_auto")
    await controller.handle_input("Implement integration scenario")

    assert controller._last_selected_session == "session-worker-001"
    assert controller._last_scoreboard["worker-1"]["score"] == 95.0
    auto_logs = [payload["text"] for event, payload in events if event == "log" and "[flow full_auto]" in payload.get("text", "")]
    assert auto_logs, "自動選択のログが出力されていません。"

    scoreboard_events = [payload for event, payload in events if event == "scoreboard"]
    assert scoreboard_events and scoreboard_events[-1]["scoreboard"]["worker-1"]["score"] == 95.0

    sessions = manifest_store.list_sessions()
    assert len(sessions) == 1
    manifest = manifest_store.load_manifest(sessions[0].session_id)
    assert manifest.selected_session_id == "session-worker-001"
    assert Path(manifest.conversation_log).exists()
