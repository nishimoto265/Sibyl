"""High-level orchestration logic for parallel Codex sessions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


@dataclass(slots=True)
class OrchestrationResult:
    """Return value for a full orchestration cycle."""

    selected_session: str
    sessions_summary: Mapping[str, Any] = field(default_factory=dict)
    artifact: Optional["CycleArtifact"] = None


@dataclass(slots=True)
class CandidateInfo:
    key: str
    label: str
    session_id: Optional[str]
    branch: str
    worktree: Path


@dataclass(slots=True)
class SelectionDecision:
    selected_key: str
    scores: Dict[str, float]
    comments: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CycleLayout:
    """Resolved tmux layout with worker metadata."""

    main_pane: str
    boss_pane: str
    worker_panes: List[str]
    worker_names: List[str]
    pane_to_worker: Dict[str, str]
    pane_to_path: Dict[str, Path]


@dataclass(slots=True)
class CycleArtifact:
    main_session_id: str
    worker_sessions: Dict[str, str]
    boss_session_id: Optional[str]
    worker_paths: Dict[str, Path]
    boss_path: Optional[Path]
    instruction: str
    tmux_session: str


class Orchestrator:
    """Coordinates tmux, git worktrees, Codex monitoring, and Boss evaluation."""

    def __init__(
        self,
        *,
        tmux_manager: Any,
        worktree_manager: Any,
        monitor: Any,
        log_manager: Any,
        worker_count: int,
        session_name: str,
    ) -> None:
        self._tmux = tmux_manager
        self._worktree = worktree_manager
        self._monitor = monitor
        self._log = log_manager
        self._worker_count = worker_count
        self._session_name = session_name

    def run_cycle(
        self,
        instruction: str,
        selector: Optional[Callable[[List[CandidateInfo]], SelectionDecision]] = None,
    ) -> OrchestrationResult:
        """Execute a single orchestrated instruction cycle."""

        worker_roots = self._worktree.prepare()
        boss_path = self._worktree.boss_path

        self._tmux.set_boss_path(boss_path)

        layout, baseline = self._prepare_layout(worker_roots)
        main_session_id, formatted_instruction = self._start_main_session(
            layout=layout,
            instruction=instruction,
            baseline=baseline,
        )

        baseline = self._monitor.snapshot_rollouts()
        fork_map = self._fork_worker_sessions(
            layout=layout,
            main_session_id=main_session_id,
            baseline=baseline,
        )

        completion_info = self._await_worker_completion(fork_map)

        boss_session_id, boss_metrics = self._run_boss_phase(
            layout=layout,
            main_session_id=main_session_id,
            user_instruction=instruction.rstrip(),
            completion_info=completion_info,
        )

        candidates = self._build_candidates(layout, fork_map, boss_session_id, boss_path)

        worker_sessions = {
            layout.pane_to_worker[pane_id]: session_id
            for pane_id, session_id in fork_map.items()
        }
        worker_paths = {
            layout.pane_to_worker[pane_id]: layout.pane_to_path[pane_id]
            for pane_id in layout.worker_panes
            if pane_id in fork_map
        }
        artifact = CycleArtifact(
            main_session_id=main_session_id,
            worker_sessions=worker_sessions,
            boss_session_id=boss_session_id,
            worker_paths=worker_paths,
            boss_path=boss_path if boss_session_id else None,
            instruction=formatted_instruction,
            tmux_session=self._session_name,
        )

        if not candidates:
            scoreboard = {
                "main": {
                    "score": None,
                    "comment": "",
                    "session_id": main_session_id,
                    "branch": None,
                    "worktree": str(self._worktree.root),
                    "selected": True,
                }
            }
            result = OrchestrationResult(
                selected_session=main_session_id,
                sessions_summary=scoreboard,
                artifact=artifact,
            )
            self._log.record_cycle(
                instruction=formatted_instruction,
                layout={
                    "main": layout.main_pane,
                    "boss": layout.boss_pane,
                    "workers": list(layout.worker_panes),
                },
                fork_map=fork_map,
                completion=completion_info,
                result=result,
            )
            return result

        decision, scoreboard = self._auto_or_select(
            candidates,
            completion_info,
            selector,
            boss_metrics,
        )

        selected_info = self._validate_selection(decision, candidates)
        self._finalize_selection(selected=selected_info, main_pane=layout.main_pane)

        result = OrchestrationResult(
            selected_session=selected_info.session_id,
            sessions_summary=scoreboard,
            artifact=artifact,
        )

        self._log.record_cycle(
            instruction=formatted_instruction,
            layout={
                "main": layout.main_pane,
                "boss": layout.boss_pane,
                "workers": list(layout.worker_panes),
            },
            fork_map=fork_map,
            completion=completion_info,
            result=result,
        )

        return result

    # --------------------------------------------------------------------- #
    # Layout preparation
    # --------------------------------------------------------------------- #

    def _prepare_layout(self, worker_roots: Mapping[str, Path]) -> tuple[CycleLayout, Mapping[Path, float]]:
        baseline = self._monitor.snapshot_rollouts()
        layout_map = self._ensure_layout()
        cycle_layout = self._build_cycle_layout(layout_map, worker_roots)
        return cycle_layout, baseline

    def _build_cycle_layout(
        self,
        layout_map: Mapping[str, Any],
        worker_roots: Mapping[str, Path],
    ) -> CycleLayout:
        main_pane = layout_map["main"]
        boss_pane = layout_map["boss"]
        worker_panes = list(layout_map["workers"])

        worker_names = [f"worker-{idx + 1}" for idx in range(len(worker_panes))]
        pane_to_worker = dict(zip(worker_panes, worker_names))
        pane_to_path: Dict[str, Path] = {}

        for pane_id, worker_name in pane_to_worker.items():
            if worker_name not in worker_roots:
                raise RuntimeError(
                    f"Worktree for {worker_name} not prepared; aborting fork sequence."
                )
            pane_to_path[pane_id] = Path(worker_roots[worker_name])

        return CycleLayout(
            main_pane=main_pane,
            boss_pane=boss_pane,
            worker_panes=worker_panes,
            worker_names=worker_names,
            pane_to_worker=pane_to_worker,
            pane_to_path=pane_to_path,
        )

    def _start_main_session(
        self,
        *,
        layout: CycleLayout,
        instruction: str,
        baseline: Mapping[Path, float],
    ) -> tuple[str, str]:
        self._tmux.launch_main_session(pane_id=layout.main_pane)
        main_session_id = self._monitor.register_new_rollout(
            pane_id=layout.main_pane,
            baseline=baseline,
        )

        user_instruction = instruction.rstrip()
        formatted_instruction = self._ensure_done_directive(user_instruction)

        self._tmux.send_instruction_to_pane(
            pane_id=layout.main_pane,
            instruction=formatted_instruction,
        )
        self._tmux.interrupt_pane(pane_id=layout.main_pane)
        self._monitor.capture_instruction(
            pane_id=layout.main_pane,
            instruction=formatted_instruction,
        )
        return main_session_id, formatted_instruction

    # --------------------------------------------------------------------- #
    # Worker handling
    # --------------------------------------------------------------------- #

    def _fork_worker_sessions(
        self,
        *,
        layout: CycleLayout,
        main_session_id: str,
        baseline: Mapping[Path, float],
    ) -> Dict[str, str]:
        worker_paths = {pane_id: layout.pane_to_path[pane_id] for pane_id in layout.worker_panes}
        worker_pane_list = self._tmux.fork_workers(
            workers=layout.worker_panes,
            base_session_id=main_session_id,
            pane_paths=worker_paths,
        )
        self._maybe_pause(
            "PARALLEL_DEV_PAUSE_AFTER_RESUME",
            "[parallel-dev] Debug pause after worker resume. Inspect tmux panes and press Enter to continue...",
        )
        fork_map = self._monitor.register_worker_rollouts(
            worker_panes=worker_pane_list,
            baseline=baseline,
        )
        self._tmux.confirm_workers(fork_map)
        return fork_map

    def _await_worker_completion(self, fork_map: Mapping[str, str]) -> Dict[str, Any]:
        completion_info = self._monitor.await_completion(
            session_ids=list(fork_map.values())
        )
        if os.getenv("PARALLEL_DEV_DEBUG_STATE") == "1":
            print("[parallel-dev] Worker completion status:", completion_info)
        return completion_info

    # --------------------------------------------------------------------- #
    # Boss handling
    # --------------------------------------------------------------------- #

    def _run_boss_phase(
        self,
        *,
        layout: CycleLayout,
        main_session_id: str,
        user_instruction: str,
        completion_info: Dict[str, Any],
    ) -> tuple[Optional[str], Dict[str, Dict[str, Any]]]:
        if not layout.worker_panes:
            return None, {}
        baseline = self._monitor.snapshot_rollouts()
        self._tmux.fork_boss(
            pane_id=layout.boss_pane,
            base_session_id=main_session_id,
            boss_path=self._worktree.boss_path,
        )
        boss_session_id = self._monitor.register_new_rollout(
            pane_id=layout.boss_pane,
            baseline=baseline,
        )
        self._tmux.prepare_for_instruction(pane_id=layout.boss_pane)

        self._maybe_pause(
            "PARALLEL_DEV_PAUSE_BEFORE_BOSS",
            "[parallel-dev] All workers reported completion. Inspect boss pane, then press Enter to send boss instructions...",
        )

        boss_instruction = self._build_boss_instruction(layout.worker_names, user_instruction)
        self._tmux.send_instruction_to_pane(
            pane_id=layout.boss_pane,
            instruction=boss_instruction,
        )

        boss_completion = self._monitor.await_completion(session_ids=[boss_session_id])
        completion_info.update(boss_completion)
        boss_metrics = self._extract_boss_scores(boss_session_id)
        return boss_session_id, boss_metrics

    # --------------------------------------------------------------------- #
    # Candidate selection
    # --------------------------------------------------------------------- #

    def _build_candidates(
        self,
        layout: CycleLayout,
        fork_map: Mapping[str, str],
        boss_session_id: Optional[str],
        boss_path: Path,
    ) -> List[CandidateInfo]:
        candidates: List[CandidateInfo] = []
        for pane_id, session_id in fork_map.items():
            worker_name = layout.pane_to_worker[pane_id]
            branch_name = self._worktree.worker_branch(worker_name)
            worktree_path = layout.pane_to_path[pane_id]
            candidates.append(
                CandidateInfo(
                    key=worker_name,
                    label=f"{worker_name} (session {session_id})",
                    session_id=session_id,
                    branch=branch_name,
                    worktree=worktree_path,
                )
            )

        if boss_session_id:
            candidates.append(
                CandidateInfo(
                    key="boss",
                    label=f"boss (session {boss_session_id})",
                    session_id=boss_session_id,
                    branch=self._worktree.boss_branch,
                    worktree=boss_path,
                )
            )
        return candidates

    def _validate_selection(
        self,
        decision: SelectionDecision,
        candidates: Iterable[CandidateInfo],
    ) -> CandidateInfo:
        candidate_map = {candidate.key: candidate for candidate in candidates}
        if decision.selected_key not in candidate_map:
            raise ValueError(
                f"Selector returned unknown candidate '{decision.selected_key}'. "
                f"Known candidates: {sorted(candidate_map)}"
            )
        selected_info = candidate_map[decision.selected_key]
        if selected_info.session_id is None:
            raise RuntimeError("Selected candidate has no session id; cannot resume main session.")
        return selected_info

    def _finalize_selection(
        self,
        *,
        selected: CandidateInfo,
        main_pane: str,
    ) -> None:
        if selected.branch:
            self._worktree.merge_into_main(selected.branch)
        if selected.session_id:
            self._tmux.promote_to_main(session_id=selected.session_id, pane_id=main_pane)

    # --------------------------------------------------------------------- #
    # Existing helper utilities
    # --------------------------------------------------------------------- #

    def _ensure_layout(self) -> MutableMapping[str, Any]:
        layout = self._tmux.ensure_layout(
            session_name=self._session_name,
            worker_count=self._worker_count,
        )
        self._validate_layout(layout)
        return layout

    def _validate_layout(self, layout: Mapping[str, Any]) -> None:
        if "main" not in layout or "boss" not in layout or "workers" not in layout:
            raise ValueError(
                "tmux_manager.ensure_layout must return mapping with "
                "'main', 'boss', and 'workers' keys"
            )
        workers = layout["workers"]
        if not isinstance(workers, Sequence):
            raise ValueError("layout['workers'] must be a sequence")
        if len(workers) != self._worker_count:
            raise ValueError(
                "tmux_manager.ensure_layout returned "
                f"{len(workers)} workers but {self._worker_count} expected"
            )

    def _ensure_done_directive(self, instruction: str) -> str:
        marker = "/done"
        if marker in instruction:
            return instruction
        directive = (
            "\n\nWhen you have completed the requested work, respond with exactly `/done`."
        )
        return instruction.rstrip() + directive

    def _auto_or_select(
        self,
        candidates: List[CandidateInfo],
        completion_info: Mapping[str, Any],
        selector: Optional[Callable[[List[CandidateInfo]], SelectionDecision]],
        metrics: Optional[Mapping[str, Mapping[str, Any]]],
    ) -> tuple[SelectionDecision, Dict[str, Dict[str, Any]]]:
        base_scoreboard = self._build_scoreboard(candidates, completion_info, metrics)
        if selector is None:
            raise RuntimeError(
                "Selection requires a selector; automatic boss scoring is not available."
            )

        try:
            decision = selector(candidates, base_scoreboard)
        except TypeError:
            decision = selector(candidates)

        scoreboard = self._apply_selection(base_scoreboard, decision)
        return decision, scoreboard

    def _build_scoreboard(
        self,
        candidates: List[CandidateInfo],
        completion_info: Mapping[str, Any],
        metrics: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        scoreboard: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            entry: Dict[str, Any] = {
                "score": None,
                "comment": "",
                "session_id": candidate.session_id,
                "branch": candidate.branch,
                "worktree": str(candidate.worktree),
            }
            if candidate.session_id and candidate.session_id in completion_info:
                entry.update(completion_info[candidate.session_id])
            if metrics and candidate.key in metrics:
                metric_entry = metrics[candidate.key]
                if "score" in metric_entry:
                    try:
                        entry["score"] = float(metric_entry["score"])
                    except (TypeError, ValueError):
                        entry["score"] = metric_entry.get("score")
                comment_text = metric_entry.get("comment")
                if comment_text:
                    entry["comment"] = comment_text
            scoreboard[candidate.key] = entry
        return scoreboard

    def _apply_selection(
        self,
        scoreboard: Dict[str, Dict[str, Any]],
        decision: SelectionDecision,
    ) -> Dict[str, Dict[str, Any]]:
        for key, comment in decision.comments.items():
            entry = scoreboard.setdefault(key, {})
            if comment:
                entry["comment"] = comment
        for key in scoreboard:
            entry = scoreboard[key]
            entry["selected"] = key == decision.selected_key
        return scoreboard

    def _extract_boss_scores(self, boss_session_id: str) -> Dict[str, Dict[str, Any]]:
        raw = self._monitor.get_last_assistant_message(boss_session_id)
        if not raw:
            return {}

        def _parse_json_from(raw_text: str) -> Optional[Dict[str, Any]]:
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            for line in lines:
                if line == "/done":
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            return None

        data = _parse_json_from(raw)
        if data is None:
            return {
                "boss": {
                    "score": None,
                    "comment": f"Failed to parse boss output as JSON: {raw[:80]}...",
                }
            }

        scores = data.get("scores")
        if not isinstance(scores, dict):
            return {}

        metrics: Dict[str, Dict[str, Any]] = {}
        for key, value in scores.items():
            if not isinstance(value, dict):
                continue
            metrics[key] = {
                "score": value.get("score"),
                "comment": value.get("comment", ""),
            }
        return metrics

    def _build_boss_instruction(
        self,
        worker_names: Sequence[str],
        user_instruction: str,
    ) -> str:
        worker_lines = "\n".join(
            f"- Evaluate the proposal from {name}" for name in worker_names
        )
        return (
            "Boss evaluation phase:\n"
            "You are the reviewer. The original user instruction was:\n"
            f"""{user_instruction}\n\n"""
            "Tasks:\n"
            f"{worker_lines}\n\n"
            "For each candidate, assign a numeric score between 0 and 100 and provide a short comment.\n"
            "Respond with JSON using the schema:\n"
            "{\n  \"scores\": {\n    \"worker-1\": {\"score\": <number>, \"comment\": <string>},\n"
            "    ... other candidates ...\n  }\n}\n"
            "Only output the JSON object. After that, send /done."
        )

    def _maybe_pause(self, env_var: str, message: str) -> None:
        if os.getenv(env_var) == "1":
            input(message)
