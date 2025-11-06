"""High-level orchestration logic for parallel Codex sessions."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


class BossMode(str, Enum):
    SKIP = "skip"
    SCORE = "score"
    REWRITE = "rewrite"


@dataclass(slots=True)
class OrchestrationResult:
    """Return value for a full orchestration cycle."""

    selected_session: str
    sessions_summary: Mapping[str, Any] = field(default_factory=dict)
    artifact: Optional["CycleArtifact"] = None
    continue_requested: bool = False


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
    log_paths: Dict[str, Path] = field(default_factory=dict)
    selected_session_id: Optional[str] = None


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
        main_session_hook: Optional[Callable[[str], None]] = None,
        worker_decider: Optional[Callable[[Mapping[str, str], Mapping[str, Any], "CycleLayout"], bool]] = None,
        boss_mode: BossMode = BossMode.SCORE,
    ) -> None:
        self._tmux = tmux_manager
        self._worktree = worktree_manager
        self._monitor = monitor
        self._log = log_manager
        self._worker_count = worker_count
        self._session_name = session_name
        self._boss_mode = boss_mode if isinstance(boss_mode, BossMode) else BossMode(str(boss_mode))
        self._active_worker_sessions: List[str] = []
        self._main_session_hook: Optional[Callable[[str], None]] = main_session_hook
        self._worker_decider = worker_decider

    def set_main_session_hook(self, hook: Optional[Callable[[str], None]]) -> None:
        self._main_session_hook = hook

    def set_worker_decider(
        self,
        decider: Optional[Callable[[Mapping[str, str], Mapping[str, Any], CycleLayout], bool]],
    ) -> None:
        self._worker_decider = decider

    def run_cycle(
        self,
        instruction: str,
        selector: Optional[Callable[[List[CandidateInfo]], SelectionDecision]] = None,
        resume_session_id: Optional[str] = None,
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
            resume_session_id=resume_session_id,
        )

        baseline = self._monitor.snapshot_rollouts()
        fork_map = self._fork_worker_sessions(
            layout=layout,
            main_session_id=main_session_id,
            baseline=baseline,
        )
        self._dispatch_worker_instructions(
            layout=layout,
            user_instruction=instruction,
        )
        self._active_worker_sessions = [session_id for session_id in fork_map.values() if session_id]
        completion_info = self._await_worker_completion(fork_map)

        worker_sessions = {
            layout.pane_to_worker[pane_id]: session_id
            for pane_id, session_id in fork_map.items()
        }
        worker_paths = {
            layout.pane_to_worker[pane_id]: layout.pane_to_path[pane_id]
            for pane_id in layout.worker_panes
            if pane_id in fork_map
        }

        if self._worker_decider:
            try:
                continue_requested = self._worker_decider(fork_map, completion_info, layout)
            except Exception:
                continue_requested = False
        else:
            continue_requested = False

        artifact = CycleArtifact(
            main_session_id=main_session_id,
            worker_sessions=worker_sessions,
            boss_session_id=None,
            worker_paths=worker_paths,
            boss_path=boss_path,
            instruction=formatted_instruction,
            tmux_session=self._session_name,
        )

        if continue_requested:
            artifact.selected_session_id = main_session_id
            self._active_worker_sessions = []
            return OrchestrationResult(
                selected_session=main_session_id,
                sessions_summary={},
                artifact=artifact,
                continue_requested=True,
            )

        if self._boss_mode == BossMode.SKIP:
            boss_session_id = None
            boss_metrics: Dict[str, Dict[str, Any]] = {}
        else:
            boss_session_id, boss_metrics = self._run_boss_phase(
                layout=layout,
                main_session_id=main_session_id,
                user_instruction=instruction.rstrip(),
                completion_info=completion_info,
            )

        candidates = self._build_candidates(layout, fork_map, boss_session_id, boss_path)
        artifact.boss_session_id = boss_session_id
        artifact.boss_path = boss_path if boss_session_id else None

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
            log_paths = self._log.record_cycle(
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
            artifact.log_paths = log_paths
            artifact.selected_session_id = main_session_id
            self._active_worker_sessions = []
            return result

        decision, scoreboard = self._auto_or_select(
            candidates,
            completion_info,
            selector,
            boss_metrics,
        )

        selected_info = self._validate_selection(decision, candidates)
        self._finalize_selection(selected=selected_info, main_pane=layout.main_pane)
        artifact.selected_session_id = selected_info.session_id

        result = OrchestrationResult(
            selected_session=selected_info.session_id,
            sessions_summary=scoreboard,
            artifact=artifact,
        )

        log_paths = self._log.record_cycle(
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
        artifact.log_paths = log_paths
        self._active_worker_sessions = []

        return result

    def force_complete_workers(self) -> int:
        if not self._active_worker_sessions:
            return 0
        self._monitor.force_completion(self._active_worker_sessions)
        return len(self._active_worker_sessions)

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
        resume_session_id: Optional[str],
    ) -> tuple[str, str]:
        if resume_session_id:
            self._monitor.bind_existing_session(
                pane_id=layout.main_pane,
                session_id=resume_session_id,
            )
            main_session_id = resume_session_id
        else:
            self._tmux.launch_main_session(pane_id=layout.main_pane)
            main_session_id = self._monitor.register_new_rollout(
                pane_id=layout.main_pane,
                baseline=baseline,
            )

        if self._main_session_hook:
            self._main_session_hook(main_session_id)

        user_instruction = instruction.rstrip()
        formatted_instruction = self._ensure_done_directive(user_instruction)
        fork_prompt = self._build_main_fork_prompt()

        self._tmux.send_instruction_to_pane(
            pane_id=layout.main_pane,
            instruction=fork_prompt,
        )
        self._monitor.wait_for_rollout_activity(
            main_session_id,
            timeout_seconds=10.0,
        )
        self._tmux.prepare_for_instruction(pane_id=layout.main_pane)
        self._monitor.capture_instruction(
            pane_id=layout.main_pane,
            instruction=fork_prompt,
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
        return fork_map

    def _dispatch_worker_instructions(
        self,
        *,
        layout: CycleLayout,
        user_instruction: str,
    ) -> None:
        for pane_id in layout.worker_panes:
            worker_name = layout.pane_to_worker[pane_id]
            worker_path = layout.pane_to_path.get(pane_id)
            if worker_path is None:
                continue
            self._tmux.prepare_for_instruction(pane_id=pane_id)
            location_notice = self._worktree_location_notice(custom_path=worker_path)
            base_message = (
                f"You are {worker_name}. Your dedicated worktree is `{worker_path}`.\n"
                "Do not respond with `/done` until you finish the task below.\n\n"
                "Task:\n"
                f"{user_instruction.rstrip()}"
            )
            message = self._ensure_done_directive(base_message, location_notice=location_notice)
            self._tmux.send_instruction_to_pane(
                pane_id=pane_id,
                instruction=message,
            )

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

        boss_metrics: Dict[str, Dict[str, Any]] = {}
        if self._boss_mode == BossMode.REWRITE:
            boss_metrics = self._wait_for_boss_scores(boss_session_id)
            followup = self._build_boss_rewrite_followup()
            if followup:
                self._tmux.prepare_for_instruction(pane_id=layout.boss_pane)
                self._tmux.send_instruction_to_pane(
                    pane_id=layout.boss_pane,
                    instruction=followup,
                )
        boss_completion = self._monitor.await_completion(session_ids=[boss_session_id])
        completion_info.update(boss_completion)
        if not boss_metrics:
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
        self._tmux.interrupt_pane(pane_id=main_pane)
        if selected.branch:
            self._worktree.merge_into_main(selected.branch)
        if selected.session_id:
            self._tmux.promote_to_main(session_id=selected.session_id, pane_id=main_pane)
            bind_existing = getattr(self._monitor, "bind_existing_session", None)
            if callable(bind_existing):
                try:
                    bind_existing(pane_id=main_pane, session_id=selected.session_id)
                except Exception:
                    pass
            consume = getattr(self._monitor, "consume_session_until_eof", None)
            if callable(consume):
                try:
                    consume(selected.session_id)
                except Exception:
                    pass

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

    def _ensure_done_directive(self, instruction: str, *, location_notice: Optional[str] = None) -> str:
        directive = "\n\nWhen you have completed the requested task, respond with exactly `/done`."
        notice = location_notice or self._worktree_location_notice()

        parts = [instruction.rstrip()]
        if notice and notice.strip() not in instruction:
            parts.append(notice.rstrip())
        if directive.strip() not in instruction:
            parts.append(directive)

        return "".join(parts)

    def _auto_or_select(
        self,
        candidates: List[CandidateInfo],
        completion_info: Mapping[str, Any],
        selector: Optional[Callable[[List[CandidateInfo]], SelectionDecision]],
        metrics: Optional[Mapping[str, Mapping[str, Any]]],
    ) -> tuple[SelectionDecision, Dict[str, Dict[str, Any]]]:
        base_scoreboard = self._build_scoreboard(candidates, completion_info, metrics)
        candidate_map = {candidate.key: candidate for candidate in candidates}
        if selector is None:
            raise RuntimeError(
                "Selection requires a selector; automatic boss scoring is not available."
            )

        try:
            decision = selector(candidates, base_scoreboard)
        except TypeError:
            decision = selector(candidates)

        selected_candidate = candidate_map.get(decision.selected_key)
        if selected_candidate and selected_candidate.session_id:
            refresh = getattr(self._monitor, "refresh_session_id", None)
            if callable(refresh):
                try:
                    resolved_id = refresh(selected_candidate.session_id)
                except Exception:
                    resolved_id = selected_candidate.session_id
                else:
                    if resolved_id and resolved_id != selected_candidate.session_id:
                        selected_candidate.session_id = resolved_id
                        entry = base_scoreboard.get(decision.selected_key)
                        if entry is not None:
                            entry["session_id"] = resolved_id

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

    def _worktree_location_hint(self, role: Optional[str] = None) -> str:
        base_dir = getattr(self._worktree, "worktrees_dir", None)
        base_path: Optional[Path] = None
        if base_dir is not None:
            try:
                base_path = Path(base_dir)
            except TypeError:
                base_path = None
        if base_path is not None:
            target_path = base_path / role if role else base_path
            return str(target_path)
        namespace = getattr(self._worktree, "session_namespace", None)
        if namespace:
            base = f".parallel-dev/sessions/{namespace}/worktrees"
        else:
            base = ".parallel-dev/worktrees"
        if role:
            return f"{base}/{role}"
        return base

    def _worktree_location_notice(self, role: Optional[str] = None, custom_path: Optional[Path] = None) -> str:
        hint = str(custom_path) if custom_path is not None else self._worktree_location_hint(role)
        target_path = str(custom_path) if custom_path is not None else hint
        return (
            "\n\nBefore you make any edits:\n"
            f"1. Run `pwd`. If the path does not contain `{hint}`, run `cd {target_path}`.\n"
            "2. Run `pwd` again to confirm you are now in the correct worktree.\n"
            "Keep every edit within this worktree and do not `cd` outside it.\n"
        )

    def _build_main_fork_prompt(self) -> str:
        return "Fork"

    def _build_boss_instruction(
        self,
        worker_names: Sequence[str],
        user_instruction: str,
    ) -> str:
        worker_lines = "\n".join(
            f"- Evaluate the proposal from {name}" for name in worker_names
        )
        base = (
            "You are the reviewer. The original user instruction was:\n"
            f"""{user_instruction}\n\n"""
            "Tasks:\n"
            f"{worker_lines}\n\n"
            "For each candidate, assign a numeric score between 0 and 100 and provide a short comment.\n"
            "Respond with JSON using the schema:\n"
            "{\n  \"scores\": {\n    \"worker-1\": {\"score\": <number>, \"comment\": <string>},\n"
            "    ... other candidates ...\n  }\n}\n"
            "Output only the JSON object for the evaluation.\n"
        )
        if self._boss_mode == BossMode.REWRITE:
            safety = self._worktree_location_notice(custom_path=self._worktree.boss_path).strip()
            return (
                "Boss evaluation and rewrite phase:\n"
                f"{base}"
                f"{safety}\n"
                "After you emit the JSON scoreboard, stay in this boss workspace and produce the final merged implementation.\n"
                "Refactor or combine the strongest worker contributions. If one worker result is already ideal, copy it into this boss workspace instead of rewriting from scratch.\n"
                "When the boss implementation is complete, respond with /done."
            )
        return (
            "Boss evaluation phase:\n"
            f"{base}"
            "After the JSON response, send /done."
        )

    def _build_boss_rewrite_followup(self) -> str:
        if self._boss_mode != BossMode.REWRITE:
            return ""
        return (
            "Boss integration phase:\n"
            "You have already produced the JSON scoreboard for the workers.\n"
            "Now stay in this boss workspace and deliver the final merged implementation.\n"
            "- Review the worker outputs you just scored and decide how to combine or refine them.\n"
            "- If one worker result is already ideal, copy it into this boss workspace; otherwise, refactor or merge the strongest parts.\n"
            "Make all required edits here so this boss workspace becomes the final solution.\n"
            "When the boss implementation is complete, respond with /done."
        )

    def _wait_for_boss_scores(self, boss_session_id: str, timeout: float = 120.0) -> Dict[str, Dict[str, Any]]:
        start = time.time()
        poll = getattr(self._monitor, "poll_interval", 1.0)
        try:
            interval = float(poll)
            if interval <= 0:
                interval = 1.0
        except (TypeError, ValueError):
            interval = 1.0
        metrics: Dict[str, Dict[str, Any]] = {}
        while time.time() - start < timeout:
            metrics = self._extract_boss_scores(boss_session_id)
            if metrics:
                break
            time.sleep(interval)
        return metrics

    def _maybe_pause(self, env_var: str, message: str) -> None:
        if os.getenv(env_var) == "1":
            input(message)
