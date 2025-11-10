"""High-level orchestration logic for parallel Codex sessions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Literal

from .stores import default_config_dir


class BossMode(str, Enum):
    SKIP = "skip"
    SCORE = "score"
    REWRITE = "rewrite"


class MergeMode(str, Enum):
    MANUAL = "manual"
    AUTO = "auto"


@dataclass(slots=True)
class OrchestrationResult:
    """Return value for a full orchestration cycle."""

    selected_session: str
    sessions_summary: Mapping[str, Any] = field(default_factory=dict)
    artifact: Optional["CycleArtifact"] = None
    continue_requested: bool = False
    merge_outcome: Optional["MergeOutcome"] = None


@dataclass(slots=True)
class CandidateInfo:
    key: str
    label: str
    session_id: Optional[str]
    branch: str
    worktree: Path
    pane_id: Optional[str] = None


@dataclass(slots=True)
class WorkerDecision:
    action: Literal["continue", "done"]
    instruction: Optional[str] = None


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


@dataclass(slots=True)
class SignalPaths:
    cycle_id: str
    root: Path
    worker_flags: Dict[str, Path]
    boss_flag: Path


@dataclass(slots=True)
class MergeOutcome:
    strategy: MergeMode
    status: Literal["skipped", "merged", "delegate", "failed"]
    branch: Optional[str] = None
    error: Optional[str] = None
    reason: Optional[str] = None


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
        worker_decider: Optional[Callable[[Mapping[str, str], Mapping[str, Any], "CycleLayout"], WorkerDecision]] = None,
        boss_mode: BossMode = BossMode.SCORE,
        log_hook: Optional[Callable[[str], None]] = None,
        merge_mode: MergeMode = MergeMode.MANUAL,
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
        self._active_signals: Optional[SignalPaths] = None
        self._log_hook = log_hook
        self._merge_mode = merge_mode if isinstance(merge_mode, MergeMode) else MergeMode(str(merge_mode))

    def set_main_session_hook(self, hook: Optional[Callable[[str], None]]) -> None:
        self._main_session_hook = hook

    def set_worker_decider(
        self,
        decider: Optional[Callable[[Mapping[str, str], Mapping[str, Any], CycleLayout], WorkerDecision]],
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
        self._cleanup_signal_paths()
        signal_paths: Optional[SignalPaths] = None

        try:
            layout, baseline = self._prepare_layout(worker_roots)
            signal_paths = self._prepare_signal_paths(layout.worker_names)
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
            worker_flag_map: Dict[str, Path] = signal_paths.worker_flags if signal_paths else {}
            self._dispatch_worker_instructions(
                layout=layout,
                user_instruction=instruction,
                signal_flags=worker_flag_map,
            )
            self._active_worker_sessions = [session_id for session_id in fork_map.values() if session_id]
            session_signal_map: Dict[str, Path] = {}
            for pane_id, session_id in fork_map.items():
                worker_name = layout.pane_to_worker.get(pane_id)
                flag_path = worker_flag_map.get(worker_name) if worker_name else None
                if session_id and flag_path is not None:
                    session_signal_map[session_id] = flag_path
            completion_info = self._await_worker_completion(fork_map, session_signal_map)

            while True:
                if self._worker_decider:
                    try:
                        worker_decision = self._worker_decider(fork_map, completion_info, layout)
                    except Exception:
                        worker_decision = WorkerDecision(action="done")
                else:
                    worker_decision = WorkerDecision(action="done")

                if worker_decision.action == "continue":
                    continuation_text = (worker_decision.instruction or "").strip()
                    if not continuation_text:
                        raise RuntimeError("/continue が選択されましたが追加指示が取得できませんでした。")
                    self._dispatch_worker_continuation(
                        layout=layout,
                        user_instruction=continuation_text,
                        signal_flags=worker_flag_map,
                    )
                    completion_info = self._await_worker_completion(fork_map, session_signal_map)
                    continue
                break

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
                boss_session_id=None,
                worker_paths=worker_paths,
                boss_path=boss_path,
                instruction=formatted_instruction,
                tmux_session=self._session_name,
            )

            if self._boss_mode == BossMode.SKIP:
                boss_session_id = None
                boss_metrics: Dict[str, Dict[str, Any]] = {}
            else:
                if not signal_paths:
                    raise RuntimeError("Signal paths were not initialized for boss phase.")
                boss_session_id, boss_metrics = self._run_boss_phase(
                    layout=layout,
                    main_session_id=main_session_id,
                    user_instruction=instruction.rstrip(),
                    completion_info=completion_info,
                    boss_flag=signal_paths.boss_flag,
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
            self._phase_log(f"候補 {selected_info.key} を採択しました。", status="マージ処理中")
            if self._merge_mode == MergeMode.AUTO:
                merge_outcome = self._auto_commit_and_finalize(
                    selected=selected_info,
                    layout=layout,
                    signal_paths=signal_paths,
                )
            else:
                merge_outcome = self._finalize_selection(selected=selected_info, main_pane=layout.main_pane)
            artifact.selected_session_id = selected_info.session_id

            result = OrchestrationResult(
                selected_session=selected_info.session_id,
                sessions_summary=scoreboard,
                artifact=artifact,
                merge_outcome=merge_outcome,
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
        finally:
            self._cleanup_signal_paths()

    def force_complete_workers(self) -> int:
        if not self._active_worker_sessions:
            return 0
        sessions = list(self._active_worker_sessions)
        self._monitor.force_completion(sessions)
        self._active_worker_sessions = []
        return len(sessions)

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

    def _prepare_signal_paths(self, worker_names: Sequence[str]) -> SignalPaths:
        raw_namespace = getattr(self._worktree, "session_namespace", None)
        if isinstance(raw_namespace, str) and raw_namespace.strip():
            namespace = raw_namespace
        else:
            namespace = "default"
        cycle_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        root = self._signal_base_dir(namespace) / cycle_id
        root.mkdir(parents=True, exist_ok=True)
        worker_flags = {name: root / f"{name}.done" for name in worker_names}
        boss_flag = root / "boss.done"
        bundle = SignalPaths(
            cycle_id=cycle_id,
            root=root,
            worker_flags=worker_flags,
            boss_flag=boss_flag,
        )
        self._active_signals = bundle
        return bundle

    def _signal_base_dir(self, namespace: str) -> Path:
        signals_root = default_config_dir() / "sessions" / namespace / "signals"
        signals_root.mkdir(parents=True, exist_ok=True)
        return signals_root

    def _cleanup_signal_paths(self) -> None:
        bundle = self._active_signals
        if not bundle:
            return
        try:
            shutil.rmtree(bundle.root)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        finally:
            self._active_signals = None

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
        signal_flags: Mapping[str, Path],
    ) -> None:
        for pane_id in layout.worker_panes:
            worker_name = layout.pane_to_worker[pane_id]
            worker_path = layout.pane_to_path.get(pane_id)
            if worker_path is None:
                continue
            completion_flag = signal_flags.get(worker_name)
            self._tmux.prepare_for_instruction(pane_id=pane_id)
            location_notice = self._worktree_location_notice(custom_path=worker_path)
            base_message = (
                f"You are {worker_name}. Your dedicated worktree is `{worker_path}`.\n"
                "Task:\n"
                f"{user_instruction.rstrip()}"
            )
            message = self._ensure_done_directive(
                base_message,
                location_notice=location_notice,
                completion_flag=completion_flag,
            )
            self._tmux.send_instruction_to_pane(
                pane_id=pane_id,
                instruction=message,
            )
        self._phase_log("ワーカー実行を開始しました。", status="ワーカー実行中")

    def _dispatch_worker_continuation(
        self,
        *,
        layout: CycleLayout,
        user_instruction: str,
        signal_flags: Mapping[str, Path],
    ) -> None:
        trimmed = user_instruction.rstrip("\n")
        for pane_id in layout.worker_panes:
            worker_name = layout.pane_to_worker.get(pane_id)
            worker_path = layout.pane_to_path.get(pane_id)
            if not worker_name or worker_path is None:
                continue
            self._tmux.send_instruction_to_pane(
                pane_id=pane_id,
                instruction=trimmed,
            )

    def _await_worker_completion(
        self,
        fork_map: Mapping[str, str],
        signal_map: Mapping[str, Path],
    ) -> Dict[str, Any]:
        completion_info = self._monitor.await_completion(
            session_ids=list(fork_map.values()),
            signal_paths=signal_map,
        )
        if os.getenv("PARALLEL_DEV_DEBUG_STATE") == "1":
            print("[parallel-dev] Worker completion status:", completion_info)
        self._phase_log("ワーカー処理が完了しました。", status="採点準備中")
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
        boss_flag: Path,
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

        self._maybe_pause(
            "PARALLEL_DEV_PAUSE_BEFORE_BOSS",
            "[parallel-dev] All workers reported completion. Inspect boss pane, then press Enter to send boss instructions...",
        )

        boss_instruction = self._build_boss_instruction(
            layout.worker_names,
            user_instruction,
        )
        self._phase_log("採点フェーズを開始します。", status="採点中")
        self._tmux.send_instruction_to_pane(
            pane_id=layout.boss_pane,
            instruction=boss_instruction,
        )

        boss_metrics = self._wait_for_boss_scores(boss_session_id)
        if not boss_metrics:
            boss_metrics = self._extract_boss_scores(boss_session_id)

        if self._boss_mode == BossMode.REWRITE:
            followup = self._build_boss_rewrite_followup(boss_flag=boss_flag)
            if followup:
                self._tmux.send_instruction_to_pane(
                    pane_id=layout.boss_pane,
                    instruction=followup,
                )
            boss_completion = self._monitor.await_completion(
                session_ids=[boss_session_id],
                signal_paths={boss_session_id: boss_flag},
            )
            completion_info.update(boss_completion)
        else:
            completion_info[boss_session_id] = {"done": True, "scores_detected": True}

        self._phase_log("採点フェーズが完了しました。", status="採択待ち")
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
            resolved_session = self._resolve_session_id(session_id) or session_id
            candidates.append(
                CandidateInfo(
                    key=worker_name,
                    label=f"{worker_name} (session {resolved_session})",
                    session_id=resolved_session,
                    branch=branch_name,
                    worktree=worktree_path,
                    pane_id=pane_id,
                )
            )

        include_boss = (
            boss_session_id
            and self._boss_mode == BossMode.REWRITE
        )
        if include_boss:
            resolved_boss = self._resolve_session_id(boss_session_id) or boss_session_id
            candidates.append(
                CandidateInfo(
                    key="boss",
                    label=f"boss (session {resolved_boss})",
                    session_id=resolved_boss,
                    branch=self._worktree.boss_branch,
                    worktree=boss_path,
                    pane_id=layout.boss_pane,
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
        if selected_info.key == "boss" and self._boss_mode != BossMode.REWRITE:
            raise ValueError("Boss candidate is only available in rewrite mode.")
        return selected_info

    def _finalize_selection(
        self,
        *,
        selected: CandidateInfo,
        main_pane: str,
        delegate_reason: Optional[str] = None,
    ) -> MergeOutcome:
        self._tmux.interrupt_pane(pane_id=main_pane)
        reason = delegate_reason
        if reason is None:
            if self._merge_mode == MergeMode.AUTO:
                reason = "agent_auto"
            else:
                reason = "manual_user"
        merge_outcome = MergeOutcome(
            strategy=self._merge_mode,
            status="delegate",
            branch=selected.branch,
            reason=reason,
        )
        if selected.session_id:
            self._tmux.promote_to_main(session_id=selected.session_id, pane_id=main_pane)
            bind_existing = getattr(self._monitor, "bind_existing_session", None)
            if callable(bind_existing):
                try:
                    bind_existing(pane_id=main_pane, session_id=selected.session_id)
                except Exception:
                    pass
            self._phase_log("メインセッションを再開しました。", status="再開中")
        consume = getattr(self._monitor, "consume_session_until_eof", None)
        if callable(consume):
            try:
                consume(selected.session_id)
            except Exception:
                pass
        return merge_outcome

    def _auto_commit_and_finalize(
        self,
        *,
        selected: CandidateInfo,
        layout: CycleLayout,
        signal_paths: Optional[SignalPaths],
    ) -> MergeOutcome:
        pane_id = selected.pane_id or self._resolve_candidate_pane(selected.key, layout)
        if pane_id is None:
            return self._finalize_selection(
                selected=selected,
                main_pane=layout.main_pane,
                delegate_reason="manual_user",
            )

        flag_path = self._resolve_flag_path(selected.key, signal_paths)
        if flag_path:
            try:
                flag_path.unlink()
            except FileNotFoundError:
                pass

        instruction = self._build_agent_commit_instruction(selected, flag_path)
        self._phase_log("採択エージェントへコミット/統合手順を送信しました。", status="コミット指示中")
        self._tmux.prepare_for_instruction(pane_id=pane_id)
        self._tmux.send_instruction_to_pane(pane_id=pane_id, instruction=instruction)

        if flag_path:
            self._phase_log("コミット結果を待っています。", status="コミット待ち")
            self._wait_for_flag(flag_path)

        self._phase_log("コミット報告を受け取りました。", status="マージ処理中")
        self._pull_main_after_auto()
        return self._finalize_selection(
            selected=selected,
            main_pane=layout.main_pane,
            delegate_reason="agent_auto",
        )

    def _resolve_candidate_pane(self, key: str, layout: CycleLayout) -> Optional[str]:
        if key == "boss":
            return layout.boss_pane
        for pane_id, worker_name in layout.pane_to_worker.items():
            if worker_name == key:
                return pane_id
        return None

    def _resolve_flag_path(self, key: str, signal_paths: Optional[SignalPaths]) -> Optional[Path]:
        if not signal_paths:
            return None
        if key == "boss":
            return signal_paths.boss_flag
        return signal_paths.worker_flags.get(key)

    def _build_agent_commit_instruction(self, selected: CandidateInfo, flag_path: Optional[Path]) -> str:
        worktree = str(selected.worktree)
        branch = selected.branch
        commit_message = f"Auto update from {selected.key}"
        flag_text = str(flag_path) if flag_path else None

        lines = [
            "追加タスク: あなたが編集した成果物を本レポジトリへ統合してください。",
            f"- 作業ディレクトリ: {worktree}",
            f"- ブランチ: {branch}",
            "",
            "【目的】",
            "- 必要なファイルだけをコミットし、main を fast-forward で最新化する",
            "- 進捗や問題があれば必ずログに記録する",
            "",
            "【禁止事項】",
            "- `git push --force` や `git reset --hard origin/main` などの破壊的操作",
            "- main 以外のブランチを削除・上書きすること",
            "- エラーを黙って無視すること",
            "",
            "【推奨フロー（状況に応じて調整可）】",
            f"1. cd {worktree} && git status -sb で変更内容を確認",
            "2. コミット対象のみ git add （例: dev_test, docs/experiment.yaml など）",
            f"3. git commit -m \"{commit_message}\" （既存コミットを使い回さない）",
            f"4. main を最新化して fast-forward で統合（例: git checkout main && git pull --ff-only && git merge --ff-only {branch} && git checkout {branch})",
            "5. 衝突が発生した場合は安全な方法で解消し、解決できなければエラー内容とともに報告",
        ]
        if flag_text:
            lines.append(f"6. コミット/統合が完了したら `touch {flag_text}` で完了を通知してください。")
        lines.append("※ 進め方に迷った場合やエラーが発生した場合は状況とログを共有してください。")
        return "\n".join(lines)

    def _pull_main_after_auto(self) -> None:
        root = getattr(self._worktree, "root", None)
        if not root:
            return
        ff_cmd = ["git", "-C", str(root), "pull", "--ff-only"]
        try:
            subprocess.run(
                ff_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._log_merge_sync("git pull --ff-only")
            return
        except subprocess.CalledProcessError as exc:  # noqa: PERF203
            self._log_git_failure("git pull --ff-only", exc)

        remote, branch = self._resolve_upstream_tracking(root)
        rebase_cmd = ["git", "-C", str(root), "pull", "--rebase", remote, branch]
        try:
            subprocess.run(
                rebase_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._log_merge_sync(f"git pull --rebase {remote} {branch}")
            return
        except subprocess.CalledProcessError as exc:  # noqa: PERF203
            conflict_msg = (
                "[merge] ホスト側 git pull --rebase {remote} {branch} が失敗しました: {detail}. "
                "衝突を解消し `git rebase --continue` したら /done で次へ進めてください。"
            ).format(remote=remote, branch=branch, detail=exc.stderr or exc.stdout or str(exc))
            if self._log_hook:
                try:
                    self._log_hook(conflict_msg)
                except Exception:
                    pass

    def _resolve_upstream_tracking(self, root: Path) -> tuple[str, str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            upstream = result.stdout.strip()
            if upstream and "/" in upstream:
                remote, branch = upstream.split("/", 1)
                return remote, branch
        except subprocess.CalledProcessError:
            pass
        return "origin", "main"

    def _log_git_failure(self, action: str, exc: subprocess.CalledProcessError) -> None:
        if not self._log_hook:
            return
        message = f"[merge] ホスト側 {action} が失敗しました: {exc.stderr or exc.stdout or exc}"
        try:
            self._log_hook(message)
        except Exception:
            pass

    def _log_merge_sync(self, action: str) -> None:
        if not self._log_hook:
            return
        message = f"[merge] ホスト側 {action} で main を同期しました。"
        try:
            self._log_hook(message)
        except Exception:
            pass

    def _wait_for_flag(self, flag_path: Path, timeout: float = 600.0) -> None:
        deadline = time.time() + timeout
        flag_path.parent.mkdir(parents=True, exist_ok=True)
        while time.time() < deadline:
            if flag_path.exists():
                return
            time.sleep(1.0)
        if self._log_hook:
            try:
                self._log_hook(f"[merge] 完了フラグ {flag_path} が {timeout}s 以内に検出できませんでした。")
            except Exception:
                pass

    def _phase_log(self, message: str, status: Optional[str] = None) -> None:
        if not self._log_hook:
            return
        payload = f"[phase] {message}"
        if status:
            payload = f"{payload} ::status::{status}"
        try:
            self._log_hook(payload)
        except Exception:
            pass

    def _resolve_session_id(self, session_id: Optional[str]) -> Optional[str]:
        if not session_id:
            return session_id
        refresher = getattr(self._monitor, "refresh_session_id", None)
        if callable(refresher):
            try:
                resolved = refresher(session_id)
            except Exception:
                return session_id
            return resolved or session_id
        return session_id

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

    def _ensure_done_directive(
        self,
        instruction: str,
        *,
        location_notice: Optional[str] = None,
        completion_flag: Optional[Path] = None,
    ) -> str:
        if completion_flag is not None:
            flag_text = str(completion_flag)
            directive = (
                "\n\nCompletion protocol:\n"
                f"- When the entire task is complete, run `touch {flag_text}` (no markdown, single command).\n"
                "- The host watches that file and will automatically continue once it exists—no `/done` line is required.\n"
                f"- If you signaled completion too early, remove the flag with `rm -f {flag_text}` and keep working."
            )
        else:
            directive = (
                "\n\nCompletion protocol:\n"
                "- After you finish the requested work and share any summary, you MUST send a new line containing only `/done`.\n"
                "- Do not describe completion in prose or embed `/done` inside a sentence; the standalone `/done` line is mandatory.\n"
                "Tasks are treated as unfinished until that literal `/done` line is sent."
            )
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
            # Try a direct parse first (handles single-line JSON responses).
            cleaned = raw_text.strip()
            if cleaned and cleaned != "/done":
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

            def _sanitize(line: str) -> str:
                stripped = line.strip()
                if stripped.startswith("• "):
                    return stripped[2:]
                if stripped.startswith(('-', '*')) and len(stripped) > 1 and stripped[1] == ' ':
                    return stripped[2:]
                return stripped

            lines = [_sanitize(line) for line in raw_text.splitlines() if line.strip() and line.strip() != "/done"]
            buffer: List[str] = []
            depth = 0
            capturing = False
            for line in lines:
                if not capturing:
                    if line.startswith('{'):
                        capturing = True
                        depth = 0
                        buffer = []
                    else:
                        continue
                buffer.append(line)
                depth += line.count('{') - line.count('}')
                if depth <= 0:
                    candidate = "\n".join(buffer)
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        capturing = False
                        buffer = []
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
        worker_paths: Dict[str, Path] = getattr(self._worktree, "_worker_paths", {})
        worker_lines: List[str] = []
        for name in worker_names:
            path = worker_paths.get(name)
            if path is None:
                path = Path(self._worktree_location_hint(role=name))
            worker_lines.append(f"- {name} (worktree: {path})")
        worker_section = "\n".join(worker_lines)

        instruction = (
            "Boss evaluation phase:\n"
            "You are the reviewer. The original user instruction was:\n"
            f"{user_instruction}\n\n"
            "Candidates:\n"
            f"{worker_section}\n\n"
            "Tasks:\n"
            "- Review each worker proposal and assess its quality.\n"
            "- For each candidate, assign a numeric score between 0 and 100 and provide a short comment.\n\n"
            "Evaluation checklist:\n"
            "- Confirm whether each worker obeyed the user instruction exactly.\n"
            "- Penalize answers that responded only with `/done` or omitted required content.\n"
            "- Note formatting mistakes or any extra/unrequested text.\n\n"
            "Respond with JSON only, using the structure:\n"
            "{\n"
            '  "scores": {\n'
            '    "worker-1": {"score": <number>, "comment": "<string>"},\n'
            "    ... other candidates ...\n"
            "  }\n"
            "}\n\n"
            "Output only the JSON object for the evaluation—do NOT return Markdown or prose at this stage.\n"
        )
        if self._boss_mode == BossMode.REWRITE:
            instruction += (
                "After you emit the JSON scoreboard, wait for the follow-up instructions to perform the final integration."
            )
        else:
            instruction += "After the JSON response, stop and wait for the host to continue."

        notice = self._worktree_location_notice(role="boss", custom_path=self._worktree.boss_path)
        parts = [instruction.rstrip()]
        if notice and notice.strip() not in instruction:
            parts.append(notice)
        return "".join(parts)

    def _build_boss_rewrite_followup(self, *, boss_flag: Path) -> str:
        if self._boss_mode != BossMode.REWRITE:
            return ""
        flag_text = str(boss_flag)
        return (
            "Boss integration phase:\n"
            "You have already produced the JSON scoreboard for the workers.\n"
            "Now stay in this boss workspace and deliver the final merged implementation.\n"
            "- Review the worker outputs you just scored and decide how to combine or refine them.\n"
            "- If one worker result is already ideal, copy it into this boss workspace; otherwise, refactor or merge the strongest parts.\n"
            f"When the integration is completely finished, run `touch {flag_text}` to signal completion.\n"
            f"If you need to continue editing after signaling, remove the flag with `rm -f {flag_text}` and keep working."
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
