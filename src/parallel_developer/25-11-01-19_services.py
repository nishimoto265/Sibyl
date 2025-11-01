"""Service layer components for tmux, worktree, Codex monitoring, and Boss evaluation."""

from __future__ import annotations

import json
import shlex
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import git
import libtmux
import yaml

from .orchestrator import CandidateInfo, SelectionDecision

class TmuxLayoutManager:
    """Manage tmux session layout for parallel Codex agents."""

    def __init__(
        self,
        session_name: str,
        worker_count: int,
        monitor: "CodexMonitor",
        *,
        root_path: Path,
        startup_delay: float = 0.0,
    ) -> None:
        self.session_name = session_name
        self.worker_count = worker_count
        self.monitor = monitor
        self.root_path = Path(root_path)
        self.boss_path = self.root_path
        self.startup_delay = startup_delay
        self._server = libtmux.Server()

    def ensure_layout(self, *, session_name: str, worker_count: int) -> Dict[str, Any]:
        if session_name != self.session_name or worker_count != self.worker_count:
            raise ValueError("session_name/worker_count mismatch with manager configuration")

        session = self._get_or_create_session(fresh=True)
        window = getattr(session, "attached_window", None) or session.windows[0]

        target_pane_count = self.worker_count + 2  # main + boss + workers
        while len(window.panes) < target_pane_count:
            window.split_window(attach=False)
        window.select_layout("tiled")

        panes = window.panes
        layout = {
            "main": panes[0].pane_id,
            "boss": panes[1].pane_id,
            "workers": [pane.pane_id for pane in panes[2 : 2 + self.worker_count]],
        }
        return layout

    def launch_main_session(self, *, pane_id: str) -> None:
        command = (
            f"cd {shlex.quote(str(self.root_path))} && "
            f"codex --cd {shlex.quote(str(self.root_path))}"
        )
        self._send_command(pane_id, command)
        self._maybe_wait()

    def launch_boss_session(self, *, pane_id: str) -> None:
        command = (
            f"cd {shlex.quote(str(self.boss_path))} && "
            f"codex --cd {shlex.quote(str(self.boss_path))}"
        )
        self._send_command(pane_id, command)
        self._maybe_wait()

    def fork_workers(self, *, workers: Iterable[str], base_session_id: str) -> Dict[str, str]:
        worker_list = list(workers)
        for pane_id in worker_list:
            pane = self._get_pane(pane_id)
            for key in ("Escape", "Escape", "Enter"):
                pane.cmd("send-keys", "-t", pane_id, key)
        return worker_list

    def resume_workers(self, fork_map: Mapping[str, str], pane_paths: Mapping[str, Path]) -> None:
        for pane_id, session_id in fork_map.items():
            worker_path = pane_paths.get(pane_id, self.root_path)
            command = (
                f"cd {shlex.quote(str(worker_path))} && "
                f"codex resume {shlex.quote(session_id)}"
            )
            self._send_command(pane_id, command)
        self._maybe_wait()

    def send_instruction_to_pane(self, *, pane_id: str, instruction: str) -> None:
        self._send_text(pane_id, instruction)

    def send_instruction_to_workers(self, fork_map: Mapping[str, str], instruction: str) -> None:
        for pane_id in fork_map:
            self._send_text(pane_id, instruction)

    def promote_to_main(self, *, session_id: str, pane_id: str) -> None:
        pane = self._get_pane(pane_id)
        pane.send_keys(f"codex resume {session_id}", enter=True)

    def interrupt_pane(self, *, pane_id: str) -> None:
        pane = self._get_pane(pane_id)
        pane.cmd("send-keys", "-t", pane_id, "Escape")

    def _get_or_create_session(self, fresh: bool = False):
        session = self._server.find_where({"session_name": self.session_name})
        if session is not None and not fresh:
            return session

        kill_existing = fresh and session is not None
        session = self._server.new_session(
            session_name=self.session_name,
            attach=False,
            kill_session=kill_existing,
        )
        return session

    def _get_pane(self, pane_id: str):
        for session in self._server.sessions:
            for window in session.windows:
                for pane in window.panes:
                    if pane.pane_id == pane_id:
                        return pane
        raise RuntimeError(f"Pane {pane_id!r} not found in tmux session {self.session_name}")

    def _send_command(self, pane_id: str, command: str) -> None:
        pane = self._get_pane(pane_id)
        pane.send_keys(command, enter=True)

    def _maybe_wait(self) -> None:
        if self.startup_delay > 0:
            time.sleep(self.startup_delay)

    def _send_text(self, pane_id: str, text: str) -> None:
        pane = self._get_pane(pane_id)
        pane.cmd("send-keys", "-t", pane_id, "-l", text)
        pane.cmd("send-keys", "-t", pane_id, "C-m")


class WorktreeManager:
    """Manage git worktrees for each worker."""

    def __init__(self, root: Path, worker_count: int) -> None:
        self.root = Path(root)
        self.worker_count = worker_count
        self._repo = git.Repo(self.root)
        self._ensure_repo_initialized()
        self.worktrees_dir = self.root / ".parallel-dev" / "worktrees"
        self.boss_path = self.root / ".parallel-dev" / "boss"
        self._worker_branch_template = "parallel-dev/{name}"
        self._boss_branch = "parallel-dev/boss"

    def prepare(self) -> Dict[str, Path]:
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        mapping: Dict[str, Path] = {}
        for index in range(1, self.worker_count + 1):
            worker_name = f"worker-{index}"
            worktree_path = self.worktrees_dir / worker_name
            branch_name = self.worker_branch(worker_name)
            self._recreate_worktree(worktree_path, branch_name)
            mapping[worker_name] = worktree_path
        self._recreate_worktree(self.boss_path, self.boss_branch)
        return mapping

    def _ensure_repo_initialized(self) -> None:
        try:
            _ = self._repo.head.commit  # type: ignore[attr-defined]
        except ValueError as exc:
            raise RuntimeError(
                "Git repository has no commits. Create an initial commit before running parallel-dev."
            ) from exc

    def _recreate_worktree(self, path: Path, branch_name: str) -> None:
        if path.exists():
            try:
                self._repo.git.worktree("remove", "--force", str(path))
            except git.GitCommandError:
                shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        self._repo.git.worktree(
            "add",
            "-B",
            branch_name,
            str(path),
            "HEAD",
        )

    def worker_branch(self, worker_name: str) -> str:
        return self._worker_branch_template.format(name=worker_name)

    @property
    def boss_branch(self) -> str:
        return self._boss_branch

    def merge_into_main(self, branch_name: str) -> None:
        if self._repo.is_dirty(untracked_files=False):
            raise RuntimeError("Main repository has uncommitted changes; cannot merge results.")
        self._repo.git.merge(branch_name, "--ff-only")


class CodexMonitor:
    """Inspect Codex rollout JSONL files and monitor completion."""

    def __init__(
        self,
        logs_dir: Path,
        session_map_path: Path,
        *,
        codex_sessions_root: Optional[Path] = None,
        poll_interval: float = 1.0,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.session_map_path = Path(session_map_path)
        self.poll_interval = poll_interval
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "sessions").mkdir(parents=True, exist_ok=True)
        if not self.session_map_path.exists():
            self.session_map_path.write_text("{}\n", encoding="utf-8")
        self.codex_sessions_root = (
            Path(codex_sessions_root)
            if codex_sessions_root is not None
            else Path.home() / ".codex" / "sessions"
        )

    def register_session(self, *, pane_id: str, session_id: str, rollout_path: Path) -> None:
        data = self._load_map()
        panes = data.setdefault("panes", {})
        sessions = data.setdefault("sessions", {})
        panes[pane_id] = {
            "session_id": session_id,
            "rollout_path": str(rollout_path),
        }
        sessions[session_id] = {
            "pane_id": pane_id,
            "rollout_path": str(rollout_path),
        }
        self._write_map(data)

    def snapshot_rollouts(self) -> Dict[Path, float]:
        if not self.codex_sessions_root.exists():
            return {}
        result: Dict[Path, float] = {}
        for path in self.codex_sessions_root.glob("**/rollout-*.jsonl"):
            try:
                result[path] = path.stat().st_mtime
            except FileNotFoundError:
                continue
        return result

    def register_new_rollout(
        self,
        *,
        pane_id: str,
        baseline: Mapping[Path, float],
        timeout_seconds: float = 30.0,
    ) -> str:
        paths = self._wait_for_new_rollouts(baseline, expected=1, timeout_seconds=timeout_seconds)
        if not paths:
            raise TimeoutError("Failed to detect new Codex session rollout")
        rollout_path = paths[0]
        session_id = self._parse_session_meta(rollout_path)
        self.register_session(pane_id=pane_id, session_id=session_id, rollout_path=rollout_path)
        return session_id

    def register_worker_rollouts(
        self,
        *,
        worker_panes: Sequence[str],
        baseline: Mapping[Path, float],
        timeout_seconds: float = 30.0,
    ) -> Dict[str, str]:
        if not worker_panes:
            return {}
        paths = self._wait_for_new_rollouts(
            baseline,
            expected=len(worker_panes),
            timeout_seconds=timeout_seconds,
        )
        paths = paths[: len(worker_panes)]
        fork_map: Dict[str, str] = {}
        for pane_id, path in zip(worker_panes, paths):
            session_id = self._parse_session_meta(path)
            self.register_session(pane_id=pane_id, session_id=session_id, rollout_path=path)
            fork_map[pane_id] = session_id
        return fork_map

    def capture_instruction(self, *, pane_id: str, instruction: str) -> str:
        data = self._load_map()
        pane_entry = data.get("panes", {}).get(pane_id)
        if pane_entry is None:
            raise RuntimeError(
                f"Pane {pane_id!r} is not registered in session_map; ensure Codex session detection succeeded."
            )

        instruction_log = self.logs_dir / "instruction.log"
        with instruction_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"pane": pane_id, "instruction": instruction}) + "\n")

        return pane_entry["session_id"]

    def await_completion(
        self,
        *,
        session_ids: Iterable[str],
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        data = self._load_map()
        sessions = data.get("sessions", {})

        targets: Dict[str, Path] = {}
        for session_id in session_ids:
            entry = sessions.get(session_id)
            if entry is None:
                raise RuntimeError(f"Session {session_id!r} not found in session_map")
            targets[session_id] = Path(entry["rollout_path"])

        remaining = set(targets)
        completion: Dict[str, Any] = {}
        deadline = None if timeout_seconds is None else time.time() + timeout_seconds

        while remaining:
            for session_id in list(remaining):
                rollout_path = targets[session_id]
                if self._contains_done(rollout_path):
                    completion[session_id] = {"done": True, "rollout_path": str(rollout_path)}
                    remaining.remove(session_id)
            if not remaining:
                break
            if deadline is not None and time.time() >= deadline:
                break
            time.sleep(self.poll_interval)

        for session_id in remaining:
            completion[session_id] = {"done": False, "rollout_path": str(targets[session_id])}

        return completion

    def _wait_for_new_rollouts(
        self,
        baseline: Mapping[Path, float],
        *,
        expected: int,
        timeout_seconds: float,
    ) -> List[Path]:
        deadline = time.time() + timeout_seconds
        baseline_paths = set(baseline.keys())
        while True:
            current = self.snapshot_rollouts()
            new_paths = [path for path in current.keys() if path not in baseline_paths]
            if len(new_paths) >= expected:
                new_paths.sort(key=lambda p: current.get(p, 0.0))
                return new_paths
            if time.time() >= deadline:
                new_paths.sort(key=lambda p: current.get(p, 0.0))
                return new_paths
            time.sleep(self.poll_interval)

    def _parse_session_meta(self, rollout_path: Path) -> str:
        try:
            with rollout_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") == "session_meta" and "payload" in obj:
                        return obj["payload"].get("id")
        except FileNotFoundError:
            pass
        suffix = int(time.time() * 1000)
        return f"unknown-{suffix}"

    def _load_map(self) -> Dict[str, Any]:
        text = self.session_map_path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        return yaml.safe_load(text) or {}

    def _write_map(self, data: Mapping[str, Any]) -> None:
        self.session_map_path.write_text(yaml.safe_dump(dict(data), sort_keys=True), encoding="utf-8")

    def _contains_done(self, rollout_path: Path) -> bool:
        if not rollout_path.exists():
            return False
        try:
            with rollout_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if "/done" in line:
                        return True
        except OSError:
            return False
        return False


class BossManager:
    """Score worker sessions and compile scoreboard information."""

    def __init__(self, repo_path: Path) -> None:
        self._repo = git.Repo(repo_path)

    def finalize_scores(
        self,
        candidates: List[CandidateInfo],
        decision: SelectionDecision,
        completion: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        return self._build_summary(candidates, decision.scores, decision.comments, completion)

    def auto_select(
        self,
        candidates: List[CandidateInfo],
        completion: Mapping[str, Any],
    ) -> tuple[SelectionDecision, Dict[str, Dict[str, Any]]]:
        scores: Dict[str, float] = {}
        comments: Dict[str, str] = {}

        for candidate in candidates:
            score, comment = self._auto_score_candidate(candidate, completion)
            scores[candidate.key] = score
            if comment:
                comments[candidate.key] = comment

        best_key = max(scores, key=scores.get)
        summary = self._build_summary(candidates, scores, comments, completion)
        decision = SelectionDecision(selected_key=best_key, scores=scores, comments=comments)
        return decision, summary

    # Helpers -----------------------------------------------------------------
    def _build_summary(
        self,
        candidates: List[CandidateInfo],
        scores: Mapping[str, float],
        comments: Mapping[str, str],
        completion: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            session_id = candidate.session_id
            entry: Dict[str, Any] = {
                "score": float(scores.get(candidate.key, 0.0)),
                "comment": comments.get(candidate.key, ""),
                "session_id": session_id,
                "branch": candidate.branch,
                "worktree": str(candidate.worktree),
            }
            if session_id and session_id in completion:
                entry.update(completion[session_id])
            else:
                entry.setdefault("done", True)
            summary[candidate.key] = entry
        return summary

    def _auto_score_candidate(
        self,
        candidate: CandidateInfo,
        completion: Mapping[str, Any],
    ) -> tuple[float, str]:
        session_id = candidate.session_id
        info = completion.get(session_id or "", {}) if session_id else {}
        done = bool(info.get("done", True))

        if not done:
            return 0.0, "未完了"

        stats = self._diff_stats(candidate.branch)
        score = 60.0
        comment_parts = []

        files = stats.get("files", 0)
        insertions = stats.get("insertions", 0)
        deletions = stats.get("deletions", 0)

        if files > 0:
            score += min(15.0, files * 3.0)
            comment_parts.append(f"files={files}")
        if insertions > 0:
            score += min(15.0, insertions / 10.0)
            comment_parts.append(f"+{insertions}")
        if deletions > 0:
            score += min(5.0, deletions / 20.0)
            comment_parts.append(f"-{deletions}")
        if info.get("errors"):
            score -= 10.0
            comment_parts.append("errors detected")

        score = max(0.0, min(100.0, score))
        comment = ", ".join(comment_parts)
        return score, comment

    def _diff_stats(self, branch_name: str) -> Dict[str, int]:
        stats = {"files": 0, "insertions": 0, "deletions": 0}
        try:
            numstat = self._repo.git.diff("HEAD", branch_name, "--numstat")
        except git.GitCommandError:
            return stats

        for line in numstat.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                insertions, deletions = parts[:2]
                try:
                    stats["insertions"] += int(insertions)
                    stats["deletions"] += int(deletions)
                    stats["files"] += 1
                except ValueError:
                    continue
        return stats


class LogManager:
    """Aggregate orchestration logs."""

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.cycles_dir = self.logs_dir / "cycles"
        self.cycles_dir.mkdir(parents=True, exist_ok=True)

    def record_cycle(
        self,
        *,
        instruction: str,
        layout: Mapping[str, Any],
        fork_map: Mapping[str, str],
        completion: Mapping[str, Any],
        result: Any,
    ) -> None:
        timestamp = datetime.utcnow().strftime("%y-%m-%d-%H%M%S")
        payload = {
            "instruction": instruction,
            "layout": dict(layout),
            "fork_map": dict(fork_map),
            "completion": dict(completion),
            "result": {
                "selected_session": getattr(result, "selected_session", None),
                "sessions_summary": getattr(result, "sessions_summary", None),
            },
        }
        path = self.cycles_dir / f"{timestamp}.yaml"
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
