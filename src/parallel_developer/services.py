"""Service layer components for tmux, worktree, Codex monitoring, and Boss evaluation."""

from __future__ import annotations

import json
import shlex
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import git
import libtmux
import yaml


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
        backtrack_delay: float = 0.2,
        reuse_existing_session: bool = False,
        session_namespace: Optional[str] = None,
        codex_home: Optional[Path] = None,
    ) -> None:
        self.session_name = session_name
        self.worker_count = worker_count
        self.monitor = monitor
        self.root_path = Path(root_path)
        self.boss_path = self.root_path
        self.startup_delay = startup_delay
        self.backtrack_delay = backtrack_delay
        self.reuse_existing_session = reuse_existing_session
        self.session_namespace = session_namespace
        self.codex_home = Path(codex_home) if codex_home else None
        self._server = libtmux.Server()

    def set_boss_path(self, path: Path) -> None:
        self.boss_path = Path(path)

    def set_reuse_existing_session(self, reuse: bool) -> None:
        self.reuse_existing_session = reuse

    def ensure_layout(self, *, session_name: str, worker_count: int) -> Dict[str, Any]:
        if session_name != self.session_name or worker_count != self.worker_count:
            raise ValueError("session_name/worker_count mismatch with manager configuration")

        session = self._get_or_create_session(fresh=not self.reuse_existing_session)
        window = getattr(session, "attached_window", None) or session.windows[0]

        target_pane_count = self.worker_count + 2  # main + boss + workers
        while len(window.panes) < target_pane_count:
            self._split_largest_pane(window)
            window.select_layout("tiled")
            window = getattr(session, "attached_window", None) or session.windows[0]
        window.select_layout("tiled")

        panes = window.panes
        layout = {
            "main": panes[0].pane_id,
            "boss": panes[1].pane_id,
            "workers": [pane.pane_id for pane in panes[2 : 2 + self.worker_count]],
        }
        self._apply_role_labels(session, layout)
        return layout

    def _apply_role_labels(self, session, layout: Mapping[str, Any]) -> None:
        try:
            self._set_pane_title(session, layout.get("main"), "MAIN")
            self._set_pane_title(session, layout.get("boss"), "BOSS")
            for index, pane_id in enumerate(layout.get("workers", []), start=1):
                self._set_pane_title(session, pane_id, f"WORKER-{index}")
        except Exception:  # pragma: no cover - defensive fallback for tmux failures
            pass

    def _set_pane_title(self, session, pane_id: Optional[str], title: str) -> None:
        if not pane_id:
            return
        try:
            session.cmd("select-pane", "-t", pane_id, "-T", title)
        except Exception:  # pragma: no cover - tmux may be older than 3.2
            pass

    def _split_largest_pane(self, window) -> None:
        panes = list(window.panes)
        if not panes:
            window.split_window(attach=False)
            return
        largest = max(panes, key=lambda pane: int(pane.height) * int(pane.width))
        window.select_pane(largest.pane_id)
        window.split_window(attach=False)

    def launch_main_session(self, *, pane_id: str) -> None:
        codex = self._codex_command("codex")
        command = (
            f"cd {shlex.quote(str(self.root_path))} && "
            f"{codex}"
        )
        self._send_command(pane_id, command)
        self._maybe_wait()

    def launch_boss_session(self, *, pane_id: str) -> None:
        codex = self._codex_command("codex")
        command = (
            f"cd {shlex.quote(str(self.boss_path))} && "
            f"{codex}"
        )
        self._send_command(pane_id, command)
        self._maybe_wait()

    def resume_session(self, *, pane_id: str, workdir: Path, session_id: str) -> None:
        codex = self._codex_command(f"codex resume {shlex.quote(str(session_id))}")
        command = (
            f"cd {shlex.quote(str(workdir))} && "
            f"{codex}"
        )
        self._send_command(pane_id, command)
        self._maybe_wait()

    def fork_boss(self, *, pane_id: str, base_session_id: str, boss_path: Path) -> None:
        self.interrupt_pane(pane_id=pane_id)
        command = (
            f"cd {shlex.quote(str(boss_path))} && "
            f"{self._codex_command(f'codex resume {shlex.quote(str(base_session_id))}')}"
        )
        self._send_command(pane_id, command)
        self._maybe_wait()
        pane = self._get_pane(pane_id)
        pane.send_keys("C-[", enter=False)
        time.sleep(self.backtrack_delay)
        pane.send_keys("C-[", enter=False)
        time.sleep(self.backtrack_delay)
        pane.send_keys("", enter=True)
        time.sleep(self.backtrack_delay)
        self._maybe_wait()

    def fork_workers(
        self,
        *,
        workers: Iterable[str],
        base_session_id: str,
        pane_paths: Mapping[str, Path],
    ) -> List[str]:
        if not base_session_id:
            raise RuntimeError("base_session_id が空です。メインセッションのIDが取得できていません。")
        worker_list = list(workers)
        for pane_id in worker_list:
            try:
                worker_path = Path(pane_paths[pane_id])
            except KeyError as exc:
                raise RuntimeError(
                    f"pane {pane_id!r} に対応するワークツリーパスがありません"
                ) from exc
            self.interrupt_pane(pane_id=pane_id)
            command = (
                f"cd {shlex.quote(str(worker_path))} && "
                f"{self._codex_command(f'codex resume {shlex.quote(str(base_session_id))}')}"
            )
            entry = {
                "timestamp": time.time(),
                "pane_id": pane_id,
                "worker_path": str(worker_path),
                "command": command,
            }
            try:
                debug_log = self.root_path / "logs" / "fork_debug.jsonl"
                debug_log.parent.mkdir(parents=True, exist_ok=True)
                with debug_log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass
            self._send_command(pane_id, command)
        self._maybe_wait()
        if worker_list:
            self._broadcast_keys(worker_list, "C-[", enter=False)
            self._broadcast_keys(worker_list, "C-[", enter=False)
        self._maybe_wait()
        return worker_list

    def send_instruction_to_pane(self, *, pane_id: str, instruction: str) -> None:
        self._send_text(pane_id, instruction)

    def send_instruction_to_workers(self, fork_map: Mapping[str, str], instruction: str) -> None:
        for pane_id in fork_map:
            self._send_text(pane_id, instruction)

    def prepare_for_instruction(self, *, pane_id: str) -> None:
        pane = self._get_pane(pane_id)
        pane.send_keys("C-c", enter=False)
        if self.backtrack_delay > 0:
            time.sleep(self.backtrack_delay)

    def promote_to_main(self, *, session_id: str, pane_id: str) -> None:
        pane = self._get_pane(pane_id)
        pane.send_keys(self._codex_command(f"codex resume {shlex.quote(str(session_id))}"), enter=True)

    def interrupt_pane(self, *, pane_id: str) -> None:
        pane = self._get_pane(pane_id)
        pane.send_keys("C-c", enter=False)
        if self.backtrack_delay > 0:
            time.sleep(self.backtrack_delay)
        pane.send_keys("C-c", enter=False)
        if self.backtrack_delay > 0:
            time.sleep(self.backtrack_delay)

    def _get_or_create_session(self, fresh: bool = False):
        session = self._server.find_where({"session_name": self.session_name})
        if session is not None and not fresh:
            self._configure_session(session)
            return session

        kill_existing = fresh and session is not None
        session = self._server.new_session(
            session_name=self.session_name,
            attach=False,
            kill_session=kill_existing,
        )
        self._configure_session(session)
        return session

    def _get_pane(self, pane_id: str):
        pane = self._find_pane(pane_id)
        if pane is not None:
            return pane
        self._server = libtmux.Server()
        pane = self._find_pane(pane_id)
        if pane is not None:
            return pane
        raise RuntimeError(f"Pane {pane_id!r} not found in tmux session {self.session_name}")

    def _find_pane(self, pane_id: str):
        for session in getattr(self._server, "sessions", []):
            for window in getattr(session, "windows", []):
                for pane in getattr(window, "panes", []):
                    if getattr(pane, "pane_id", None) == pane_id:
                        return pane
        return None

    def _send_command(self, pane_id: str, command: str) -> None:
        pane = self._get_pane(pane_id)
        pane.send_keys(command, enter=True)

    def _maybe_wait(self) -> None:
        if self.startup_delay > 0:
            time.sleep(self.startup_delay)

    def _send_text(self, pane_id: str, text: str) -> None:
        pane = self._get_pane(pane_id)
        payload = text.replace("\r\n", "\n")
        pane.send_keys(f"\x1b[200~{payload}\x1b[201~", enter=True)

    def _codex_command(self, command: str) -> str:
        if self.codex_home:
            home = shlex.quote(str(self.codex_home))
            codex_home = shlex.quote(str(self.codex_home / ".codex"))
            return f"env HOME={home} CODEX_HOME={codex_home} {command}"
        return command

    def _broadcast_keys(self, pane_ids: Sequence[str], key: str, *, enter: bool) -> None:
        for pane_id in pane_ids:
            pane = self._get_pane(pane_id)
            pane.send_keys(key, enter=enter)
        if self.backtrack_delay > 0:
            time.sleep(self.backtrack_delay)

    def _configure_session(self, session) -> None:
        commands = [
            ("set-option", "-g", "mouse", "on"),
            ("set-option", "-g", "pane-border-style", "fg=green"),
            ("set-option", "-g", "pane-active-border-style", "fg=orange"),
            ("set-option", "-g", "pane-border-status", "top"),
            ("set-option", "-g", "pane-border-format", "#{pane_title}"),
            ("set-option", "-g", "display-panes-colour", "green"),
            ("set-option", "-g", "display-panes-active-colour", "orange"),
        ]
        for args in commands:
            try:
                session.cmd(*args)
            except Exception:  # pragma: no cover - 一部オプション非対応のtmux向けフォールバック
                continue


class WorktreeManager:
    """Manage git worktrees for each worker."""

    def __init__(self, root: Path, worker_count: int, session_namespace: Optional[str] = None) -> None:
        self.root = Path(root)
        self.worker_count = worker_count
        self.session_namespace = session_namespace
        self._repo = git.Repo(self.root)
        self._ensure_repo_initialized()
        self._session_root = self._resolve_session_root()
        self.worktrees_dir = self._session_root / "worktrees"
        self.boss_path = self.worktrees_dir / "boss"
        self._worker_branch_template, self._boss_branch = self._resolve_branch_templates()
        self._initialized = False
        self._worker_paths: Dict[str, Path] = {}

    def prepare(self) -> Dict[str, Path]:
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        mapping: Dict[str, Path] = {}
        for index in range(1, self.worker_count + 1):
            worker_name = f"worker-{index}"
            worktree_path = self.worktrees_dir / worker_name
            branch_name = self.worker_branch(worker_name)
            if not self._initialized or worker_name not in self._worker_paths:
                self._recreate_worktree(worktree_path, branch_name)
            else:
                self._reset_worktree(worktree_path)
            mapping[worker_name] = worktree_path
        if self._initialized:
            existing = set(self._worker_paths)
            target = set(mapping)
            for obsolete in existing - target:
                self._remove_worktree(self.worktrees_dir / obsolete)

        if not self._initialized:
            self._recreate_worktree(self.boss_path, self.boss_branch)
        else:
            self._reset_worktree(self.boss_path)
        self._worker_paths = mapping
        self._initialized = True
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

    def _reset_worktree(self, path: Path) -> None:
        if not path.exists():
            return
        repo = git.Repo(path)
        repo.git.reset("--hard", "HEAD")
        repo.git.clean("-fdx")

    def _remove_worktree(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            self._repo.git.worktree("remove", "--force", str(path))
        except git.GitCommandError:
            shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def worker_branch(self, worker_name: str) -> str:
        return self._worker_branch_template.format(name=worker_name)

    @property
    def boss_branch(self) -> str:
        return self._boss_branch

    def merge_into_main(self, branch_name: str) -> None:
        if self._repo.is_dirty(untracked_files=False):
            raise RuntimeError("Main repository has uncommitted changes; cannot merge results.")
        self._repo.git.merge(branch_name, "--ff-only")

    def dispose(self) -> None:
        if not self.session_namespace:
            return
        if self._session_root.exists():
            shutil.rmtree(self._session_root, ignore_errors=True)
        self._initialized = False
        self._worker_paths = {}

    def _resolve_session_root(self) -> Path:
        base = self.root / ".parallel-dev"
        if self.session_namespace:
            return base / "sessions" / self.session_namespace
        return base

    def _resolve_branch_templates(self) -> tuple[str, str]:
        if self.session_namespace:
            prefix = f"parallel-dev/{self.session_namespace}"
            return f"{prefix}/{{name}}", f"{prefix}/boss"
        return "parallel-dev/{name}", "parallel-dev/boss"


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
        self._forced_done: Set[str] = set()

    def register_session(self, *, pane_id: str, session_id: str, rollout_path: Path) -> None:
        data = self._load_map()
        panes = data.setdefault("panes", {})
        sessions = data.setdefault("sessions", {})
        offset = 0
        try:
            offset = rollout_path.stat().st_size
        except OSError:
            offset = 0
        panes[pane_id] = {
            "session_id": session_id,
            "rollout_path": str(rollout_path),
            "offset": int(offset),
        }
        sessions[session_id] = {
            "pane_id": pane_id,
            "rollout_path": str(rollout_path),
            "offset": int(offset),
        }
        self._write_map(data)

    def bind_existing_session(self, *, pane_id: str, session_id: str) -> None:
        data = self._load_map()
        sessions = data.setdefault("sessions", {})
        entry = sessions.get(session_id)
        if entry is None:
            raise RuntimeError(f"Session {session_id!r} not found in session_map")

        rollout_path = Path(entry.get("rollout_path", ""))
        try:
            offset = rollout_path.stat().st_size
        except OSError:
            offset = int(entry.get("offset", 0))

        entry["pane_id"] = pane_id
        entry["offset"] = int(offset)

        panes = data.setdefault("panes", {})
        for existing_pane, pane_entry in list(panes.items()):
            if existing_pane == pane_id or pane_entry.get("session_id") == session_id:
                panes.pop(existing_pane, None)

        panes[pane_id] = {
            "session_id": session_id,
            "rollout_path": entry["rollout_path"],
            "offset": int(offset),
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
        if len(paths) < len(worker_panes):
            raise TimeoutError(
                f"Detected {len(paths)} worker rollouts but {len(worker_panes)} required."
            )
        fork_map: Dict[str, str] = {}
        for pane_id, path in zip(worker_panes, paths):
            session_id = self._parse_session_meta(path)
            self.register_session(pane_id=pane_id, session_id=session_id, rollout_path=path)
            fork_map[pane_id] = session_id
        return fork_map

    def get_last_assistant_message(self, session_id: str) -> Optional[str]:
        data = self._load_map()
        sessions = data.get("sessions", {})
        entry = sessions.get(session_id)
        if entry is None:
            return None

        rollout_path = Path(entry.get("rollout_path", ""))
        if not rollout_path.exists():
            return None

        last_text: Optional[str] = None
        try:
            with rollout_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if obj.get("type") != "response_item":
                        continue

                    payload = obj.get("payload", {})
                    if payload.get("role") != "assistant":
                        continue

                    texts: List[str] = []
                    for block in payload.get("content", []):
                        block_type = block.get("type")
                        if block_type in {"output_text", "text"}:
                            texts.append(block.get("text", ""))
                        elif block_type == "output_markdown":
                            texts.append(block.get("markdown", ""))
                        elif block_type == "output_json":
                            data = block.get("json")
                            if data is not None:
                                texts.append(json.dumps(data))
                    if texts:
                        last_text = "\n".join(part for part in texts if part).strip()
        except OSError:
            return None

        return last_text

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
        offsets: Dict[str, int] = {}
        for session_id in session_ids:
            entry = sessions.get(session_id)
            if entry is None:
                raise RuntimeError(f"Session {session_id!r} not found in session_map")
            targets[session_id] = Path(entry["rollout_path"])
            offsets[session_id] = int(entry.get("offset", 0))

        remaining = set(targets)
        completion: Dict[str, Any] = {}

        def consume_forced() -> None:
            forced_now = remaining.intersection(self._forced_done)
            for session_id in list(forced_now):
                path = targets[session_id]
                try:
                    offset = path.stat().st_size
                except OSError:
                    offset = 0
                offsets[session_id] = offset
                self._update_session_offset(session_id, offset)
                completion[session_id] = {
                    "done": True,
                    "rollout_path": str(path),
                    "forced": True,
                }
                remaining.discard(session_id)
                self._forced_done.discard(session_id)

        consume_forced()
        deadline = None if timeout_seconds is None else time.time() + timeout_seconds

        while remaining:
            consume_forced()
            if not remaining:
                break
            for session_id in list(remaining):
                rollout_path = targets[session_id]
                done, new_offset = self._contains_done(
                    session_id=session_id,
                    rollout_path=rollout_path,
                    offset=offsets.get(session_id, 0),
                )
                if new_offset != offsets.get(session_id, 0):
                    offsets[session_id] = new_offset
                    self._update_session_offset(session_id, new_offset)
                if done:
                    completion[session_id] = {"done": True, "rollout_path": str(rollout_path)}
                    remaining.remove(session_id)
            if not remaining:
                break
            if deadline is not None and time.time() >= deadline:
                break
            time.sleep(self.poll_interval)

        for session_id in remaining:
            completion[session_id] = {
                "done": False,
                "rollout_path": str(targets[session_id]),
            }

        return completion

    def force_completion(self, session_ids: Iterable[str]) -> None:
        for session_id in session_ids:
            if session_id:
                self._forced_done.add(session_id)

    def wait_for_rollout_activity(
        self,
        session_id: str,
        *,
        min_bytes: int = 1,
        timeout_seconds: float = 5.0,
    ) -> None:
        data = self._load_map()
        sessions = data.get("sessions", {})
        entry = sessions.get(session_id)
        if entry is None:
            return
        rollout_path = Path(entry.get("rollout_path", ""))
        baseline = int(entry.get("offset", 0))
        deadline = time.time() + timeout_seconds
        last_size = baseline

        while time.time() < deadline:
            try:
                size = rollout_path.stat().st_size
            except OSError:
                size = last_size

            if size >= baseline + min_bytes:
                self._update_session_offset(session_id, size)
                return

            last_size = size
            time.sleep(self.poll_interval)

        if last_size > baseline:
            self._update_session_offset(session_id, last_size)

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

    def _contains_done(
        self,
        *,
        session_id: str,
        rollout_path: Path,
        offset: int,
    ) -> tuple[bool, int]:
        if not rollout_path.exists():
            return False, offset
        try:
            with rollout_path.open("rb") as fh:
                fh.seek(offset)
                chunk = fh.read()
                new_offset = fh.tell()
        except OSError:
            return False, offset

        if not chunk:
            return False, new_offset

        done_detected = False
        for line in chunk.decode("utf-8", errors="ignore").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") != "response_item":
                continue

            payload = obj.get("payload", {})
            if payload.get("role") != "assistant":
                continue

            for block in payload.get("content", []):
                block_type = block.get("type")
                if block_type in {"output_text", "text"}:
                    text = block.get("text", "")
                    lines = [segment.strip() for segment in text.splitlines() if segment.strip()]
                    if any(segment == "/done" for segment in lines):
                        done_detected = True
                        break
                    for segment in lines:
                        try:
                            maybe_json = json.loads(segment)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(maybe_json, dict) and "scores" in maybe_json:
                            done_detected = True
                            break
                    if done_detected:
                        break
                elif block_type == "output_json":
                    data = block.get("json")
                    if isinstance(data, dict) and "scores" in data:
                        done_detected = True
                        break
            if done_detected:
                break

        return done_detected, new_offset

    def _update_session_offset(self, session_id: str, new_offset: int) -> None:
        data = self._load_map()
        sessions = data.get("sessions", {})
        panes = data.get("panes", {})
        session_entry = sessions.get(session_id)
        if session_entry is not None:
            session_entry["offset"] = int(new_offset)
            pane_id = session_entry.get("pane_id")
            if pane_id and pane_id in panes:
                panes[pane_id]["offset"] = int(new_offset)
            self._write_map(data)


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
    ) -> Dict[str, Path]:
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
        jsonl_path = self.cycles_dir / f"{timestamp}.jsonl"
        events = self._build_cycle_events(
            instruction=instruction,
            layout=layout,
            fork_map=fork_map,
            completion=completion,
            result=result,
        )
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        return {"yaml": path, "jsonl": jsonl_path}

    def _build_cycle_events(
        self,
        *,
        instruction: str,
        layout: Mapping[str, Any],
        fork_map: Mapping[str, str],
        completion: Mapping[str, Any],
        result: Any,
    ) -> List[Dict[str, Any]]:
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        events: List[Dict[str, Any]] = []

        events.append(
            {
                "type": "instruction",
                "timestamp": timestamp,
                "instruction": instruction,
                "layout": dict(layout),
            }
        )

        if fork_map:
            events.append(
                {
                    "type": "fork",
                    "timestamp": timestamp,
                    "fork_map": dict(fork_map),
                }
            )

        if completion:
            events.append(
                {
                    "type": "completion",
                    "timestamp": timestamp,
                    "completion": dict(completion),
                }
            )

        scoreboard = getattr(result, "sessions_summary", None) or {}
        if scoreboard:
            events.append(
                {
                    "type": "scoreboard",
                    "timestamp": timestamp,
                    "scoreboard": scoreboard,
                }
            )

        selected_session = getattr(result, "selected_session", None)
        selected_key = None
        for key, data in scoreboard.items():
            if data.get("selected"):
                selected_key = key
                break
        events.append(
            {
                "type": "selection",
                "timestamp": timestamp,
                "selected_session": selected_session,
                "selected_key": selected_key,
            }
        )

        artifact = getattr(result, "artifact", None)
        if artifact is not None:
            events.append(
                {
                    "type": "artifact",
                    "timestamp": timestamp,
                    "main_session_id": getattr(artifact, "main_session_id", None),
                    "worker_sessions": getattr(artifact, "worker_sessions", {}),
                    "boss_session_id": getattr(artifact, "boss_session_id", None),
                    "worker_paths": {k: str(v) for k, v in getattr(artifact, "worker_paths", {}).items()},
                    "boss_path": str(getattr(artifact, "boss_path", "")) if getattr(artifact, "boss_path", None) else None,
                }
            )

        return events
