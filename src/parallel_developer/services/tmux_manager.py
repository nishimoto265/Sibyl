"""tmux レイアウトとキーストロークを制御するサービス."""

from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, TYPE_CHECKING, List

import libtmux

if TYPE_CHECKING:  # pragma: no cover - 型チェック専用
    from .codex_monitor import CodexMonitor


class TmuxLayoutManager:
    """Parallel Codex 用の tmux セッションを構成・制御する."""

    def __init__(
        self,
        session_name: str,
        worker_count: int,
        monitor: "CodexMonitor",
        *,
        root_path: Path,
        startup_delay: float = 0.0,
        backtrack_delay: float = 0.05,
        reuse_existing_session: bool = False,
        session_namespace: Optional[str] = None,
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
        self._server = libtmux.Server()

    def set_boss_path(self, path: Path) -> None:
        self.boss_path = Path(path)

    def set_reuse_existing_session(self, reuse: bool) -> None:
        self.reuse_existing_session = reuse

    def ensure_layout(self, *, session_name: str, worker_count: int) -> dict[str, Any]:
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

    def launch_main_session(self, *, pane_id: str) -> None:
        codex = self._codex_command("codex")
        command = f"cd {shlex.quote(str(self.root_path))} && {codex}"
        self._send_command(pane_id, command)
        self._maybe_wait()

    def resume_session(self, *, pane_id: str, workdir: Path, session_id: str) -> None:
        codex = self._codex_command(f"codex resume {shlex.quote(str(session_id))}")
        command = f"cd {shlex.quote(str(workdir))} && {codex}"
        self._send_command(pane_id, command)
        self._maybe_wait()

    def fork_boss(self, *, pane_id: str, base_session_id: str, boss_path: Path) -> None:
        self.interrupt_pane(pane_id=pane_id)
        command = f"cd {shlex.quote(str(boss_path))} && {self._codex_command(f'codex resume {shlex.quote(str(base_session_id))}')}"
        self._send_command(pane_id, command)
        self._maybe_wait()
        self._send_prompt_reset(pane_id=pane_id)
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
                raise RuntimeError(f"pane {pane_id!r} に対応するワークツリーパスがありません") from exc
            self.interrupt_pane(pane_id=pane_id)
            command = f"cd {shlex.quote(str(worker_path))} && {self._codex_command(f'codex resume {shlex.quote(str(base_session_id))}')}"
            self._send_command(pane_id, command)
            time.sleep(max(0.5, self.backtrack_delay))
            self._send_prompt_reset(pane_id=pane_id)
        self._maybe_wait()
        return worker_list

    def send_instruction_to_pane(self, *, pane_id: str, instruction: str) -> None:
        self._send_text(pane_id, instruction)

    def prepare_for_instruction(self, *, pane_id: str) -> None:
        pane = self._get_pane(pane_id)
        pane.send_keys("C-c", enter=False)
        if self.backtrack_delay > 0:
            time.sleep(self.backtrack_delay)

    def promote_to_main(self, *, session_id: str, pane_id: str) -> None:
        command = self._codex_command(f"codex resume {shlex.quote(str(session_id))}")
        if self.backtrack_delay > 0:
            time.sleep(self.backtrack_delay)
        self._send_command(pane_id, command)

    def interrupt_pane(self, *, pane_id: str) -> None:
        pane = self._get_pane(pane_id)
        pane.send_keys("C-c", enter=False)
        if self.backtrack_delay > 0:
            time.sleep(self.backtrack_delay)
        pane.send_keys("C-c", enter=False)
        if self.backtrack_delay > 0:
            time.sleep(self.backtrack_delay)

    def _apply_role_labels(self, session, layout: Mapping[str, Any]) -> None:
        try:
            self._set_pane_title(session, layout.get("main"), "MAIN")
            self._set_pane_title(session, layout.get("boss"), "BOSS")
            for index, pane_id in enumerate(layout.get("workers", []), start=1):
                self._set_pane_title(session, pane_id, f"WORKER-{index}")
        except Exception:  # pragma: no cover - tmux互換対策
            pass

    def _set_pane_title(self, session, pane_id: Optional[str], title: str) -> None:
        if not pane_id:
            return
        try:
            session.cmd("select-pane", "-t", pane_id, "-T", title)
        except Exception:  # pragma: no cover - 古いtmux向け
            pass

    def _split_largest_pane(self, window) -> None:
        panes = list(window.panes)
        if not panes:
            window.split_window(attach=False)
            return
        largest = max(panes, key=lambda pane: int(pane.height) * int(pane.width))
        window.select_pane(largest.pane_id)
        window.split_window(attach=False)

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
        return command

    def _send_prompt_reset(self, *, pane_id: str) -> None:
        pane = self._get_pane(pane_id)
        pane.send_keys("C-[", enter=False)
        time.sleep(max(0.1, self.backtrack_delay))
        pane.send_keys("C-[", enter=False)
        time.sleep(max(0.1, self.backtrack_delay))
        pane.send_keys("", enter=True)
        time.sleep(max(0.1, self.backtrack_delay))

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
