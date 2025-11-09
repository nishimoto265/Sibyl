"""Git worktree を管理するサービス."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Optional

import git


class WorktreeManager:
    """ワーカー/ボス用の git worktree を生成・整備する."""

    def __init__(
        self,
        root: Path,
        worker_count: int,
        session_namespace: Optional[str] = None,
        storage_root: Optional[Path] = None,
    ) -> None:
        self.root = Path(root)
        self.worker_count = worker_count
        self.session_namespace = session_namespace
        self._repo = git.Repo(self.root)
        self._ensure_repo_initialized()
        self._storage_root = Path(storage_root) if storage_root is not None else self.root
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

    def worker_branch(self, worker_name: str) -> str:
        return self._worker_branch_template.format(name=worker_name)

    @property
    def boss_branch(self) -> str:
        return self._boss_branch

    def merge_into_main(self, branch_name: str) -> None:
        if self._repo.is_dirty(untracked_files=False):
            raise RuntimeError("Main repository has uncommitted changes; cannot merge results.")
        self._repo.git.merge(branch_name, "--ff-only")

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

    def _resolve_session_root(self) -> Path:
        base = self._storage_root / ".parallel-dev"
        if self.session_namespace:
            return base / "sessions" / self.session_namespace
        return base

    def _resolve_branch_templates(self) -> tuple[str, str]:
        if self.session_namespace:
            prefix = f"parallel-dev/{self.session_namespace}"
            return f"{prefix}/{{name}}", f"{prefix}/boss"
        return "parallel-dev/{name}", "parallel-dev/boss"
