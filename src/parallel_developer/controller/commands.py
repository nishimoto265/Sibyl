"""Controller command definitions and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional


@dataclass
class CommandSuggestion:
    name: str
    description: str


@dataclass
class CommandOption:
    label: str
    value: object
    description: Optional[str] = None
    display: Optional[str] = None


@dataclass
class CommandSpecEntry:
    description: str
    handler: Callable[[Optional[object]], Awaitable[None]]
    options: Optional[List[CommandOption]] = None
    options_provider: Optional[Callable[[], List[CommandOption]]] = None


def build_command_specs(
    controller,
    *,
    flow_mode_cls,
    boss_mode_cls,
    merge_mode_cls,
) -> Dict[str, CommandSpecEntry]:
    FlowMode = flow_mode_cls  # alias for brevity
    MergeMode = merge_mode_cls
    specs = {
        "/attach": CommandSpecEntry(
            "tmux接続の方法を切り替える",
            controller._cmd_attach,
            options=[
                CommandOption("auto - 自動でターミナルを開く", "auto", "自動でターミナルを開く"),
                CommandOption("manual - コマンド入力で手動接続", "manual", "コマンド入力で手動接続"),
                CommandOption("now - 即座にtmux attachを実行", "now", "即座にtmux attachを実行"),
            ],
        ),
        "/boss": CommandSpecEntry(
            "Bossの挙動を切り替える",
            controller._cmd_boss,
            options=[
                CommandOption("skip - Boss評価をスキップ", boss_mode_cls.SKIP.value, "Boss評価をスキップ"),
                CommandOption("score - 採点のみ実施", boss_mode_cls.SCORE.value, "採点のみ実施"),
                CommandOption("rewrite - 採点後にBossが統合実装", boss_mode_cls.REWRITE.value, "採点後にBossが統合実装"),
            ],
        ),
        "/flow": CommandSpecEntry(
            "フロー自動化レベルを切り替える",
            controller._cmd_flow,
            options=[
                CommandOption("manual - 採点段階への移行や採択を手動で行う", FlowMode.MANUAL.value, "採点段階への移行や採択を手動で行う"),
                CommandOption("auto_review - 採点段階への移行は自動、採択は手動", FlowMode.AUTO_REVIEW.value, "採点段階への移行は自動、採択は手動"),
                CommandOption("auto_select - 採点段階への移行は手動、採択は自動", FlowMode.AUTO_SELECT.value, "採点段階への移行は手動、採択は自動"),
                CommandOption("full_auto - 採点段階への移行・採択まで自動", FlowMode.FULL_AUTO.value, "採点段階への移行・採択まで自動"),
            ],
        ),
        "/merge": CommandSpecEntry(
            "マージ方式を切り替える",
            controller._cmd_merge,
            options=[
                CommandOption(
                    "manual - ホストが従来どおりマージ",
                    MergeMode.MANUAL.value,
                    "ホスト側がFast-Forwardなどを実行",
                ),
                CommandOption(
                    "auto - エージェントがコミット・統合作業",
                    MergeMode.AUTO.value,
                    "採択エージェントにコミット/統合手順を送る",
                ),
            ],
        ),
        "/parallel": CommandSpecEntry(
            "ワーカー数を設定する",
            controller._cmd_parallel,
            options=[CommandOption(f"{n} - ワーカーを{n}人起動", str(n), f"ワーカーを{n}人起動") for n in range(1, 5)],
        ),
        "/mode": CommandSpecEntry(
            "実行対象を切り替える",
            controller._cmd_mode,
            options=[
                CommandOption("main - メインCodexのみ稼働", "main", "メインCodexのみ稼働"),
                CommandOption("parallel - メイン+ワーカーを起動", "parallel", "メイン+ワーカーを起動"),
            ],
        ),
        "/resume": CommandSpecEntry(
            "保存セッションを再開する",
            controller._cmd_resume,
            options_provider=controller._build_resume_options,
        ),
        "/continue": CommandSpecEntry("ワーカーの作業を続行する", controller._cmd_continue),
        "/log": CommandSpecEntry(
            "ログをコピーや保存する",
            controller._cmd_log,
            options=[
                CommandOption("copy - ログをクリップボードへコピー", "copy", "ログをクリップボードへコピー"),
                CommandOption("save - ログをファイルへ保存", "save", "ログをファイルへ保存"),
            ],
        ),
        "/commit": CommandSpecEntry(
            "Gitコミットを操作する",
            controller._cmd_commit,
            options=[
                CommandOption("manual - 現在の変更をコミット", "manual", "現在の変更をその場でコミット"),
                CommandOption("auto - 自動コミットをON/OFF", "auto", "サイクル開始時に自動コミットをON/OFF"),
            ],
        ),
        "/status": CommandSpecEntry(
            "現在の状態を表示する",
            controller._cmd_status,
        ),
        "/scoreboard": CommandSpecEntry(
            "最新スコアを表示する",
            controller._cmd_scoreboard,
        ),
        "/done": CommandSpecEntry("採点フェーズへ移行する", controller._cmd_done),
        "/help": CommandSpecEntry("コマンド一覧を表示する", controller._cmd_help),
        "/exit": CommandSpecEntry("CLI を終了する", controller._cmd_exit),
    }
    return specs
