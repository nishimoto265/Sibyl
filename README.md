# Parallel Developer CLI

tmux と git worktree を土台に、複数の Codex/LLM エージェントを同じコンテキストで同時起動し、指示・比較・採点・統合までを 1 つの CLI から操作できるツールです。指示を 1 回入力するだけでワーカーが並列に実装し、採点エージェント（Boss）が結果を評価・統合することで、より良い実装案を自動的に得られます。

## 特徴
- **コンテキスト維持型マルチエージェント**: メインへ与えた指示が各ワーカーへ自動コピーされるため、毎回プロジェクト概要を説明し直す必要がありません。
- **柔軟な並列度**: `/parallel` コマンドでワーカー数を自由に増減可能。途中で人数を変えてもセッション履歴は保持されます。
- **比較＆自動採点**: 複数実装を並べて比較し、採点エージェントが自動でスコアリング。点数に応じてベスト案を選択できます。
- **採点後の統合能力**: Boss は高得点の断片を組み合わせたり、複数案を見た後に改めて実装することで、より洗練された最終成果物を生成します。
- **統一 CLI からの操作**: 指示送信、エージェント起動、採点、統合、再開までを Textual ベースの CLI 1 つで完結。
- **フルオートにも対応**: エージェント起動→指示送信→採点→マージ→再開の一連の流れを自動化（Flow/Boss/マージ戦略はコマンドで切り替え）。

## セットアップ
### もっとも簡単な使い方（PyPI）
```bash
pip install pdev
pdev
```

これだけでCLIが起動し、以降はどこからでも `pdev` コマンドを実行できます。

### 開発したい場合（ソースから）
```bash
git clone https://github.com/nishimoto265/PDev.git parallel-developer
cd parallel-developer

# 推奨: uv で仮想環境を作成
uv venv
uv sync

# CLI を実行
uv run pdev
```

**pip のみでローカル開発を行う場合**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pdev
```

## 使い方の流れ
1. `uv run pdev` を起動し、CLI で指示または `/command` を入力。
2. ワーカーが同じコンテキストを共有したまま並列実装。必要に応じて `/parallel <n>` で人数を増減。
3. Boss（採点エージェント）が自動でスコアリング。rewrite モードでは複数案を踏まえた実装も実施。
4. スコアに応じて候補を採択し、選ばれたセッションを `codex resume` でメインへリジューム。
5. `/flow full_auto` 等を使えば上記を全自動で回せます。

## 主なコマンド
| コマンド | 役割（概要） |
| --- | --- |
| `/attach [auto\|manual\|now]` | tmux へのアタッチ方法を切り替える |
| `/mode [main\|parallel]` | メインのみ or 並列モードを選ぶ |
| `/parallel <n>` | ワーカー数を設定（途中変更も可） |
| `/flow [manual\|auto_review\|auto_select\|full_auto]` | レビュー/採択の自動化レベルを決める |
| `/boss [skip\|score\|rewrite]` | 採点エージェントの役割を決める |
| `/merge` (`/marge`) | マージの進め方を選ぶ |
| `/continue` | 完了扱いのワーカーに追加指示を送る |
| `/done` | 採点フェーズへ進める |
| `/resume` | 保存済みセッションを再開する |
| `/log copy` | その場でログをコピーする |
| `/log save <path>` | ログをファイルに保存する |
| `/commit manual` | 今の変更を即コミットする |
| `/commit auto` | サイクル開始時の自動コミットを切り替える |
| `/status` | 現在の状態を表示する |
| `/scoreboard` | 直近の採点結果を表示する |
| `/help` | コマンド一覧を表示する |
| `/exit` | CLI を終了する |

以下では「何ができるか」に絞って簡潔にまとめます。設定系コマンドは実行すると `~/.parallel-dev/config.yaml` に保存され、次回も同じ振る舞いになります。

### `/attach [auto|manual|now]`
- `auto`: CLIが自動で tmux に張り付く。何もせず画面を確認したいとき向け。
- `manual`: 必要なときにだけ自分でアタッチ。バックグラウンドで回したいとき向け。
- `now`: その場でアタッチを実行。

### `/mode [main|parallel]`
- `main`: メインペインだけを使い、並列化を一時停止。
- `parallel`: メイン＋ワーカー＋Bossをすべて起動する通常運転。

### `/parallel <n>`
- 指示中のプロジェクトに、そのままワーカーを増減させたいときに使います（例: 2→4人に増員）。既存の文脈は保たれます。

### `/flow [manual|auto_review|auto_select|full_auto]`
- レビューや採択をどこまで自動化するかを段階的に切り替えます。例えば `auto_select` なら「採点開始の合図だけ手動、採択は自動」といった運用が可能です。

### `/boss [skip|score|rewrite]`
- Bossの役目を決めます。`skip`で完全省略、`score`で採点役、`rewrite`で「採点＋統合実装まで担当」という流れを用意できます。

### `/merge`（`/marge`）
- 自動マージの態度を決めます。衝突ゼロなら `fast_only`、常にエージェントへ任せたいなら `agent_only`、まずは自動マージを試したいなら `fast_then_agent`。

### `/continue` と `/done`
- `/continue`: 完了扱いになったワーカーを再び動かし、ユーザーが追加入力した内容をそのまま送ります。
- `/done`: 今の段階で採点フェーズへ進めたいときに押す合図。Workersがフラグを出さなくても前に進めます。

### `/resume`
- 過去のサイクルを一覧表示し、任意のセッションをそのまま再開できます。中断した作業を引き継ぐときに便利です。

### `/log copy|save`
- `copy`: 現在のログ全文を手元クリップボードへ取り込み、別エディタに貼りたいときに使います。
- `save <path>`: ログをファイルとして書き出し、共有や検証用の資料にできます。

### `/commit [manual|auto]`
- `manual`: 今ある変更をすぐ記録したいときのショートカット。
- `auto`: サイクル開始時に自動コミットを入れる/止める設定。毎周回でクリーンな差分にしたいときに便利。

### `/status`, `/scoreboard`, `/help`, `/exit`
- `/status`: 現在ステップ（待機・一時停止など）を確認。
- `/scoreboard`: 直近の採点結果を再掲。
- `/help`: コマンド一覧を表示。
- `/exit`: CLI を終了。
