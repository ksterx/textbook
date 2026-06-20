# 体系学習テキスト

音声 (Audio)・強化学習 (Reinforcement Learning)・LLM・Physical AI などを、**一から体系的に**学ぶための技術テキスト。
[Zensical](https://zensical.org)（Material for MkDocs チームによる静的サイトジェネレータ）で構築しています。

すべての分野を **同じ構成・同じ品質** で書くことを重視しており、章のフォーマットは統一されています。
分野は今後も増えていく前提で設計しています。

## セットアップ

[uv](https://docs.astral.sh/uv/) を使います。

```bash
uv sync                  # 依存（zensical）をインストール
```

## ローカルで見る・ビルドする

```bash
uv run zensical serve    # ローカルプレビュー → http://localhost:8000
uv run zensical build    # 静的サイトを site/ に生成
```

## ディレクトリ構成

```text
.
├── zensical.toml                 # サイト設定（テーマ・ナビゲーション・数式など）
├── pyproject.toml                # uv プロジェクト（依存: zensical）
├── AUTHORING.md                  # 執筆ガイド（品質基準・トーン・数式/コード規約）★最重要
├── templates/                    # 章・分野ランディングの雛形
│   ├── chapter-template.md
│   └── domain-index-template.md
├── docs/                         # 本文（Markdown）
│   ├── index.md                  #   トップ（全体マップ）
│   ├── javascripts/mathjax.js    #   数式レンダリング設定（全ページ共通）
│   ├── stylesheets/extra.css     #   共通スタイル
│   ├── audio/                    #   分野: Audio
│   │   ├── index.md              #     ロードマップ
│   │   └── 01-digital-audio-basics.md
│   ├── reinforcement-learning/
│   ├── llm/
│   └── physical-ai/
└── .claude/skills/textbook/      # 章・分野を統一フォーマットで生成する Claude Code スキル
```

## 章を追加する / 分野を追加する

Claude Code に「次は◯◯の章を書いて」「△△という分野を追加して」と指示すると、
[`AUTHORING.md`](AUTHORING.md) の規約と `templates/` の雛形に従って、統一フォーマットで生成されます。

手で追加する場合の手順も [`AUTHORING.md`](AUTHORING.md) の §10・§11 にまとめています。

## 各章の構成（フルセット）

学習目標 → 前提知識 → 直感 → 全体像 → 理論 → 数式の導出 → 実装 → 演習 → まとめ → 用語ミニ辞典 → 次のアクション → 参考文献

品質の心臓部は [`AUTHORING.md`](AUTHORING.md) の **§2「説明の深さ基準」**。お手本は [`docs/audio/01-digital-audio-basics.md`](docs/audio/01-digital-audio-basics.md)（Mermaid・Canvas 図・数式・実測コードまで含む完成章）。
