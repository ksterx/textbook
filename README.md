# Textbook

音声 (Audio)・強化学習 (Reinforcement Learning)・LLM・Physical AI などを、**一から体系的に**学ぶための技術テキスト。
**[Astro](https://astro.build)** で構築し、GitHub Pages に公開しています。

公開URL: **https://ksterx.github.io/textbook/**

すべての分野を **同じ構成・同じ品質** で書くことを重視しており、章のフォーマットは統一されています。分野は今後も増やしていく前提で設計しています。

## セットアップ

[Node.js](https://nodejs.org)（18.20.8+ / 20.3+ / 22+）が必要です。

```bash
npm install
```

## ローカルで見る・ビルドする

```bash
npm run dev      # 開発サーバ（http://localhost:4321/textbook/）
npm run build    # 静的サイトを dist/ に生成（Astro ビルド + Pagefind 索引）
npm run preview  # ビルド結果をプレビュー（検索も有効）
```

## 技術スタック

- **Astro** — 自前レイアウト（ヘッダ・左ナビ・右TOC）。テーマと格闘せず、デザインを完全制御。
- **KaTeX** — 数式を**ビルド時に静的 HTML へレンダー**（実行時 JS・外部 CDN なし）。
- **Pagefind** — 静的全文検索（サーバ不要）。
- **mermaid** — 図をローカルバンドルで描画（CDN なし）。
- **Canvas** — 領域固有の図（波形・格子・ヒートマップ等）を `public/figures.js` に集約。
- 自作 remark/rehype プラグイン — `:::` ディレクティブ→callout、```mermaid 変換、base パス付与。

## ディレクトリ構成

```text
.
├── astro.config.mjs              # Astro 設定（remark/rehype プラグイン・base）
├── src/
│   ├── pages/                    # 本文（Markdown）。ファイル＝ルート
│   │   ├── index.md              #   トップ（全体マップ）
│   │   ├── audio/                #   分野: Audio（index + 01〜04）
│   │   ├── reinforcement-learning/
│   │   ├── llm/
│   │   └── physical-ai/
│   ├── layouts/Page.astro        # 全ページ共通レイアウト
│   ├── lib/nav.mjs               # ナビゲーション定義（章を足したら 1 行追加）
│   ├── plugins/                  # 自作 remark/rehype プラグイン
│   └── styles/global.css         # 全スタイル・デザイントークン
├── public/figures.js             # Canvas 図のレジストリ
├── templates/                    # 章・分野ランディングの雛形
├── AUTHORING.md                  # ★執筆ガイド（品質基準・規約・手順）
└── .github/workflows/docs.yml    # GitHub Pages へ自動デプロイ
```

## 章を追加する / 分野を追加する

Claude Code に「次は◯◯の章を書いて」「△△という分野を追加して」と指示すると、
[`AUTHORING.md`](AUTHORING.md) の規約と `templates/` の雛形に従って、統一フォーマットで生成されます。
手で追加する手順は [`AUTHORING.md`](AUTHORING.md) の §10・§11 を参照。

**このテキストは会話の中で育ちます。** 理解できず質問した箇所は、回答で終わらせず該当章の本文を厚くします（[`AUTHORING.md`](AUTHORING.md) §2「質問駆動で厚くする」）。

## 各章の構成（フルセット）

学習目標 → 前提知識 → 直感 → 全体像 → 理論 → 数式の導出 → 実装 → 演習 → まとめ → 用語ミニ辞典 → 次のアクション → 参考文献

品質の心臓部は [`AUTHORING.md`](AUTHORING.md) の **§2「説明の深さ基準」**。お手本は [`src/pages/audio/01-digital-audio-basics.md`](src/pages/audio/01-digital-audio-basics.md)。
