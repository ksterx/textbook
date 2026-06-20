---
layout: ../layouts/Page.astro
title: Textbook — モダリティ地図
description: AI を「モダリティ（入出力の様式）」を主軸に体系的に学ぶ技術テキスト。言語・音声・視覚・身体性・マルチモーダル、横断軸として強化学習。
---

# Textbook

このテキストは、AI を **モダリティ（入出力の様式）を主軸**に体系的に学びます。
各モダリティを **同じ構成・同じ品質** で書き、まず「全体地図（分類）」を描いてから各論に降ります。

:::tip[このテキストの読み方]
- **モダリティで分類。** 「言語」「音声」「視覚」「身体性」「マルチモーダル」が縦軸。
- **手法は横断軸。** 強化学習などの学習パラダイムはモダリティに直交し、各モダリティから参照される。
- **わかりやすさ最優先。** まず直感、その後で理論・数式・実装。専門用語は英語のまま。
:::

## モダリティ × 横断手法

```mermaid
flowchart TB
  subgraph M[モダリティ（様式）]
    L[🔤 言語] 
    A[🎵 音声]
    V[👁 視覚]
    P[🦾 身体性・行動]
    X[🔀 マルチモーダル]
  end
  subgraph C[横断軸（学び方）]
    RL[🎯 強化学習・学習パラダイム]
  end
  RL -.->|RLHF| L
  RL -.->|制御| P
  RL -.->|報酬設計| A
```

## モダリティ一覧

<div class="cards">
  <div class="card">
    <h3>🔤 言語（LLM）</h3>
    <p>トークン列の言語モデリング。Transformer・事前学習・適応・推論最適化・推論モデル/エージェント。</p>
    <p><a href="/llm/">→ ロードマップ</a></p>
  </div>
  <div class="card">
    <h3>🎵 音声（Audio）</h3>
    <p>波形 ⇄ 特徴量 ⇄ トークン。ASR と TTS（自己回帰 / flow matching / 全二重 streaming）。<b>全8章 公開</b>。</p>
    <p><a href="/audio/">→ ロードマップ</a></p>
  </div>
  <div class="card">
    <h3>👁 視覚（Vision）</h3>
    <p>画像・動画の理解と生成。CNN/ViT、拡散モデル、表現学習。</p>
    <p><a href="/vision/">→ ロードマップ</a></p>
  </div>
  <div class="card">
    <h3>🦾 身体性・行動（Physical AI）</h3>
    <p>ロボット・制御・知覚・sim-to-real。物理世界で動く AI の知能。</p>
    <p><a href="/physical-ai/">→ ロードマップ</a></p>
  </div>
  <div class="card">
    <h3>🔀 マルチモーダル</h3>
    <p>モダリティ横断。vision-language・audio-language・any-to-any。</p>
    <p><a href="/multimodal/">→ ロードマップ</a></p>
  </div>
  <div class="card">
    <h3>🎯 強化学習（横断）</h3>
    <p>モダリティに直交する「学び方」。MDP から深層強化学習、RLHF・制御まで。</p>
    <p><a href="/reinforcement-learning/">→ ロードマップ</a></p>
  </div>
</div>

## 各章の構成（フルセット）

どのモダリティのどの章も、次の流れで書かれています。

```mermaid
flowchart LR
  A[学習目標] --> B[前提知識] --> C[直感] --> D[全体像]
  D --> E[理論] --> F[数式の導出] --> G[実装] --> H[演習]
  H --> I[まとめ] --> J[用語ミニ辞典] --> K[次のアクション] --> L[参考文献]
```

品質の心臓部は [`AUTHORING.md`](https://github.com/ksterx/textbook/blob/main/AUTHORING.md) の **§2「説明の深さ基準」**。
完成済みの **[音声モダリティ](/audio/)（全8章）** がお手本です。
