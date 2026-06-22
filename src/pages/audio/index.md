---
layout: ../../layouts/Page.astro
title: Audio（音声・音響）
description: Audio 分野のロードマップ。サンプリングから streaming ASR / flow-matching TTS まで。
---

# Audio（音声・音響）

音をコンピュータで扱うための基礎から、信号処理、機械学習による音声認識・合成までを体系的に学びます。

:::abstract[この分野で身につくこと]
- 音をデジタルで表現する仕組み（sampling, quantization）を説明できる
- 時間領域・周波数領域を行き来して信号を分析できる（Fourier 変換, STFT）
- 音声特徴量（spectrogram, MFCC など）を自分で計算できる
- 音声認識・音声合成の基本的な仕組みを理解する
:::

## North Star（最終目標）

この分野を学び切った先で「自力で学習でき、新しいアーキも提案できる」状態を目指します。

1. **Streaming ASR** — 低遅延で逐次に音声を認識する（CTC / RNN-T / chunk-wise attention）
2. **Flow-matching TTS** — 非自己回帰・数ステップで音声を合成する（F5-TTS 系）
3. **Streaming TTS（CALM / Kyutai 系）** — テキストと音声を単一モデルで同時に流す

:::tip[LLM 出身者向けの近道]
Transformer・自己回帰デコード・トークナイザの知識はそのまま使えます。音声で新しいのは主に 2 点
——「**連続音声のトークン化／特徴量化**」と「**ストリーミング遅延**」。ここに時間を集中させます。
:::

## 前提知識

- 高校〜大学初年度の微積分・線形代数
- Python の基本（NumPy で配列を扱える程度）

## ロードマップ

```mermaid
flowchart TD
  C1[1. デジタル音声の基礎<br/>サンプリングと量子化] --> C2[2. 周波数領域と特徴量<br/>Fourier・STFT・mel・MFCC]
  C2 --> C3[3. ニューラル音声コーデック<br/>VQ・RVQ・Mimi]
  C2 --> C4[4. 音声認識 ASR<br/>CTC・RNN-T・streaming / 目標①]
  C3 --> C5[5. 現代 TTS の全体地図<br/>分類とトレンド]
  C5 --> C6[6. トークンベース TTS<br/>VALL-E 系]
  C5 --> C7[7. 連続生成 TTS<br/>flow matching / 目標②]
  C5 --> C8[8. 統合・全二重 streaming<br/>Moshi・DSM / 目標③]
```

各ステージは **学ぶ（理論）/ 橋渡し（既知との接続）/ 作る（最小実装）** で進めます。

### 1. デジタル音声の基礎 ✅ Ready

- **学ぶ**: サンプリング・Nyquist・aliasing・量子化・SQNR
- **橋渡し**: 「波形 = 非常に長い 1 次元数列」。LLM のトークン列より桁違いに長く冗長 → 周波数領域へ移して圧縮する動機
- **作る**: NumPy でサンプリング／エイリアシング／量子化を実装し、理論式を実測で確認

→ [読む](/audio/01-digital-audio-basics/)

### 2. 周波数領域とスペクトル特徴量 ✅ Ready

- **学ぶ**: DFT/FFT・STFT・スペクトログラム・メルフィルタバンク・log-mel・MFCC・Griffin-Lim/vocoder
- **橋渡し**: 「STFT→mel = トークナイザ」「log-mel フレーム = 埋め込み」「vocoder = デトークナイザ」
- **作る**: torchaudio / NumPy で 波形 ⇄ STFT ⇄ mel ⇄ 波形 を一周

→ [読む](/audio/02-frequency-and-features/)

### 3. ニューラル音声コーデック ✅ Ready

- **学ぶ**: VQ・RVQ・FSQ・EnCodec/SoundStream/DAC/Mimi・frame/bit rate・semantic↔acoustic
- **橋渡し**: 「音声 = 離散トークン列」→ TTS が言語モデリングに化ける入口（VALL-E / Moshi）
- **作る**: 学習済み codec で encode→decode し、トークン列を可視化

→ [読む](/audio/03-neural-audio-codecs/)

### 4. 音声認識 ASR ✅ Ready ／ 目標①

- **学ぶ**: アライメント問題・CTC・RNN-T(transducer)・attention・Conformer/FastConformer・chunk streaming・遅延↔精度
- **橋渡し**: prediction net = ラベルの自己回帰 LM、causal mask = streaming の前提（codec と同根）
- **作る**: LibriSpeech サブセットで CTC を学習 → streaming 推論化し、遅延↔WER を測る

→ [読む](/audio/04-asr/)

### 5. 現代 TTS の全体地図 ✅ Ready

- **学ぶ**: TTS の分類軸（離散/連続・AR/NAR・統合 streaming）・代表研究の年表・研究トレンド
- **橋渡し**: 各論（6〜8章）に入る前の地図。CALM/Moshi は「統合・全二重」ブランチの一部と位置づける
- **作る**: 学習済み TTS を俯瞰して触り、分類のどこに当たるか確かめる

→ [読む](/audio/05-tts-landscape/)

### 6–8. TTS 各論 ✅ Ready ／ 🎯 目標②③

- **6. トークンベース TTS（VALL-E 系）** ✅ — codec トークン + 自己回帰 LM（[読む](/audio/06-token-based-tts/)）
- **7. 連続生成 TTS（flow matching・F5-TTS 系）** 🎯 目標② — 非自己回帰・数ステップ生成（[読む](/audio/07-flow-matching-tts/)）
- **8. 統合・全二重 streaming TTS（Moshi / Kyutai DSM）** 🎯 目標③ — テキストも音声も streaming（[読む](/audio/08-unified-streaming-tts/)）

### 9. ケーススタディ：Nemotron 3.5 ASR ✅ Ready

- **学ぶ**: 章04 の理論を 1 実モデルで読む —— 言語 ID 条件付け（1 モデルで 40 言語）・5 段階レイテンシ（`att_context_size`）・cache-aware のスループット経済性・擬似ラベル学習
- **橋渡し**: 言語 ID one-hot = LLM のタスクトークン／system prompt。cache-aware = encoder 側の KV cache
- **作る**: `att_context_size` → 遅延、buffered vs cache-aware コストを NumPy で実測

→ [読む](/audio/09-nemotron-streaming-asr/)

### 10. ケーススタディ：Qwen3-ASR ✅ Ready

- **学ぶ**: 章04 の AED 系の最新形 —— **デコーダが LLM そのもの**（LALM）。AuT エンコーダ＋Qwen3 LLM・context biasing・2 秒チャンク streaming・4 段階学習・forced aligner
- **橋渡し**: 章09 Nemotron（transducer・低遅延）の**対極**。LLM のマルチモーダル入力＋プロンプトで ASR を解く
- **作る**: LALM 入力（context あり/なし）の組み立て、forced aligner の AAS を NumPy で実測

→ [読む](/audio/10-qwen3-asr/)

## 章一覧

| # | 章 | 状態 |
| --- | --- | --- |
| 1 | [デジタル音声の基礎 — サンプリングと量子化](/audio/01-digital-audio-basics/) | ✅ 公開 |
| 2 | [周波数領域とスペクトル特徴量](/audio/02-frequency-and-features/) | ✅ 公開 |
| 3 | [ニューラル音声コーデック](/audio/03-neural-audio-codecs/) | ✅ 公開 |
| 4 | [音声認識 (ASR) とストリーミング](/audio/04-asr/) | ✅ 公開 |
| 5 | [現代 TTS の全体地図 — 分類と研究トレンド](/audio/05-tts-landscape/) | ✅ 公開 |
| 6 | [トークンベース TTS（VALL-E 系）](/audio/06-token-based-tts/) | ✅ 公開 |
| 7 | [連続生成 TTS（flow matching・F5-TTS 系）](/audio/07-flow-matching-tts/) | ✅ 公開 / 🎯目標② |
| 8 | [統合・全二重 streaming TTS（Moshi / DSM）](/audio/08-unified-streaming-tts/) | ✅ 公開 / 🎯目標③ |
| 9 | [ケーススタディ：Nemotron 3.5 多言語ストリーミング ASR](/audio/09-nemotron-streaming-asr/) | ✅ 公開 |
| 10 | [ケーススタディ：Qwen3-ASR（LLM デコーダ型 ASR）](/audio/10-qwen3-asr/) | ✅ 公開 |

:::note[章は順次追加されます]
「次は◯◯の章を書いて」と指示すると、統一フォーマットで新しい章が追加されます。
:::
