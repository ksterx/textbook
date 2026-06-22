---
layout: ../../layouts/Page.astro
title: ケーススタディ：Qwen3-ASR（LLM デコーダ型 ASR）
description: 章04 の3系統でいう attention（AED）系の最新形 —— デコーダが LLM そのものの Qwen3-ASR を、技術報告（arXiv:2601.21337）を精査して読み解く。AuT エンコーダ＋Qwen3 LLM の LALM 構成、context biasing、2 秒チャンク streaming、4 段階学習、forced aligner まで。章09 の Nemotron（transducer）との対極として位置づける。
---

# ケーススタディ：Qwen3-ASR（LLM デコーダ型 ASR）

:::abstract[学習目標]
この章を読み終えると、次のことができるようになります。

- **LALM（large audio-language model）型 ASR** が、従来の CTC/RNN-T/AED と何が違うかを **説明** できる
- Qwen3-ASR の構成（**AuT エンコーダ ＋ projector ＋ Qwen3 LLM**）を、章04 の AED の語彙で **述べ** られる
- **context biasing**（プロンプトに文脈を入れて認識を誘導）の仕組みを説明できる
- なぜ AED 系は streaming が苦手で、Qwen3-ASR が **2 秒チャンク**でどう streaming するかを **述べ**、章09 の Nemotron（transducer）と **対比** できる
- 4 段階学習（AuT 事前学習 → Omni 事前学習 → ASR SFT → RL）と forced aligner の役割を説明できる
:::

## 前提知識

- 章04 [音声認識 (ASR)](/audio/04-asr/)：**3系統（CTC / RNN-T / AED）**、とくに **AED（attention encoder-decoder）が label 同期で streaming が苦手**という点。
- 章09 [ケーススタディ：Nemotron 3.5 ASR](/audio/09-nemotron-streaming-asr/)：**対極の設計**（FastConformer + RNN-T、frame 同期の低遅延 streaming）。本章はこれと対比して読むと理解が深まります。
- LLM の基礎（自己回帰デコード・トークナイザ・プロンプト・KV cache）：[LLM 分野](/llm/) 全般。

## 直感

章09 の Nemotron は「音響を frame 同期で文字に変える」transducer でした。Qwen3-ASR は**真逆の発想**です ——

> **まず音声を「理解」し、その理解に条件づけて LLM が書き起こしを生成する。**

技術報告はこれを **LALM（large audio-language model）型**と呼び、従来の transducer/AED を「ボトムアップの音響パターンマッチング」と対比します。要は **デコーダを LLM そのものにした ASR** で、章04 の3系統でいう **AED（attention）系の最新形**です。

この設計の旨みは、LLM が持つ言語力をそのまま使えること —— **文脈（固有名・専門用語）をプロンプトで与えて認識を誘導**でき（context biasing）、雑音・歌唱・コードスイッチにも強い。代償は、AED 系ゆえ **streaming が本質的に苦手**で、低遅延では transducer に譲ること。この「精度・文脈 ↔ 低遅延」の対極を、Nemotron と並べて掴むのが本章のゴールです。

## 全体像

技術報告のアーキテクチャ図（CC BY 4.0）をそのまま引用します。**左が AuT 単体、右が Qwen3-ASR 全体**です。

<figure>
  <img src="/textbook/img/qwen3-asr/architecture.png" alt="左: AuT エンコーダ・デコーダ（AED）。右: Entities・音声・言語タグを Qwen3 LM に入れて書き起こしを生成する Qwen3-ASR 全体図">
  <figcaption class="fig-cap"><span>Fig.2「Architecture of AuT (left) and the overview of Qwen3-ASR (right).」左: AuT は 100Hz Fbank → 3× Downsampling Conv2d → 32× Self-Attention（12.5Hz）→ 8× デコーダ の AED。右: 事前学習済み AuT エンコーダ＋文脈(Entities)＋言語タグを Qwen3 LM に流す。</span><span>出典: Xian Shi et al., <a href="https://arxiv.org/abs/2601.21337">arXiv:2601.21337</a> / <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>（リサイズして掲載）</span></figcaption>
</figure>

章04・章09 との対応:

| Qwen3-ASR の部品 | 章04 / 章09 のどこ |
| --- | --- |
| AuT エンコーダ（音響 → 埋め込み） | 章04「エンコーダ」（ただし Conformer でなく AED 由来の self-attention エンコーダ） |
| Qwen3 LLM デコーダ | 章04「AED の decoder」を **LLM に置換**（章09 の小さな LSTM prediction net とは対極） |
| context（Entities）をプロンプトに | **本章の新顔**（LLM ならではの文脈バイアス） |
| 2 秒チャンク streaming | 章04「chunk-wise」／章09「dynamic chunk」の AED 版 |

:::note[Nemotron（章09）との対極]
| | Nemotron 3.5 ASR | Qwen3-ASR |
| --- | --- | --- |
| パラダイム | FastConformer + **RNN-T（transducer）** | AuT エンコーダ + **LLM デコーダ（AED/LALM）** |
| デコーダ | 2 層 LSTM（小さな内部 LM） | **Qwen3 LLM そのもの**（0.6B / 1.7B） |
| 同期 | frame 同期（低遅延の本命） | label 同期（AED）→ チャンク化で streaming |
| チャンク遅延 | 80 ms 〜 1.12 s | 2 秒（計算 TTFT は 0.6B で 92 ms と速い） |
| 強み | 低遅延・高スループット | 文脈バイアス・高精度・歌唱/雑音 |
:::

## 理論

### 1. LALM 型とは（従来 E2E との違い）

章04 では ASR を **CTC / RNN-T / AED** の3系統に分けました。LALM 型は **AED の decoder を汎用 LLM にしたもの**で、技術報告の主張はこうです（要約）。

- 従来（transducer/AED）：音響特徴を**ボトムアップにパターンマッチ**して文字へ。
- LALM：まず音声の**高レベルな理解**を作り、その理解に条件づけて**次トークン予測**で書き起こす。音響マッチングだけに依存しない。

基盤は **Qwen3-Omni**（音声理解の強い基盤モデル）。そこから ASR 用に post-train します。

:::warning[「LLM が音を直接聞く」わけではない]
LLM はテキストトークンしか扱えません。音声は **AuT エンコーダ**が埋め込み列に変換し、**projector** で LLM の入力空間へ写してから LLM に渡します（Fig.2 右）。LLM は「音声の埋め込み＋文脈テキスト＋言語タグ」という**1 本の系列**を受け取り、その続きを生成するだけです。
:::

### 2. アーキテクチャ（数値）

技術報告 §2.1 の実値です。

| 部品 | 1.7B 版 | 0.6B 版 |
| --- | --- | --- |
| LLM バックボーン | Qwen3-1.7B | Qwen3-0.6B |
| AuT エンコーダ | **300M・hidden 1024** | **180M・hidden 896** |
| projector | エンコーダ出力を LLM 入力空間へ写像（内部構造は論文に明示なし） | 同左 |

AuT エンコーダの要点:

- **入力 128 次元 Fbank（100Hz）→ 8× ダウンサンプル → 12.5Hz token rate**（章09 の FastConformer 8× と同じ「間引きで軽くする」発想。フレームは 1/12.5 = **80 ms**）。
- 図のとおり **3× Downsampling Conv2d → 32× Self-Attention**。AuT は単体では **8× のデコーダ**を持つ **AED** として事前学習され、Qwen3-ASR ではその**エンコーダ部分**を流用します。
- **dynamic flash attention window 1〜8 秒**：注意の参照窓を可変にし、**短窓＝streaming・長窓＝offline** を 1 モデルで切替（後述「streaming」）。

### 3. ASR を「次トークン予測」として解く ＋ context biasing

Qwen3-ASR は書き起こしを **LLM の次トークン予測**で生成します。入力系列は概ね 3 つの部品の連結です（Fig.2 右）。

1. **context（Entities）**：固有名・専門用語などのテキストをトークン化して与える。
2. **音声**：AuT エンコーダ → projector の埋め込み列。
3. **言語タグ**：`language Chinese <asr_text>` のような指定。続きに書き起こしが生成される。

**context biasing** は ① のことです。モデルは **system prompt 内の context トークンを背景知識として使う**よう学習されており、固有名や用語をあらかじめ渡すと、その語に寄せた認識ができます。

:::warning[context biasing の厳密な書式は論文に明示なし]
「プロンプトに文脈テキストを入れて誘導する」までが技術報告の明言範囲です。特殊トークンの正確な書式や引数名は本文に書かれていないので、実装時は公式 README / サンプルで確認してください。同様に、**projector の内部構造**や、音声が連続埋め込みか離散トークンかも論文は明示していません。
:::

:::note[LLM ↔ Speech]
これは LLM の **マルチモーダル入力＋プロンプト**そのものです。音声は「画像を埋め込みで入れる」のと同じ要領で埋め込みとして差し込み、context は **system prompt の指示**、言語タグは**タスク指定トークン**にあたります。章09 の「言語 ID の one-hot」が、ここでは**自由テキストのプロンプト**に進化した、と捉えると橋が架かります。
:::

### 4. streaming：2 秒チャンクで「理解型」を逐次化する

AED 系は章04 のとおり **label 同期＝原則すべてのフレームが揃ってから動く**ので、本来 streaming が苦手です。Qwen3-ASR はこれを **chunk 化**で乗り越えます。

- 基盤は **dynamic flash attention window（1〜8 秒）**。短い窓で動かせば逐次処理になります。
- 技術報告 §4.5 の streaming 設定：**2 秒チャンク・5-token fallback・直近 4 チャンクを未確定（unfixed）に保つ**。
  - **直近 4 チャンクを未確定**：出したばかりの出力は**後続の文脈で訂正可能**として扱い、確定を遅らせる。
  - **5-token fallback**：直近トークンの扱いに関するフォールバック（**論文に明確な機構説明はなく**、断定は避けます）。
- 公開実装では **streaming は vLLM バックエンド限定**、かつ **streaming 時はタイムスタンプ非対応**（README）。

精度は offline に肉薄します（3 ベンチ平均, Table 8）。

<figure>
  <canvas id="qwen3-offline-streaming" width="1200" height="640" aria-hidden="true"></canvas>
  <figcaption class="fig-cap"><span>offline vs streaming WER（低いほど良い・3ベンチ平均, Table 8）</span><span>劣化は 1.7B で +0.64pt、0.6B で +0.92pt 程度</span></figcaption>
</figure>

:::warning[「遅延」を2層に分けて読む（混同しやすい）]
Qwen3-ASR の遅延は2層あります（章04 の「アルゴリズム遅延 ＋ emission/計算遅延」と同じ構図）。**「計算が遅い」と「チャンクが粗い」は別物**です。

| 遅延の種類 | 何を測る | Qwen3-ASR |
| --- | --- | --- |
| **チャンク（バッファ）遅延** | 処理前にどれだけ音声を貯めるか | streaming は **2 秒チャンク**＝2 秒ぶんコミット |
| **計算遅延（TTFT / RTF）** | 投げてから最初のトークンまで | **0.6B は最小 92 ms**・RTF 0.064 と**高速** |

つまり **Qwen3-ASR は計算自体は速い（TTFT 92 ms）が、streaming では 2 秒粒度で音声をコミットする**設計です（体感遅延 ≈ チャンク貯め ＋ TTFT）。章09 Nemotron との差は **チャンク粒度**（80 ms 〜 1.12 s vs 2 秒＝どれだけ先の音声を待って確定するか）の話で、計算の速さとは別軸。**frame 同期の低遅延 streaming は依然 transducer 系（Nemotron）**が主役、Qwen3-ASR は「高精度・文脈理解を保ったまま逐次化したい」用途、という棲み分けです。なお論文は frame/label 同期の語で機構を明言していません（AED 原理から label 同期的、と読むに留める）。
:::

効率（Table 2, vLLM）も小型版が強力です：**0.6B は TTFT 最小 92 ms・RTF 0.064・同時 128 でスループット 2000**（1 秒で 2,000 秒ぶんの音声）、最大系列長 **1200 秒**。**TTFT 92 ms は「投げてから最初の字が出るまで」の速さ**で、上のチャンク遅延（2 秒）とは別軸であることに注意。

### 5. 学習：4 段階（本報告による）

技術報告は 4 段階を挙げます（規模値は本報告の記載）。

1. **AuT 事前学習**：AED 枠組みで、**約 4,000 万時間**の擬似ラベル ASR データ（大半が中英）からエンコーダを学習。出力 12.5Hz。
2. **Omni 事前学習**：音声・視覚・テキストのマルチタスクを Qwen3-Omni パイプラインで、**3 兆トークン**学習。
3. **ASR SFT**：ASR 入出力フォーマットへの style transfer。出力テンプレートと **context トークンの活用**を学習。
4. **ASR RL**：**GSPO（Group Sequence Policy Optimization）** で、雑音頑健性・安定性・難例解析を強化（RL データ約 5 万発話）。

:::note[LLM ↔ Speech]
「巨大データで基盤を作り → タスク用に SFT → RL で仕上げる」という **LLM のポストトレーニングそのもの**を ASR に適用しています。章09 の Nemotron が「擬似ラベル＋ end-to-end」だったのに対し、Qwen3-ASR は **LLM の学習レシピ（SFT＋RL）**を持ち込んでいるのが特徴です。
:::

### 6. forced aligner（タイムスタンプ）

Qwen3-ASR ファミリーには、書き起こしに**単語/文字の開始・終了時刻**を付ける専用モデル **Qwen3-ForcedAligner-0.6B** が含まれます。

<figure>
  <img src="/textbook/img/qwen3-asr/forced-aligner.png" alt="Qwen3-ForcedAligner の図。マスクしたタイムスタンプ・スロットをトークン列に挿入し、Qwen3-0.6B LLM＋タイムスタンプ予測層で各スロットの時刻インデックスを予測する">
  <figcaption class="fig-cap"><span>Fig.3 Qwen3-ForcedAligner-0.6B：マスクしたタイムスタンプ・スロットを系列に挿入 → Qwen3-0.6B LLM ＋ timestamp 予測層で各スロットの時刻を予測（境界のみ CE 損失）。</span><span>出典: Xian Shi et al., <a href="https://arxiv.org/abs/2601.21337">arXiv:2601.21337</a> / <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>（リサイズして掲載）</span></figcaption>
</figure>

- **非自己回帰（NAR）**：次トークン予測をやめ、**全スロットの時刻を同時に予測**（ASR 本体の自己回帰とは対照的）。
- 対応 **11 言語**、最大 **300 秒（5 分）**。学習は **MFA（Montreal Forced Aligner）の擬似タイムスタンプ**を蒸留・平滑化し、**timestamp スロットだけ**にクロスエントロピーを当てる。
- 指標は **AAS（Accumulated Average Shift）**：人手ラベルで平均 **約 32.4 ms**、他手法比 **相対 67〜77% 減**。

## 数式の導出

### LALM（AED）の分解 vs RNN-T の周辺化

LALM/AED は書き起こしを**自己回帰**で生成します。入力プロンプト（音声埋め込み＋context＋言語タグ）を $c$ とすると、

$$
P(y \mid c) = \prod_{u=1}^{U} P\!\left(y_u \mid y_{<u},\, c\right)
$$

各トークン $y_u$ は「直前まで $y_{<u}$ ＋入力全体 $c$」を条件に出ます。**入力全体に依存する＝原則すべて揃ってから**動く（label 同期）。これが AED 系で streaming が難しい理由で、章09 の RNN-T が

$$
P(y \mid x) = \sum_{a \in B^{-1}(y)} \prod_{t=1}^{T} P(a_t \mid x)
$$

と**アライメントを周辺化して frame 同期**で出せたのと対照的です。Qwen3-ASR はこの $P(y\mid c)$ を**チャンクごとに**走らせ、直近を未確定にして後で直す、という形で逐次化します。$\blacksquare$

### AAS（forced aligner の指標）

予測時刻 $\hat n_i$ と参照 $n_i$、スロット総数 $N$ として、

$$
\mathrm{AAS} = \frac{1}{N}\sum_{i=1}^{N} \left| \hat n_i - n_i \right|
$$

境界の平均ズレ（小さいほど良い）です。$\blacksquare$

## 実装

実モデルは GPU と vLLM（streaming）が要るのでここでは未実行ですが、**核となる 2 つの考え方**は手元で確かめられます —— ① LALM 入力（context biasing 込み）の組み立て、② forced aligner の AAS。

```python title="qwen_demo.py"
import numpy as np

# --- 1) LALM の入力の「組み立て」（概念）---
# Qwen3-ASR は (文脈 entities) + (音声=AuT 埋め込み) + (言語タグ) を 1 本の系列にして
# Qwen3 LM に入れ、次トークン予測で書き起こしを出す（論文 Fig.2 右）。
# ※特殊トークンの正確な書式は実装依存。ここでは構造だけを示す。
def build_input(audio_emb_len, language, context_entities):
    parts = []
    if context_entities:                       # 文脈バイアス：固有名などを背景知識として与える
        parts.append(("context", "tokens", len(" ".join(context_entities).split())))
    parts.append(("audio",   "AuT埋め込み", audio_emb_len))   # 連続埋め込み（12.5Hz）
    parts.append(("langtag", f"language {language} <asr_text>", 3))
    return parts

print("=== LALM 入力の構造（context あり／なし）===")
for ctx in (["Tongyi Lab", "Qwen", "Qwen3-Omni"], []):
    seq = build_input(audio_emb_len=25, language="Chinese", context_entities=ctx)
    total = sum(n for _, _, n in seq)
    label = "context あり" if ctx else "context なし"
    print(f"[{label}] " + " | ".join(f"{k}:{n}" for k, _, n in seq) + f"  → 系列長~{total} トークン")
print("→ この系列の続きを LLM が自己回帰生成 = 書き起こし")

# --- 2) Forced aligner の指標 AAS ---
# AAS = (1/N) Σ |予測 - 参照|（境界タイムスタンプの平均ズレ・小さいほど良い, 論文 Eq.1）
ref  = np.array([0.00, 0.32, 0.55, 0.88, 1.20])  # 参照 (秒)
pred = np.array([0.02, 0.30, 0.58, 0.85, 1.23])  # 予測 (秒)
aas_ms = np.mean(np.abs(pred - ref)) * 1000
print("\n=== AAS（forced aligner 指標）===")
print(f"スロット数 N = {len(ref)}")
print(f"AAS = mean(|pred-ref|) = {aas_ms:.1f} ms（小さいほど高精度。論文の人手ラベル平均は ~32.4ms）")
```

```text title="出力"
=== LALM 入力の構造（context あり／なし）===
[context あり] context:4 | audio:25 | langtag:3  → 系列長~32 トークン
[context なし] audio:25 | langtag:3  → 系列長~28 トークン
→ この系列の続きを LLM が自己回帰生成 = 書き起こし

=== AAS（forced aligner 指標）===
スロット数 N = 5
AAS = mean(|pred-ref|) = 26.0 ms（小さいほど高精度。論文の人手ラベル平均は ~32.4ms）
```

context を入れると系列の頭に文脈トークンが増え、それを条件に書き起こしが変わる —— これが context biasing の最小形です。

## 演習

::::question[演習 1: なぜ AED は streaming が苦手か]
(a) RNN-T が frame 同期で低遅延に出せるのに、Qwen3-ASR（AED/LALM）が原則「全フレーム待ち」になるのはなぜですか。式 $P(y\mid c)=\prod_u P(y_u\mid y_{<u},c)$ を使って答えてください。(b) Qwen3-ASR はそれをどう逐次化していますか。

:::details[解答]
(a) 各トークン $y_u$ が**入力全体 $c$**（音声プロンプト全部）を条件にするため、$c$ が揃わないと正しい分布が出せません（label 同期）。RNN-T は時刻 $t$ までの $f_t$ と出力済み $g_u$ だけで出せる（frame 同期）ので未来を待ちません。
(b) 入力を **2 秒チャンク**に区切り、dynamic attention window（1〜8s）を短窓で使って逐次処理。直近 4 チャンクを**未確定**にして後続で訂正します。ただし遅延はチャンク幅 ≈ 秒オーダーで、frame 同期ほど低くはなりません。
:::
::::

::::question[演習 2: context biasing と Nemotron の言語条件付け]
Qwen3-ASR の context biasing（プロンプトに文脈テキスト）と、章09 Nemotron の言語 ID（128 次元 one-hot の連結）は、どちらも「外から条件を与える」点で似ています。(a) 表現の自由度はどちらが高いですか。(b) その代償（コスト）は何ですか。

:::details[解答]
(a) **Qwen3-ASR**。自由テキストで固有名・用語・文脈を渡せるので、one-hot の「言語インデックス 1 個」より遥かに表現力が高い。
(b) 代償は**計算とレイテンシ**。context はトークンとして LLM 系列に乗るので長いほど重く、そもそもデコーダが LLM なので transducer より重い。Nemotron の one-hot は 1 ベクトル連結でほぼ無料だが、与えられるのは「言語」だけ。
:::
::::

::::question[演習 3: AAS を計算する]
あるアライナの予測時刻が `[0.10, 0.40, 0.70]`、参照が `[0.12, 0.35, 0.69]`（秒）でした。AAS を ms で求めてください。

:::details[解答]
差の絶対値＝`|−0.02|, |0.05|, |0.01|` ＝ `0.02, 0.05, 0.01`。平均＝`0.08/3 ≈ 0.02667` 秒 ＝ **約 26.7 ms**。
:::
::::

## まとめ

:::success[この章の要点]
- Qwen3-ASR は **LALM 型 ASR** ＝ 章04 の **AED 系の最新形で、decoder が Qwen3 LLM そのもの**。「理解してから書き起こす」。
- 構成は **AuT エンコーダ（300M/180M・128 Fbank・8×→12.5Hz）＋ projector ＋ Qwen3 LLM**。音声は埋め込みで LLM に差し込む。
- **context biasing**：プロンプトに文脈テキスト（固有名・用語）を入れて認識を誘導。Nemotron の言語 one-hot より自由度が高いが重い。
- **streaming は 2 秒チャンク**（dynamic window 1〜8s・直近4チャンク未確定・vLLM 限定）。精度は offline に肉薄（1.7B +0.64pt）。遅延は2層 ——**計算は速い（0.6B TTFT 92 ms）が、チャンク粒度が 2 秒**で frame 同期の transducer ほど細かくはコミットしない。
- 学習は **AuT 事前学習 → Omni 3T → SFT → RL（GSPO）** の LLM 流。**forced aligner** は NAR で時刻を当て、AAS ~32.4ms。
- ライセンス：**論文 CC BY 4.0／モデル Apache 2.0**（対象が別物）。
:::

### 次に学ぶこと

章09（transducer）と本章（LALM）で、streaming ASR の**両極**を見ました。次は逆向き（テキスト→音声）の合成 (TTS) へ。LLM をデコーダに据える発想は、章08 の**統合・全二重 streaming**（聞きながら話す）とも地続きです。

→ [Audio ロードマップに戻る](/audio/) ／ 対極の設計は [章09 Nemotron](/audio/09-nemotron-streaming-asr/)

## 用語ミニ辞典

| 用語 | 一言 |
| --- | --- |
| LALM | large audio-language model。音声を理解し LLM で書き起こす ASR |
| AuT エンコーダ | AED として事前学習した音声エンコーダ（128 Fbank・8×→12.5Hz） |
| projector | エンコーダ出力を LLM 入力空間へ写す橋渡し |
| context biasing | プロンプトに文脈テキストを入れて認識を誘導する |
| dynamic attention window | 注意の参照窓を 1〜8s 可変化（streaming/offline 兼用） |
| 5-token fallback | streaming 時の直近トークンに関するフォールバック（機構は論文未明示） |
| GSPO | ASR RL に使う Group Sequence Policy Optimization |
| forced aligner | 書き起こしに時刻を付ける NAR モデル（指標 AAS） |
| AAS | 境界時刻の平均ズレ。小さいほど良い |

## 次のアクション

理論を手で定着させる。**最小の写経 → 動かす → 小実験** を 1 セットで。

1. **写経**：上の `qwen_demo.py` を `uv run --with numpy python qwen_demo.py` で動かし、出力が本文と一致するか確認する。
2. **動かす**：`build_input` の `context_entities` を増減し、系列長と「context あり/なし」の差を観察する（context biasing が系列に乗る様子）。
3. **小実験**：演習3のように予測時刻を変えて AAS が増減するのを確かめ、「境界が 50ms ずれると AAS がどう動くか」を体感する。
4. 余力があれば、GPU＋vLLM 環境で公開実装を動かし、同じ音声を **offline と 2 秒チャンク streaming** で回して WER と体感遅延の差を測る。

## 参考文献

1. Xian Shi, Xiong Wang, Zhifang Guo, et al., "Qwen3-ASR Technical Report," *arXiv:2601.21337*, 2026. [arXiv](https://arxiv.org/abs/2601.21337)（本章の一次情報。図 Fig.2/Fig.3 は arXiv 版から [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) のもとリサイズして掲載。数値は 2026-01 時点）
2. QwenLM, "Qwen3-ASR," GitHub（モデルは Apache 2.0。streaming は vLLM バックエンド限定）. [github.com/QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
3. 章04 [音声認識 (ASR) とストリーミング](/audio/04-asr/)（CTC/RNN-T/AED と streaming の土台）
4. 章09 [ケーススタディ：Nemotron 3.5 ASR](/audio/09-nemotron-streaming-asr/)（対極の transducer 型 streaming）
