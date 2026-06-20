// サイトのナビゲーション。トップの分類軸は「モダリティ（入出力の様式）」。
// 強化学習はモダリティに直交する「横断的な学習パラダイム」として末尾に置く。
// 章を足したら該当モダリティの items に1行追加する。href はルート絶対（base なし）。
export const NAV = [
  {
    domain: '🗺 全体地図',
    items: [{ title: 'モダリティ地図', href: '/' }],
  },
  {
    domain: '🔤 言語（LLM）',
    items: [
      { title: 'ロードマップ', href: '/llm/' },
      { title: '1. 言語モデルとトークン化', href: '/llm/01-language-model-and-tokenization/' },
      { title: '2. Attention 機構', href: '/llm/02-attention/' },
      { title: '3. Transformer の構造', href: '/llm/03-transformer/' },
      { title: '4. 事前学習とスケーリング則', href: '/llm/04-pretraining-scaling/' },
      { title: '5. 適応 — RLHF・DPO', href: '/llm/05-adaptation-rlhf/' },
      { title: '6. 推論と効率化', href: '/llm/06-inference-efficiency/' },
      { title: '7. 現代 LLM の地図', href: '/llm/07-llm-landscape/' },
    ],
  },
  {
    domain: '🎵 音声（Audio）',
    items: [
      { title: 'ロードマップ', href: '/audio/' },
      { title: '1. デジタル音声の基礎', href: '/audio/01-digital-audio-basics/' },
      { title: '2. 周波数領域とスペクトル特徴量', href: '/audio/02-frequency-and-features/' },
      { title: '3. ニューラル音声コーデック', href: '/audio/03-neural-audio-codecs/' },
      { title: '4. 音声認識 (ASR)', href: '/audio/04-asr/' },
      { title: '5. 現代 TTS の全体地図', href: '/audio/05-tts-landscape/' },
      { title: '6. トークンベース TTS（VALL-E）', href: '/audio/06-token-based-tts/' },
      { title: '7. 連続生成 TTS（flow matching）', href: '/audio/07-flow-matching-tts/' },
      { title: '8. 統合・全二重 streaming TTS', href: '/audio/08-unified-streaming-tts/' },
    ],
  },
  {
    domain: '👁 視覚（Vision）',
    items: [
      { title: 'ロードマップ', href: '/vision/' },
      { title: '1. CNN（畳み込み）', href: '/vision/01-cnn/' },
      { title: '2. Vision Transformer (ViT)', href: '/vision/02-vit/' },
      { title: '3. 自己教師あり表現学習', href: '/vision/03-self-supervised/' },
      { title: '4. 画像生成（拡散モデル）', href: '/vision/04-diffusion-generation/' },
      { title: '5. 検出とセグメンテーション', href: '/vision/05-detection-segmentation/' },
      { title: '6. 現代 Vision の地図', href: '/vision/06-vision-landscape/' },
    ],
  },
  {
    domain: '🦾 身体性・行動（Physical AI）',
    items: [
      { title: 'ロードマップ', href: '/physical-ai/' },
      { title: '1. 全体像と座標系', href: '/physical-ai/01-overview-and-frames/' },
      { title: '2. 運動学（kinematics）', href: '/physical-ai/02-kinematics/' },
      { title: '3. 古典制御（PID）', href: '/physical-ai/03-pid-control/' },
      { title: '4. 最適制御（LQR）', href: '/physical-ai/04-optimal-control/' },
      { title: '5. 知覚と状態推定（Kalman）', href: '/physical-ai/05-perception-state-estimation/' },
      { title: '6. 学習ベース制御・sim-to-real', href: '/physical-ai/06-learning-based-control-sim2real/' },
      { title: '7. シミュレーション環境とハードウェア', href: '/physical-ai/07-simulation-and-hardware/' },
      { title: '8. VLA・ロボット基盤モデル（π0 等）', href: '/physical-ai/08-vla-foundation-models/' },
    ],
  },
  {
    domain: '🔀 マルチモーダル',
    items: [
      { title: 'ロードマップ', href: '/multimodal/' },
      { title: '1. 共通表現と対照学習（CLIP）', href: '/multimodal/01-contrastive-clip/' },
      { title: '2. Vision-Language モデル', href: '/multimodal/02-vision-language-models/' },
      { title: '3. マルチモーダル生成', href: '/multimodal/03-multimodal-generation/' },
      { title: '4. any-to-any・omni', href: '/multimodal/04-any-to-any-omni/' },
      { title: '5. 現代マルチモーダルの地図', href: '/multimodal/05-multimodal-landscape/' },
    ],
  },
  {
    domain: '🎯 強化学習（横断的学習パラダイム）',
    items: [
      { title: 'ロードマップ', href: '/reinforcement-learning/' },
      { title: '1. 強化学習とは（MDP）', href: '/reinforcement-learning/01-mdp/' },
      { title: '2. 動的計画法', href: '/reinforcement-learning/02-dynamic-programming/' },
      { title: '3. モデルフリー予測（MC・TD）', href: '/reinforcement-learning/03-model-free-prediction/' },
      { title: '4. モデルフリー制御（SARSA・Q学習）', href: '/reinforcement-learning/04-model-free-control/' },
      { title: '5. 関数近似と DQN', href: '/reinforcement-learning/05-function-approximation-dqn/' },
      { title: '6. 方策勾配法', href: '/reinforcement-learning/06-policy-gradient/' },
      { title: '7. Actor-Critic', href: '/reinforcement-learning/07-actor-critic/' },
      { title: '8. 現代の深層強化学習（地図）', href: '/reinforcement-learning/08-modern-deep-rl/' },
    ],
  },
]
