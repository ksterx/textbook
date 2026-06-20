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
    items: [{ title: 'ロードマップ', href: '/vision/' }],
  },
  {
    domain: '🦾 身体性・行動（Physical AI）',
    items: [{ title: 'ロードマップ', href: '/physical-ai/' }],
  },
  {
    domain: '🔀 マルチモーダル',
    items: [{ title: 'ロードマップ', href: '/multimodal/' }],
  },
  {
    domain: '🎯 強化学習（横断的学習パラダイム）',
    items: [{ title: 'ロードマップ', href: '/reinforcement-learning/' }],
  },
]
