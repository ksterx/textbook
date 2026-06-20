// サイトのナビゲーション定義（分野＝セクション）。章を足したらここに1行追加する。
// href はルート絶対（base なし）。レイアウトで BASE_URL を前置する。
export const NAV = [
  {
    domain: 'Audio（音声・音響）',
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
    domain: '強化学習',
    items: [{ title: 'ロードマップ', href: '/reinforcement-learning/' }],
  },
  {
    domain: 'LLM',
    items: [{ title: 'ロードマップ', href: '/llm/' }],
  },
  {
    domain: 'Physical AI',
    items: [{ title: 'ロードマップ', href: '/physical-ai/' }],
  },
];
