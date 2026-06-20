// ============================================================================
// 領域固有の Canvas 図のレジストリ。
//
// なぜこの方式か:
//   navigation.instant（ページを再読み込みせずに遷移する機能）が有効なため、
//   ページ本文に書いた inline <script> はリンク遷移では再実行されない。
//   そこで MathJax と同じく document$（Material/zensical が提供する遷移イベント）を
//   購読し、各ページ表示のたびに該当 canvas を探して描き直す。
//
// 図を追加する手順:
//   1) 章の Markdown に <canvas class="tb-fig" id="<一意なid>"> を置く。
//   2) このファイルの FIGURES に { "<id>": draw関数 } を 1 行追加する。
//   3) draw関数は決定的・静的（乱数/アニメ禁止。擬似乱数は固定 seed なら可）、
//      依存ゼロ、配色はトークンで。
// ============================================================================

(function () {
  var TEAL = "#0B6E78", ORANGE = "#DD6A2B", INK = "#14191F", HAIR = "#CDD5DD", MUTED = "#5A6573";

  var FIGURES = {
    "alias-fig": drawAliasFig,     // audio/01: aliasing
    "mel-fbank": drawMelFbank,     // audio/02: メルフィルタバンク
    "vq-snap": drawVqSnap,         // audio/03: VQ 最近傍スナップ
    "rnnt-lattice": drawRnntLattice, // audio/04: RNN-T 格子
    "attn-heat": drawAttnHeat,     // audio/04: attention ヒートマップ
    "chunk-mask": drawChunkMask,   // audio/04: chunk attention マスク
  };

  function redraw() {
    for (var id in FIGURES) {
      var c = document.getElementById(id);
      if (c && c.getContext) FIGURES[id](c);
    }
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(redraw);                 // instant navigation を含む各表示で再描画
  } else {
    document.addEventListener("DOMContentLoaded", redraw);
  }

  // --- 図の定義 -------------------------------------------------------------

  // audio/01: fs=1000Hz で 100Hz と 1100Hz が標本点で一致する（aliasing）。
  function drawAliasFig(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height;
    var ml = 18, mr = 18, mt = 22, mb = 40;
    var plotW = W - ml - mr, midY = mt + (H - mt - mb) / 2, amp = (H - mt - mb) / 2 - 8;
    var Twin = 0.006, fs = 1000, f1 = 100, f2 = 1100; // 6 ms 窓
    function X(t) { return ml + (t / Twin) * plotW; }
    function Y(v) { return midY - v * amp; }
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = HAIR; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(ml, midY); ctx.lineTo(W - mr, midY); ctx.stroke();
    function curve(f, color, width) {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      for (var px = 0; px <= plotW; px++) {
        var t = (px / plotW) * Twin, y = Y(Math.sin(2 * Math.PI * f * t));
        if (px === 0) ctx.moveTo(ml + px, y); else ctx.lineTo(ml + px, y);
      }
      ctx.stroke();
    }
    curve(f2, ORANGE, 2);
    curve(f1, TEAL, 3.5);
    for (var k = 0; k <= 6; k++) {
      var t = k / fs, x = X(t), v = Math.sin(2 * Math.PI * f1 * t);
      ctx.strokeStyle = "rgba(90,101,115,0.35)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, mt); ctx.lineTo(x, H - mb); ctx.stroke();
      ctx.fillStyle = INK;
      ctx.beginPath(); ctx.arc(x, Y(v), 6, 0, 7); ctx.fill();
    }
    ctx.font = "22px ui-monospace, Menlo, monospace";
    ctx.fillStyle = TEAL; ctx.fillText("100 Hz", ml + 10, mt + 24);
    ctx.fillStyle = ORANGE; ctx.fillText("1100 Hz", ml + 10, mt + 50);
  }

  // audio/02: メル軸上で等間隔に並ぶ三角フィルタ（低域は密、高域は粗）。
  function drawMelFbank(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height;
    var FMAX = 8000, NMEL = 9, HL = 4; // 強調するフィルタ番号
    function hz2mel(f) { return 2595 * Math.log(1 + f / 700) / Math.LN10; }
    function mel2hz(m) { return 700 * (Math.pow(10, m / 2595) - 1); }
    var ml = 10, mr = 14, mt = 26, mb = 56;
    var plotW = W - ml - mr, top = mt, base = H - mb;
    function fx(f) { return ml + (f / FMAX) * plotW; }
    var pts = [], melMin = hz2mel(0), melMax = hz2mel(FMAX);
    for (var i = 0; i < NMEL + 2; i++) pts.push(mel2hz(melMin + (melMax - melMin) * i / (NMEL + 1)));
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "#9aa6b2"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(ml, base); ctx.lineTo(W - mr, base); ctx.stroke();
    ctx.fillStyle = MUTED; ctx.font = "22px ui-monospace, Menlo, monospace"; ctx.textAlign = "center";
    [0, 1000, 2000, 4000, 6000, 8000].forEach(function (f) {
      var x = fx(f);
      ctx.strokeStyle = "#c2ccd5"; ctx.beginPath(); ctx.moveTo(x, base); ctx.lineTo(x, base + 7); ctx.stroke();
      ctx.fillText(f === 0 ? "0" : (f / 1000) + "k", x, base + 32);
    });
    ctx.textAlign = "left"; ctx.fillText("Hz", W - mr - 4, base + 32);
    for (var m = 0; m < NMEL; m++) {
      var l = pts[m], cen = pts[m + 1], r = pts[m + 2], on = (m === HL);
      ctx.beginPath();
      ctx.moveTo(fx(l), base); ctx.lineTo(fx(cen), top); ctx.lineTo(fx(r), base);
      if (on) { ctx.closePath(); ctx.fillStyle = "rgba(11,110,120,0.20)"; ctx.fill(); }
      ctx.lineWidth = on ? 3 : 1.6;
      ctx.strokeStyle = on ? TEAL : "rgba(90,101,115,0.55)";
      ctx.stroke();
      if (on) {
        ctx.fillStyle = TEAL;
        ctx.beginPath(); ctx.arc(fx(cen), top, 5, 0, 7); ctx.fill();
        ctx.textAlign = "center"; ctx.fillText("重み=1", fx(cen), top - 8);
      }
    }
    ctx.fillStyle = MUTED; ctx.textAlign = "left"; ctx.fillText("重み", ml + 2, top + 4);
  }

  // audio/03: 連続な潜在ベクトル（点）が最寄りのコードワード（◆）へ丸められる。
  function drawVqSnap(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    var pad = 40;
    var cols = ["#0B6E78", "#DD6A2B", "#6A1F8E", "#C0317A", "#2B8C5A"];
    var codes = [[0.24, 0.30], [0.70, 0.22], [0.52, 0.62], [0.30, 0.74], [0.80, 0.66]];
    function X(u) { return pad + u * (W - 2 * pad); }
    function Y(v) { return pad + v * (H - 2 * pad); }
    var pts = [], seed = 7;
    function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return (seed / 0x7fffffff); }
    codes.forEach(function (cd) {
      for (var i = 0; i < 26; i++) pts.push([cd[0] + (rnd() - 0.5) * 0.26, cd[1] + (rnd() - 0.5) * 0.26]);
    });
    pts.forEach(function (p) {
      var best = 0, bd = 9;
      for (var i = 0; i < codes.length; i++) { var dx = p[0] - codes[i][0], dy = p[1] - codes[i][1], d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } }
      ctx.strokeStyle = cols[best] + "33"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(X(p[0]), Y(p[1])); ctx.lineTo(X(codes[best][0]), Y(codes[best][1])); ctx.stroke();
      ctx.fillStyle = cols[best] + "cc";
      ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5, 0, 7); ctx.fill();
    });
    codes.forEach(function (cd, i) {
      var x = X(cd[0]), y = Y(cd[1]), s = 13;
      ctx.fillStyle = cols[i]; ctx.strokeStyle = INK; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath();
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = INK; ctx.font = "bold 26px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
      ctx.fillText("e" + i, x, y - s - 10);
    });
  }

  // audio/04: RNN-T 格子（横=時間 t / 縦=出力 u）と単調な経路（CAT）。
  function drawRnntLattice(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var T = 9, U = 4, ml = 70, mb = 60, mt = 30, mr = 30;
    var gw = (W - ml - mr) / T, gh = (H - mt - mb) / U;
    function X(t) { return ml + t * gw; }
    function Y(u) { return H - mb - u * gh; }
    ctx.strokeStyle = "#d5dde4"; ctx.lineWidth = 1.5;
    for (var t = 0; t <= T; t++) { ctx.beginPath(); ctx.moveTo(X(t), Y(0)); ctx.lineTo(X(t), Y(U)); ctx.stroke(); }
    for (var u = 0; u <= U; u++) { ctx.beginPath(); ctx.moveTo(X(0), Y(u)); ctx.lineTo(X(T), Y(u)); ctx.stroke(); }
    var path = [[0, 0], [1, 0], [2, 0], [2, 1], [3, 1], [4, 1], [4, 2], [5, 2], [6, 2], [6, 3], [7, 3], [8, 3], [9, 3]];
    ctx.strokeStyle = TEAL; ctx.lineWidth = 5; ctx.beginPath();
    path.forEach(function (p, i) { if (i) ctx.lineTo(X(p[0]), Y(p[1])); else ctx.moveTo(X(p[0]), Y(p[1])); }); ctx.stroke();
    path.forEach(function (p) { ctx.fillStyle = TEAL; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 7, 0, 7); ctx.fill(); });
    ctx.fillStyle = MUTED; ctx.font = "26px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
    for (var t2 = 0; t2 < T; t2++) ctx.fillText("t" + t2, X(t2) + gw / 2, H - mb + 34);
    ctx.textAlign = "right";
    ["∅", "C", "A", "T"].forEach(function (s, u2) { ctx.fillText(s, ml - 14, Y(u2) - gh / 2 + 9); });
    ctx.save(); ctx.translate(20, H / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillStyle = TEAL;
    ctx.fillText("出力 u →", 0, 0); ctx.restore();
  }

  // audio/04: cross-attention の重み（ほぼ対角＝単調）。
  function drawAttnHeat(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var T = 18, U = 6, ml = 70, mt = 20, mb = 46, mr = 20;
    var cw = (W - ml - mr) / T, ch = (H - mt - mb) / U;
    for (var u = 0; u < U; u++) for (var t = 0; t < T; t++) {
      var center = (u + 0.5) * T / U;
      var v = Math.exp(-Math.pow((t - center) / 1.6, 2));
      ctx.fillStyle = "rgba(11,110,120," + (0.06 + v * 0.9) + ")";
      ctx.fillRect(ml + t * cw, mt + (U - 1 - u) * ch, cw - 2, ch - 2);
    }
    ctx.fillStyle = MUTED; ctx.font = "24px ui-monospace,Menlo,monospace"; ctx.textAlign = "right";
    ["C", "A", "T", "S", ".", "␣"].forEach(function (s, u2) { ctx.fillText(s, ml - 12, mt + (U - 1 - u2) * ch + ch / 2 + 8); });
    ctx.textAlign = "center"; ctx.fillText("入力フレーム t →", W / 2, H - 12);
  }

  // audio/04: chunk attention mask（塊内＋過去は参照可・未来は不可）。
  function drawChunkMask(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var T = 16, chunk = 4, pad = 20, s = (W - 2 * pad) / T;
    for (var q = 0; q < T; q++) for (var k = 0; k < T; k++) {
      var qChunk = Math.floor(q / chunk), kChunk = Math.floor(k / chunk);
      var allowed = kChunk <= qChunk;
      ctx.fillStyle = allowed ? "rgba(11,110,120," + (kChunk === qChunk ? 0.85 : 0.4) + ")" : "#eef1f4";
      ctx.strokeStyle = "#d5dde4";
      var x = pad + k * s, y = pad + q * s;
      ctx.fillRect(x, y, s - 1, s - 1); ctx.strokeRect(x, y, s - 1, s - 1);
    }
    ctx.strokeStyle = ORANGE; ctx.lineWidth = 2;
    for (var b = 0; b <= T; b += chunk) {
      var p = pad + b * s;
      ctx.beginPath(); ctx.moveTo(p, pad); ctx.lineTo(p, H - pad); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad, p); ctx.lineTo(W - pad, p); ctx.stroke();
    }
    ctx.fillStyle = MUTED; ctx.font = "28px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
    ctx.fillText("key（参照先 t）→", W / 2, H - 2);
  }
})();
