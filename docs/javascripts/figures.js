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
//   3) draw関数は決定的・静的（乱数/アニメ禁止）、依存ゼロ、配色はトークンで。
// ============================================================================

(function () {
  var TEAL = "#0B6E78", ORANGE = "#DD6A2B", INK = "#14191F", HAIR = "#CDD5DD";

  var FIGURES = {
    "alias-fig": drawAliasFig,
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
    // 中心線
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
    curve(f2, ORANGE, 2);    // 1100 Hz（速い）
    curve(f1, TEAL, 3.5);    // 100 Hz（ゆっくり）
    // 標本点（縦線＋共有する●）
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
})();
