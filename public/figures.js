// 領域固有の Canvas 図。Astro はページ全体をリロードするので DOMContentLoaded で一度描けばよい。
(function () {
  var TEAL = "#0B6E78", ORANGE = "#DD6A2B", INK = "#14191F", HAIR = "#CDD5DD", MUTED = "#5A6573";

  var FIGURES = {
    "alias-fig": drawAliasFig,
    "mel-fbank": drawMelFbank,
    "vq-snap": drawVqSnap,
    "rnnt-lattice": drawRnntLattice,
    "attn-heat": drawAttnHeat,
    "chunk-mask": drawChunkMask,
    "flow-vs-diffusion": drawFlowVsDiffusion,
    "euler-steps": drawEulerSteps,
    "conv-slide": drawConvSlide,
    "clip-matrix": drawClipMatrix,
    "pid-response": drawPidResponse,
    "gridworld-value": drawGridworldValue,
    "scaling-law": drawScalingLaw,
    "sway-sampling": drawSwaySampling,
    "nemotron-latency-wer": drawNemotronLatencyWer,
    "nemotron-throughput": drawNemotronThroughput,
  };

  function redraw() {
    for (var id in FIGURES) {
      var c = document.getElementById(id);
      if (c && c.getContext) FIGURES[id](c);
    }
  }
  if (document.readyState !== "loading") redraw();
  else document.addEventListener("DOMContentLoaded", redraw);

  // audio/01: aliasing（100Hz と 1100Hz が標本点で一致）
  function drawAliasFig(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height;
    var ml = 18, mr = 18, mt = 22, mb = 40;
    var plotW = W - ml - mr, midY = mt + (H - mt - mb) / 2, amp = (H - mt - mb) / 2 - 8;
    var Twin = 0.006, fs = 1000, f1 = 100, f2 = 1100;
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
    curve(f2, ORANGE, 2); curve(f1, TEAL, 3.5);
    for (var k = 0; k <= 6; k++) {
      var t = k / fs, x = X(t), v = Math.sin(2 * Math.PI * f1 * t);
      ctx.strokeStyle = "rgba(90,101,115,0.35)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, mt); ctx.lineTo(x, H - mb); ctx.stroke();
      ctx.fillStyle = INK; ctx.beginPath(); ctx.arc(x, Y(v), 6, 0, 7); ctx.fill();
    }
    ctx.font = "22px ui-monospace, Menlo, monospace";
    ctx.fillStyle = TEAL; ctx.fillText("100 Hz", ml + 10, mt + 24);
    ctx.fillStyle = ORANGE; ctx.fillText("1100 Hz", ml + 10, mt + 50);
  }

  // audio/02: メルフィルタバンク
  function drawMelFbank(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height;
    var FMAX = 8000, NMEL = 9, HL = 4;
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
      ctx.beginPath(); ctx.moveTo(fx(l), base); ctx.lineTo(fx(cen), top); ctx.lineTo(fx(r), base);
      if (on) { ctx.closePath(); ctx.fillStyle = "rgba(11,110,120,0.20)"; ctx.fill(); }
      ctx.lineWidth = on ? 3 : 1.6; ctx.strokeStyle = on ? TEAL : "rgba(90,101,115,0.55)"; ctx.stroke();
      if (on) {
        ctx.fillStyle = TEAL; ctx.beginPath(); ctx.arc(fx(cen), top, 5, 0, 7); ctx.fill();
        ctx.textAlign = "center"; ctx.fillText("重み=1", fx(cen), top - 8);
      }
    }
    ctx.fillStyle = MUTED; ctx.textAlign = "left"; ctx.fillText("重み", ml + 2, top + 4);
  }

  // audio/03: VQ 最近傍スナップ
  function drawVqSnap(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var pad = 40;
    var cols = ["#0B6E78", "#DD6A2B", "#6A1F8E", "#C0317A", "#2B8C5A"];
    var codes = [[0.24, 0.30], [0.70, 0.22], [0.52, 0.62], [0.30, 0.74], [0.80, 0.66]];
    function X(u) { return pad + u * (W - 2 * pad); }
    function Y(v) { return pad + v * (H - 2 * pad); }
    var pts = [], seed = 7;
    function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return (seed / 0x7fffffff); }
    codes.forEach(function (cd) { for (var i = 0; i < 26; i++) pts.push([cd[0] + (rnd() - 0.5) * 0.26, cd[1] + (rnd() - 0.5) * 0.26]); });
    pts.forEach(function (p) {
      var best = 0, bd = 9;
      for (var i = 0; i < codes.length; i++) { var dx = p[0] - codes[i][0], dy = p[1] - codes[i][1], d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } }
      ctx.strokeStyle = cols[best] + "33"; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(X(p[0]), Y(p[1])); ctx.lineTo(X(codes[best][0]), Y(codes[best][1])); ctx.stroke();
      ctx.fillStyle = cols[best] + "cc"; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5, 0, 7); ctx.fill();
    });
    codes.forEach(function (cd, i) {
      var x = X(cd[0]), y = Y(cd[1]), s = 13;
      ctx.fillStyle = cols[i]; ctx.strokeStyle = INK; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.fillStyle = INK; ctx.font = "bold 26px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
      ctx.fillText("e" + i, x, y - s - 10);
    });
  }

  // audio/04: RNN-T 格子
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

  // audio/04: attention ヒートマップ
  function drawAttnHeat(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var T = 18, U = 6, ml = 70, mt = 20, mb = 46, mr = 20;
    var cw = (W - ml - mr) / T, ch = (H - mt - mb) / U;
    for (var u = 0; u < U; u++) for (var t = 0; t < T; t++) {
      var center = (u + 0.5) * T / U, v = Math.exp(-Math.pow((t - center) / 1.6, 2));
      ctx.fillStyle = "rgba(11,110,120," + (0.06 + v * 0.9) + ")";
      ctx.fillRect(ml + t * cw, mt + (U - 1 - u) * ch, cw - 2, ch - 2);
    }
    ctx.fillStyle = MUTED; ctx.font = "24px ui-monospace,Menlo,monospace"; ctx.textAlign = "right";
    ["C", "A", "T", "S", ".", "␣"].forEach(function (s, u2) { ctx.fillText(s, ml - 12, mt + (U - 1 - u2) * ch + ch / 2 + 8); });
    ctx.textAlign = "center"; ctx.fillText("入力フレーム t →", W / 2, H - 12);
  }

  // audio/04: chunk attention マスク
  function drawChunkMask(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var T = 16, chunk = 4, pad = 20, s = (W - 2 * pad) / T;
    for (var q = 0; q < T; q++) for (var k = 0; k < T; k++) {
      var qChunk = Math.floor(q / chunk), kChunk = Math.floor(k / chunk), allowed = kChunk <= qChunk;
      ctx.fillStyle = allowed ? "rgba(11,110,120," + (kChunk === qChunk ? 0.85 : 0.4) + ")" : "#eef1f4";
      ctx.strokeStyle = "#d5dde4";
      var x = pad + k * s, y = pad + q * s;
      ctx.fillRect(x, y, s - 1, s - 1); ctx.strokeRect(x, y, s - 1, s - 1);
    }
    ctx.strokeStyle = ORANGE; ctx.lineWidth = 2;
    for (var b = 0; b <= T; b += chunk) { var p = pad + b * s; ctx.beginPath(); ctx.moveTo(p, pad); ctx.lineTo(p, H - pad); ctx.stroke(); ctx.beginPath(); ctx.moveTo(pad, p); ctx.lineTo(W - pad, p); ctx.stroke(); }
    ctx.fillStyle = MUTED; ctx.font = "28px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
    ctx.fillText("key（参照先 t）→", W / 2, H - 2);
  }

  // audio/07: diffusion（曲がった道・多ステップ）vs flow matching（OT 直線・少ステップ）
  // 同じノイズ x0 から同じデータ x1 へ運ぶが、軌道の「形」と必要ステップ数が違うことを対比する。
  function drawFlowVsDiffusion(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var seed = 7;
    function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
    function gss() { return rnd() + rnd() + rnd() + rnd() - 2; } // 近似ガウス

    var gap = 70, panelW = (W - gap) / 2;

    // 両パネル共通：ノイズ点の縦位置と、その対応データ点（上下2クラスタ）。始点・終点は左右で同一。
    var N = 6, src = [], tgt = [];
    for (var i = 0; i < N; i++) { src.push(0.5 + gss() * 0.09); tgt.push(i % 2 === 0 ? 0.30 : 0.70); }
    var srcCloud = [], tgtCloud = [];
    for (var k = 0; k < 80; k++) srcCloud.push(0.5 + gss() * 0.11);
    for (var k = 0; k < 50; k++) tgtCloud.push(k % 2 === 0 ? 0.30 + gss() * 0.03 : 0.70 + gss() * 0.03);

    function vAt(t, sy, ty, dir, curved, i) {
      var v = sy + (ty - sy) * t;
      if (curved) v += Math.sin(Math.PI * t) * 0.17 * dir + Math.sin(Math.PI * 5 * t + i) * 0.025;
      return v;
    }

    function panel(ox, name, sub, curved, steps, col) {
      var bx0 = ox + 40, bx1 = ox + panelW - 40, byT = 110, byB = H - 100;
      var L = bx0 + 78, R = bx1 - 86;
      function X(u) { return L + u * (R - L); }
      function Y(v) { return byT + 34 + v * (byB - byT - 68); }

      ctx.strokeStyle = HAIR; ctx.lineWidth = 1.5; ctx.strokeRect(bx0, byT, bx1 - bx0, byB - byT);
      ctx.textAlign = "center";
      ctx.fillStyle = col; ctx.font = "bold 34px ui-monospace,Menlo,monospace"; ctx.fillText(name, ox + panelW / 2, 50);
      ctx.fillStyle = MUTED; ctx.font = "27px ui-monospace,Menlo,monospace"; ctx.fillText(sub, ox + panelW / 2, 90);

      ctx.fillStyle = "rgba(90,101,115,0.32)";
      for (var k = 0; k < srcCloud.length; k++) { ctx.beginPath(); ctx.arc(X(0) + gss() * 11, Y(srcCloud[k]), 5, 0, 7); ctx.fill(); }
      ctx.fillStyle = "rgba(11,110,120,0.34)";
      for (var k = 0; k < tgtCloud.length; k++) { ctx.beginPath(); ctx.arc(X(1) + gss() * 11, Y(tgtCloud[k]), 5, 0, 7); ctx.fill(); }

      for (var i = 0; i < N; i++) {
        var sy = src[i], ty = tgt[i], dir = (i % 2 ? 1 : -1);
        ctx.strokeStyle = curved ? "rgba(221,106,43,0.8)" : "rgba(11,110,120,0.8)"; ctx.lineWidth = 3;
        ctx.beginPath();
        for (var s = 0; s <= 80; s++) { var t = s / 80, px = X(t), py = Y(vAt(t, sy, ty, dir, curved, i)); if (s === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
        ctx.stroke();
        ctx.fillStyle = col;
        for (var s2 = 0; s2 <= steps; s2++) { var t2 = s2 / steps; ctx.beginPath(); ctx.arc(X(t2), Y(vAt(t2, sy, ty, dir, curved, i)), curved ? 6 : 9, 0, 7); ctx.fill(); }
        if (!curved) {
          var t0 = 0.5, ang = Math.atan2(Y(ty) - Y(sy), R - L), hx = X(t0), hy = Y(vAt(t0, sy, ty, dir, curved, i)), a = 21;
          ctx.beginPath();
          ctx.moveTo(hx + Math.cos(ang) * a, hy + Math.sin(ang) * a);
          ctx.lineTo(hx + Math.cos(ang + 2.5) * a, hy + Math.sin(ang + 2.5) * a);
          ctx.lineTo(hx + Math.cos(ang - 2.5) * a, hy + Math.sin(ang - 2.5) * a);
          ctx.closePath(); ctx.fill();
        }
      }
      ctx.fillStyle = MUTED; ctx.font = "26px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
      ctx.fillText("x0 ~ N(0,I)", X(0), byB + 38);
      ctx.fillText("x1（データ = mel）", X(1), byB + 38);
      ctx.fillStyle = HAIR; ctx.fillText("t: 0 → 1", ox + panelW / 2, byB + 74);
    }

    panel(0, "diffusion", "曲がった道・多ステップ", true, 16, ORANGE);
    panel(panelW + gap, "flow matching (OT)", "まっすぐ・少ステップ", false, 5, TEAL);
  }

  // audio/07: ODE を Euler で積分する「1ステップの中身」。各点で速度を評価し、Δt だけ進む、を N 回。
  function drawEulerSteps(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var ml = 90, mr = 90, mt = 156, mb = 100;
    var plotW = W - ml - mr;
    var midY = (mt + (H - mb)) / 2, amp = (H - mt - mb) / 2 - 56;
    function cx(t) { return ml + t * plotW; }
    function cy(t) { return midY + amp * Math.cos(Math.PI * t); } // 左下→右上の弧（曲がった一般のODE経路）
    var N = 5;
    var P = [];
    for (var k = 0; k <= N; k++) { var t = k / N; P.push([cx(t), cy(t)]); }

    // 本来の連続経路（薄い破線）。折れ線（N ステップ）はこれを近似する。
    ctx.strokeStyle = HAIR; ctx.lineWidth = 2.5; ctx.setLineDash([9, 9]); ctx.beginPath();
    for (var s = 0; s <= 120; s++) { var tt = s / 120, x = cx(tt), y = cy(tt); if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
    ctx.stroke(); ctx.setLineDash([]);

    // 各ステップ：teal セグメント（Δt·v だけ進む）＋ orange 矢印（その点で評価した速度の向き）
    for (var k2 = 0; k2 < N; k2++) {
      var a = P[k2], b = P[k2 + 1];
      ctx.strokeStyle = "rgba(11,110,120,0.9)"; ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      var ang = Math.atan2(b[1] - a[1], b[0] - a[0]), r = 20;
      ctx.fillStyle = ORANGE; ctx.beginPath();
      ctx.moveTo(b[0] + Math.cos(ang) * r, b[1] + Math.sin(ang) * r);
      ctx.lineTo(b[0] + Math.cos(ang + 2.5) * r, b[1] + Math.sin(ang + 2.5) * r);
      ctx.lineTo(b[0] + Math.cos(ang - 2.5) * r, b[1] + Math.sin(ang - 2.5) * r);
      ctx.closePath(); ctx.fill();
    }

    // 点と t ラベル
    for (var k3 = 0; k3 <= N; k3++) {
      ctx.fillStyle = INK; ctx.beginPath(); ctx.arc(P[k3][0], P[k3][1], 10, 0, 7); ctx.fill();
      ctx.fillStyle = MUTED; ctx.font = "23px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
      ctx.fillText("t=" + (k3 / N).toFixed(1), P[k3][0], H - mb + 40);
    }
    ctx.fillStyle = INK; ctx.font = "bold 27px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
    ctx.fillText("x0（ノイズ）", P[0][0] + 30, P[0][1] + 44);
    ctx.fillText("x1（mel）", P[N][0] - 20, P[N][1] - 26);

    // 上部の手順注記
    ctx.textAlign = "left"; ctx.font = "27px ui-monospace,Menlo,monospace";
    ctx.fillStyle = ORANGE; ctx.fillText("① 今いる点で速度 vθ(x, t, c) を1回評価（向き = 矢印）", ml, 54);
    ctx.fillStyle = TEAL; ctx.fillText("② x ← x + Δt·v でほんの少し進む → 次の点でまた評価", ml, 96);
    ctx.fillStyle = MUTED; ctx.fillText("これを N 回（= NFE）。破線 = 本来の連続経路、折れ線 = N ステップの近似", ml, 138);
  }

  // vision/01: 畳み込み — 3×3 カーネルを入力で滑らせ、窓の9マスが出力の1マスになる（重み共有）
  function drawConvSlide(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var n = 7, m = 5, cs = 72, g = 2, ix = 70, iy = 150, ox = ix + n * cs + 250, oy = iy + cs;
    function cell(x, y, fill, stroke, lw) { ctx.fillStyle = fill; ctx.fillRect(x, y, cs - g, cs - g); ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.strokeRect(x, y, cs - g, cs - g); }
    for (var r = 0; r < n; r++) for (var col = 0; col < n; col++) cell(ix + col * cs, iy + r * cs, "#eef1f4", HAIR, 1);
    var kr = 1, kc = 2;
    for (var a = 0; a < 3; a++) for (var b = 0; b < 3; b++) cell(ix + (kc + b) * cs, iy + (kr + a) * cs, "rgba(221,106,43,0.30)", ORANGE, 3);
    for (var r2 = 0; r2 < m; r2++) for (var col2 = 0; col2 < m; col2++) cell(ox + col2 * cs, oy + r2 * cs, "#eef1f4", HAIR, 1);
    var oxc = ox + kc * cs, oyc = oy + kr * cs; cell(oxc, oyc, "rgba(11,110,120,0.85)", TEAL, 3);
    var ocx = oxc + cs / 2, ocy = oyc + cs / 2, corners = [[kc, kr], [kc + 3, kr], [kc, kr + 3], [kc + 3, kr + 3]];
    ctx.strokeStyle = "rgba(11,110,120,0.45)"; ctx.lineWidth = 2;
    for (var i = 0; i < 4; i++) { ctx.beginPath(); ctx.moveTo(ix + corners[i][0] * cs, iy + corners[i][1] * cs); ctx.lineTo(ocx, ocy); ctx.stroke(); }
    ctx.textAlign = "center"; ctx.fillStyle = MUTED; ctx.font = "bold 28px ui-monospace,Menlo,monospace";
    ctx.fillText("入力 (7×7)", ix + n * cs / 2, iy - 24); ctx.fillText("出力 = 特徴マップ (5×5)", ox + m * cs / 2, iy - 24);
    ctx.fillStyle = ORANGE; ctx.font = "26px ui-monospace,Menlo,monospace"; ctx.fillText("3×3 カーネル（9個の重み）を滑らせる", ix + n * cs / 2, iy + n * cs + 44);
    ctx.fillStyle = TEAL; ctx.fillText("窓の9マス → 出力1マス（重みは全位置で共有）", ox + m * cs / 2, oy + m * cs + 44);
  }

  // multimodal/01: CLIP の類似度行列 — 対角＝正しいペア（近づける）、非対角＝負例（遠ざける）
  function drawClipMatrix(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var n = 5, cs = 130, x0 = 430, y0 = 170, seed = 3;
    function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) {
      var diag = i === j, sim = diag ? 0.85 + rnd() * 0.1 : 0.05 + rnd() * 0.18, x = x0 + j * cs, y = y0 + i * cs;
      ctx.fillStyle = "rgba(11,110,120," + sim.toFixed(2) + ")"; ctx.fillRect(x, y, cs - 4, cs - 4);
      ctx.strokeStyle = diag ? ORANGE : HAIR; ctx.lineWidth = diag ? 4 : 1; ctx.strokeRect(x, y, cs - 4, cs - 4);
      ctx.fillStyle = sim > 0.5 ? "#fff" : MUTED; ctx.font = "22px ui-monospace,Menlo,monospace"; ctx.fillText(sim.toFixed(2), x + (cs - 4) / 2, y + (cs - 4) / 2);
    }
    ctx.fillStyle = INK; ctx.font = "bold 24px ui-monospace,Menlo,monospace";
    for (var k = 0; k < n; k++) {
      ctx.fillText("テキスト" + (k + 1), x0 + k * cs + (cs - 4) / 2, y0 - 34);
      ctx.save(); ctx.translate(x0 - 46, y0 + k * cs + (cs - 4) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("画像" + (k + 1), 0, 0); ctx.restore();
    }
    ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.font = "25px ui-monospace,Menlo,monospace";
    ctx.fillStyle = ORANGE; ctx.fillText("● 対角（画像 i × テキスト i）＝正しいペア → 類似度を上げる", x0, y0 + n * cs + 44);
    ctx.fillStyle = MUTED; ctx.fillText("● 非対角＝バッチ内の他人ペア（負例）→ 類似度を下げる", x0, y0 + n * cs + 82);
  }

  // physical-ai/03: P / PI / PID のステップ応答（目標へどう収束するか）
  function drawPidResponse(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var ml = 90, mr = 360, mt = 70, mb = 90, pw = W - ml - mr, ph = H - mt - mb, E = Math.exp;
    function X(t) { return ml + t * pw; } function Y(v) { return mt + ph - (v / 1.5) * ph; }
    ctx.strokeStyle = HAIR; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(ml, mt); ctx.lineTo(ml, mt + ph); ctx.lineTo(ml + pw, mt + ph); ctx.stroke();
    ctx.strokeStyle = MUTED; ctx.setLineDash([10, 8]); ctx.beginPath(); ctx.moveTo(ml, Y(1)); ctx.lineTo(ml + pw, Y(1)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = MUTED; ctx.font = "24px ui-monospace,Menlo,monospace"; ctx.textAlign = "left"; ctx.fillText("目標値", ml + 14, Y(1) - 12);
    function plot(f, col, w) { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); for (var s = 0; s <= 200; s++) { var t = s / 200, x = X(t), y = Y(f(t)); if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); } ctx.stroke(); }
    plot(function (t) { return 0.7 * (1 - E(-4.5 * t)); }, "rgba(90,101,115,0.9)", 3.5);
    plot(function (t) { return 1 - E(-2.3 * t) * Math.cos(7.0 * t); }, ORANGE, 3.5);
    plot(function (t) { return 1 - E(-4.2 * t) * (1 + 4.2 * t); }, TEAL, 4.5);
    ctx.font = "bold 26px ui-monospace,Menlo,monospace"; ctx.textAlign = "left";
    ctx.fillStyle = MUTED; ctx.fillText("P：定常偏差が残る", ml + pw + 12, mt + 70);
    ctx.fillStyle = ORANGE; ctx.fillText("PI：行き過ぎ＋振動", ml + pw + 12, mt + 112);
    ctx.fillStyle = TEAL; ctx.fillText("PID：なめらかに整定", ml + pw + 12, mt + 154);
    ctx.fillStyle = MUTED; ctx.font = "24px ui-monospace,Menlo,monospace"; ctx.textAlign = "center"; ctx.fillText("時間 →", ml + pw / 2, mt + ph + 52);
  }

  // rl/02: 価値反復が「ゴールから価値を伝播」する（色＝状態価値・矢印＝貪欲方策）
  function drawGridworldValue(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var n = 5, cs = 150, x0 = (W - n * cs) / 2, y0 = 110, gr = 0, gc = 4;
    function val(r, col) { return Math.pow(0.85, Math.abs(r - gr) + Math.abs(col - gc)); }
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (var r = 0; r < n; r++) for (var col = 0; col < n; col++) {
      var v = val(r, col), x = x0 + col * cs, y = y0 + r * cs, cx = x + (cs - 3) / 2, cy = y + (cs - 3) / 2;
      ctx.fillStyle = "rgba(11,110,120," + (0.12 + 0.8 * v).toFixed(2) + ")"; ctx.fillRect(x, y, cs - 3, cs - 3);
      ctx.strokeStyle = HAIR; ctx.lineWidth = 1; ctx.strokeRect(x, y, cs - 3, cs - 3);
      if (r === gr && col === gc) { ctx.fillStyle = "#fff"; ctx.font = "bold 40px ui-monospace,Menlo,monospace"; ctx.fillText("G", cx, cy); continue; }
      ctx.fillStyle = v > 0.4 ? "#fff" : MUTED; ctx.font = "22px ui-monospace,Menlo,monospace"; ctx.fillText(v.toFixed(2), cx, cy + 36);
      var best = null, bv = -1;
      [[-1, 0], [1, 0], [0, -1], [0, 1]].forEach(function (d) { var nr = r + d[0], nc = col + d[1]; if (nr >= 0 && nr < n && nc >= 0 && nc < n) { var vv = val(nr, nc); if (vv > bv) { bv = vv; best = d; } } });
      if (best) {
        ctx.save(); ctx.translate(cx, cy - 16); ctx.rotate(Math.atan2(best[0], best[1]));
        ctx.strokeStyle = ORANGE; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(-24, 0); ctx.lineTo(16, 0); ctx.stroke();
        ctx.fillStyle = ORANGE; ctx.beginPath(); ctx.moveTo(28, 0); ctx.lineTo(14, -9); ctx.lineTo(14, 9); ctx.closePath(); ctx.fill(); ctx.restore();
      }
    }
    ctx.fillStyle = MUTED; ctx.textBaseline = "alphabetic"; ctx.textAlign = "center"; ctx.font = "bold 25px ui-monospace,Menlo,monospace";
    ctx.fillText("色 = 状態価値 V(s)（ゴール G に近いほど高い）／オレンジ矢印 = 貪欲方策（価値の高い隣へ）", W / 2, y0 + n * cs + 50);
  }

  // llm/04: スケーリング則 — 計算量を増やすと損失がべき乗で下がる（log-log で直線）
  function drawScalingLaw(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var ml = 160, mr = 120, mt = 100, mb = 130, pw = W - ml - mr, ph = H - mt - mb;
    ctx.strokeStyle = INK; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(ml, mt); ctx.lineTo(ml, mt + ph); ctx.lineTo(ml + pw, mt + ph); ctx.stroke();
    var a = [ml + pw * 0.03, mt + ph * 0.08], b = [ml + pw * 0.97, mt + ph * 0.92];
    ctx.strokeStyle = TEAL; ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.fillStyle = ORANGE; for (var k = 0; k < 6; k++) { var t = k / 5, x = a[0] + (b[0] - a[0]) * t, y = a[1] + (b[1] - a[1]) * t; ctx.beginPath(); ctx.arc(x, y, 11, 0, 7); ctx.fill(); }
    ctx.textAlign = "left"; ctx.fillStyle = TEAL; ctx.font = "bold 32px ui-monospace,Menlo,monospace"; ctx.fillText("L ∝ C^(-α)", ml + pw * 0.46, mt + ph * 0.30);
    ctx.fillStyle = MUTED; ctx.font = "24px ui-monospace,Menlo,monospace"; ctx.fillText("（log-log では直線）", ml + pw * 0.46, mt + ph * 0.30 + 38);
    ctx.fillStyle = INK; ctx.font = "bold 27px ui-monospace,Menlo,monospace"; ctx.textAlign = "center"; ctx.fillText("計算量 C（log）→", ml + pw / 2, mt + ph + 62);
    ctx.save(); ctx.translate(ml - 76, mt + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("損失 L（log）→", 0, 0); ctx.restore();
    ctx.fillStyle = MUTED; ctx.textAlign = "left"; ctx.font = "24px ui-monospace,Menlo,monospace"; ctx.fillText("計算・データ・パラメータを増やすほど損失が下がる（予測可能なべき乗則）", ml, mt + ph + 108);
  }

  // audio/07: Sway Sampling — ODE の時刻を等間隔でなく序盤(ノイズ支配)に寄せて刻む
  function drawSwaySampling(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var ml = 130, mr = 130, pw = W - ml - mr, N = 8, yU = 230, yS = 470;
    function X(t) { return ml + t * pw; }
    function axis(y, label, col) {
      ctx.strokeStyle = HAIR; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(X(0), y); ctx.lineTo(X(1), y); ctx.stroke();
      ctx.fillStyle = col; ctx.font = "bold 28px ui-monospace,Menlo,monospace"; ctx.textAlign = "left"; ctx.fillText(label, X(0), y - 38);
    }
    axis(yU, "等間隔 (uniform)：序盤も終盤も同じ粗さ", MUTED);
    axis(yS, "Sway sampling (s<0)：序盤に評価点が密集", TEAL);
    for (var k = 0; k < N; k++) {
      var u = k / (N - 1), tp = 1 - Math.cos(Math.PI / 2 * u); // s=-1 のとき t' = 1 - cos(π/2 u)
      // 同じ番号の点を薄い線で結び、左（t=0）へ寄ったことを示す
      ctx.strokeStyle = "rgba(90,101,115,0.35)"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(u), yU); ctx.lineTo(X(tp), yS); ctx.stroke();
      ctx.fillStyle = MUTED; ctx.beginPath(); ctx.arc(X(u), yU, 12, 0, 7); ctx.fill();
      ctx.fillStyle = TEAL; ctx.beginPath(); ctx.arc(X(tp), yS, 12, 0, 7); ctx.fill();
    }
    // 端ラベル
    ctx.fillStyle = INK; ctx.font = "26px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
    ctx.fillText("t=0（ノイズ）", X(0), yS + 56); ctx.fillText("t=1（mel）", X(1), yS + 56);
    ctx.fillStyle = MUTED; ctx.font = "25px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
    ctx.fillText("t = u + s·(cos(π/2·u) − 1 + u)  …  s<0 で左へ寄る（少ステップでも序盤を細かく辿れる）", W / 2, yS + 102);
  }

  // audio/09: Nemotron 3.5 ASR のレイテンシ↔WER。チャンクを大きく（=待つ）すると WER が下がる。
  // 実測点は FLEURS transcription-ready 平均（80/320/1120ms の 3 点のみ公開値）。
  function drawNemotronLatencyWer(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var ml = 150, mr = 80, mt = 70, mb = 130, pw = W - ml - mr, ph = H - mt - mb;
    var xs = [80, 1120], ys = [8.4, 10.8]; // 描画範囲
    function X(ms) { return ml + (ms - xs[0]) / (xs[1] - xs[0]) * pw; }
    function Y(w) { return mt + (w - ys[0]) / (ys[1] - ys[0]) * ph; } // 上が低 WER
    // 軸
    ctx.strokeStyle = INK; ctx.lineWidth = 2.5; ctx.beginPath();
    ctx.moveTo(ml, mt); ctx.lineTo(ml, mt + ph); ctx.lineTo(ml + pw, mt + ph); ctx.stroke();
    // y グリッド
    ctx.fillStyle = MUTED; ctx.font = "24px ui-monospace,Menlo,monospace";
    [9, 10].forEach(function (w) {
      ctx.strokeStyle = "#e3e8ec"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(ml, Y(w)); ctx.lineTo(ml + pw, Y(w)); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(w + "%", ml - 14, Y(w) + 8);
    });
    // x 目盛（5 つの動作点）
    var ticks = [80, 160, 320, 560, 1120];
    ctx.textAlign = "center";
    ticks.forEach(function (ms) {
      ctx.strokeStyle = "#c2ccd5"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(ms), mt + ph); ctx.lineTo(X(ms), mt + ph + 8); ctx.stroke();
      ctx.fillStyle = MUTED; ctx.fillText(ms, X(ms), mt + ph + 38);
    });
    // 公開された 3 実測点
    var pts = [[80, 10.38], [320, 9.49], [1120, 8.84]];
    ctx.strokeStyle = TEAL; ctx.lineWidth = 5; ctx.beginPath();
    pts.forEach(function (p, i) { if (i) ctx.lineTo(X(p[0]), Y(p[1])); else ctx.moveTo(X(p[0]), Y(p[1])); }); ctx.stroke();
    pts.forEach(function (p) {
      ctx.fillStyle = ORANGE; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 11, 0, 7); ctx.fill();
      ctx.fillStyle = INK; ctx.font = "bold 25px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
      ctx.fillText(p[1].toFixed(2) + "%", X(p[0]), Y(p[1]) - 22);
    });
    // 軸ラベル
    ctx.fillStyle = INK; ctx.font = "bold 27px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
    ctx.fillText("チャンク（遅延）ms →", ml + pw / 2, H - 26);
    ctx.save(); ctx.translate(ml - 92, mt + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("WER（低いほど良い）", 0, 0); ctx.restore();
    ctx.fillStyle = TEAL; ctx.font = "26px ui-monospace,Menlo,monospace"; ctx.textAlign = "left";
    ctx.fillText("待つほど精度↑（遅延↔精度のトレードオフ）", X(360), Y(10.5));
  }

  // audio/09: 同時ストリーム数（H100 1 枚）。各遅延設定ごとに Nemotron 0.6B vs Parakeet 1.1B。
  // グループ内で各自の最大に正規化して比率（17x / 6x）を見せる。
  function drawNemotronThroughput(c) {
    var ctx = c.getContext("2d"), W = c.width, H = c.height; ctx.clearRect(0, 0, W, H);
    var groups = [
      { name: "80ms 設定", nemo: 240, para: 14, ratio: "17x" },
      { name: "1120ms 設定", nemo: 2400, para: 400, ratio: "6x" },
    ];
    var gap = 90, gw = (W - gap) / 2, mt = 90, mb = 110, barMaxH = H - mt - mb;
    groups.forEach(function (g, gi) {
      var ox = gi * (gw + gap);
      var max = g.nemo, bw = 150, cx = ox + gw / 2, x1 = cx - bw - 50, x2 = cx + 50;
      function barH(v) { return v / max * barMaxH; }
      function bar(x, v, col, label) {
        var h = barH(v), y = mt + barMaxH - h;
        ctx.fillStyle = col; ctx.fillRect(x, y, bw, h);
        ctx.fillStyle = INK; ctx.font = "bold 30px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
        ctx.fillText(v.toLocaleString(), x + bw / 2, y - 16);
        ctx.fillStyle = MUTED; ctx.font = "23px ui-monospace,Menlo,monospace";
        ctx.fillText(label, x + bw / 2, mt + barMaxH + 40);
      }
      bar(x1, g.nemo, TEAL, "Nemotron 0.6B");
      bar(x2, g.para, "rgba(221,106,43,0.85)", "Parakeet 1.1B");
      // 比率
      ctx.fillStyle = ORANGE; ctx.font = "bold 40px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
      ctx.fillText(g.ratio, cx, mt - 30);
      ctx.fillStyle = INK; ctx.font = "bold 27px ui-monospace,Menlo,monospace";
      ctx.fillText(g.name, cx, mt + barMaxH + 80);
    });
    ctx.fillStyle = MUTED; ctx.font = "24px ui-monospace,Menlo,monospace"; ctx.textAlign = "center";
    ctx.fillText("H100 1 枚あたりの同時ストリーム数（グループ内で正規化）", W / 2, 40);
  }
})();
