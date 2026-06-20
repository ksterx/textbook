// MathJax 3 のグローバル設定。
// arithmatex を generic モードで使うため、インライン数式は \( ... \)、
// ディスプレイ数式は \[ ... \] のデリミタを処理対象にする。
// .arithmatex クラスの付いた要素だけを処理し、それ以外は無視する。
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// instant navigation（ページを再読み込みせずに遷移する機能）が有効なので、
// ページが切り替わるたびに数式を組版し直す。
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
