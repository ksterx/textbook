import { visit } from 'unist-util-visit';

// ```mermaid フェンスを <pre class="mermaid">…</pre>（生テキスト）へ変換する。
// Shiki でハイライトされる前（remark 段階）に html ノードへ置き換えるので、
// mermaid.js が textContent をそのまま読んで描画できる。
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

export default function remarkMermaid() {
  return (tree) => {
    visit(tree, 'code', (node) => {
      if (node.lang === 'mermaid') {
        node.type = 'html';
        node.value = `<pre class="mermaid">${esc(node.value)}</pre>`;
      }
    });
  };
}
