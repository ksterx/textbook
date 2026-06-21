import { visit, SKIP } from 'unist-util-visit';

// markdown が出力する素の <table> を、横スクロール可能な <div class="table-wrap"> で包む。
// 狭い画面で広い表が出ても、ページ全体が横にはみ出さず（＝本文が横ずれせず）、
// 表の中だけが独立して横スクロールするようにする。デスクトップでは収まる限りスクロールは出ない。
export default function rehypeTableWrap() {
  return (tree) => {
    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'table' || !parent || index == null) return;
      const pcls = parent.properties && parent.properties.className;
      if (parent.tagName === 'div' && Array.isArray(pcls) && pcls.includes('table-wrap')) return;
      const wrapper = {
        type: 'element',
        tagName: 'div',
        properties: { className: ['table-wrap'] },
        children: [node],
      };
      parent.children[index] = wrapper;
      // 包んだ枝には再帰しない（再ラップ防止）
      return [SKIP, index + 1];
    });
  };
}
