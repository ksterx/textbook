import { visit } from 'unist-util-visit';

// :::type[Title] … ::: のコンテナディレクティブを callout / details の HTML へ写像する。
// 内側はそのまま markdown として処理されるので、数式・リスト・表が生きる。
const TYPE = {
  note: 'note', info: 'note', tip: 'note',
  abstract: 'abstract', summary: 'abstract',
  success: 'success',
  question: 'question',
  warning: 'warn', danger: 'warn', caution: 'warn',
};

export default function remarkCallout() {
  return (tree) => {
    visit(tree, (node) => {
      if (node.type !== 'containerDirective') return;
      const label = node.children[0];
      const hasLabel = label && label.data && label.data.directiveLabel;

      if (node.name === 'details') {
        node.data = { ...(node.data || {}), hName: 'details', hProperties: { className: ['callout', 'details'] } };
        if (hasLabel) { label.data.hName = 'summary'; label.data.hProperties = {}; }
        return;
      }

      const cls = TYPE[node.name];
      if (!cls) return;
      node.data = { ...(node.data || {}), hName: 'aside', hProperties: { className: ['callout', cls] } };
      if (hasLabel) { label.data.hName = 'p'; label.data.hProperties = { className: ['tag'] }; }
    });
  };
}
