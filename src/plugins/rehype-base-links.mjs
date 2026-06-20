import { visit } from 'unist-util-visit';

// 本文中のルート絶対リンク（/audio/... など）に base（/textbook）を付ける。
// 著者は base を意識せず /audio/01-.../ のように書けばよい。
export default function rehypeBaseLinks(base = '/') {
  const b = base.replace(/\/$/, '');
  if (!b) return () => {};
  return (tree) => {
    visit(tree, 'element', (node) => {
      for (const attr of ['href', 'src']) {
        const v = node.properties && node.properties[attr];
        if (typeof v === 'string' && v.startsWith('/') && !v.startsWith('//') && !v.startsWith(b + '/')) {
          node.properties[attr] = b + v;
        }
      }
    });
  };
}
