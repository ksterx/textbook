// @ts-check
import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import remarkDirective from 'remark-directive';
import rehypeKatex from 'rehype-katex';
import remarkCallout from './src/plugins/remark-callout.mjs';
import remarkMermaid from './src/plugins/remark-mermaid.mjs';
import rehypeBaseLinks from './src/plugins/rehype-base-links.mjs';
import rehypeTableWrap from './src/plugins/rehype-table-wrap.mjs';

// GitHub Pages の Project site はサブパス配信なので base を付ける。
const BASE = '/textbook';

export default defineConfig({
  site: 'https://ksterx.github.io',
  base: BASE,
  trailingSlash: 'always',
  markdown: {
    // remark: 数式 → directive(:::) → callout/details 化 → ```mermaid を <pre class="mermaid"> 化
    remarkPlugins: [remarkMath, remarkDirective, remarkCallout, remarkMermaid],
    // rehype: KaTeX をビルド時レンダー（CDN なし） → 内部リンクに base を付与 → 表を横スクロール枠で包む
    rehypePlugins: [rehypeKatex, [rehypeBaseLinks, BASE], rehypeTableWrap],
    shikiConfig: { theme: 'github-dark' },
  },
});
