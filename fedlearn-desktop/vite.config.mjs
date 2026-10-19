import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const require = createRequire(import.meta.url);
const { buildRendererCsp } = require('./webpack.csp.js');
const csp = buildRendererCsp({ allowEval: false, allowInlineStyle: true });

export default defineConfig({
  root: fileURLToPath(new URL('./src/renderer', import.meta.url)),
  define: {
    __APP_VERSION__: JSON.stringify(require('./package.json').version),
  },
  plugins: [{
    name: 'renderer-dev-html',
    transformIndexHtml(html) {
      // Share the production HTML template while supplying Vite's dev entry.
      return {
        html: html.replace('<%= htmlWebpackPlugin.options.templateParameters.csp %>', csp),
        tags: [{ tag: 'script', attrs: { type: 'module', src: '/index.tsx' }, injectTo: 'body' }],
      };
    },
  }],
  server: {
    host: 'localhost',
    port: 9000,
    strictPort: true,
    headers: { 'Content-Security-Policy': csp },
  },
});
