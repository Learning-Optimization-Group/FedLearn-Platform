// DE-14: the packaged (production) renderer must not carry 'unsafe-eval' in
// its Content-Security-Policy, and must not depend on remote Google Fonts
// hosts (fonts.googleapis.com / fonts.gstatic.com). Dev builds may keep
// 'unsafe-eval' because webpack's development build uses the `eval` devtool.
//
// DE-14 follow-up: the packaged (production) renderer also must not carry
// 'unsafe-inline' in style-src — CSS is extracted via MiniCssExtractPlugin
// and loaded through a <link rel="stylesheet"> instead of style-loader's
// runtime <style> tag injection. Dev builds keep 'unsafe-inline' because
// style-loader is required there for HMR.
//
// This also doubles as a syntax check that webpack.csp.js and both renderer
// webpack configs still load, following the pattern in webpack-app-version.test.ts.

/* eslint-disable @typescript-eslint/no-require-imports */
import * as fs from 'fs';
import * as path from 'path';

interface PluginLike {
  constructor: { name?: string };
  options?: { templateParameters?: { csp?: string } };
}

interface ConfigLike {
  name?: string;
  plugins?: PluginLike[];
}

function findHtmlPluginCsp(plugins: PluginLike[] | undefined): string | undefined {
  const plugin = (plugins ?? []).find((p) => p?.constructor?.name === 'HtmlWebpackPlugin');
  return plugin?.options?.templateParameters?.csp;
}

describe('renderer CSP (DE-14)', () => {
  it('buildRendererCsp({ allowEval: false, allowInlineStyle: false }) has no unsafe-eval, no unsafe-inline style, and no remote font hosts', () => {
    const { buildRendererCsp } = require('../../webpack.csp');
    const csp = buildRendererCsp({ allowEval: false, allowInlineStyle: false });
    expect(csp).not.toMatch(/unsafe-eval/);
    expect(csp).not.toMatch(/googleapis/);
    expect(csp).not.toMatch(/gstatic/);
    expect(csp).toMatch(/script-src 'self'/);
    expect(csp).toMatch(/style-src 'self'/);
    expect(csp).not.toMatch(/style-src[^;]*'unsafe-inline'/);
  });

  it('buildRendererCsp({ allowEval: true, allowInlineStyle: true }) still allows unsafe-eval and unsafe-inline style for dev (eval devtool + style-loader HMR)', () => {
    const { buildRendererCsp } = require('../../webpack.csp');
    const csp = buildRendererCsp({ allowEval: true, allowInlineStyle: true });
    expect(csp).toMatch(/script-src[^;]*'unsafe-eval'/);
    expect(csp).toMatch(/style-src[^;]*'unsafe-inline'/);
  });

  it('prod webpack config bakes the strict CSP (no unsafe-eval, no unsafe-inline style) into HtmlWebpackPlugin templateParameters', () => {
    const configs = require('../../webpack.prod.config.js') as ConfigLike[];
    const renderer = configs.find((c) => c.name === 'renderer');
    expect(renderer).toBeDefined();
    const csp = findHtmlPluginCsp(renderer?.plugins);
    expect(csp).toBeDefined();
    expect(csp).not.toMatch(/unsafe-eval/);
    expect(csp).not.toMatch(/googleapis|gstatic/);
    expect(csp).not.toMatch(/style-src[^;]*'unsafe-inline'/);
  });

  it('prod webpack config extracts CSS via MiniCssExtractPlugin instead of style-loader', () => {
    const configs = require('../../webpack.prod.config.js') as ConfigLike[];
    const renderer = configs.find((c) => c.name === 'renderer') as {
      module?: { rules?: { test?: RegExp; use?: (string | { loader?: string })[] }[] };
      plugins?: PluginLike[];
    };
    expect(renderer).toBeDefined();
    const cssRule = renderer.module?.rules?.find((r) => r.test?.toString() === /\.css$/.toString());
    expect(cssRule).toBeDefined();
    const loaderNames = (cssRule?.use ?? []).map((u) => (typeof u === 'string' ? u : u.loader));
    expect(loaderNames).not.toContain('style-loader');
    const hasMiniCssExtractPlugin = (renderer.plugins ?? []).some(
      (p) => p?.constructor?.name === 'MiniCssExtractPlugin',
    );
    expect(hasMiniCssExtractPlugin).toBe(true);
  });

  it('dev renderer webpack config bakes a permissive (unsafe-eval, unsafe-inline style) CSP into HtmlWebpackPlugin templateParameters', () => {
    const config = require('../../webpack.renderer.config.js') as ConfigLike;
    const csp = findHtmlPluginCsp(config.plugins);
    expect(csp).toBeDefined();
    expect(csp).toMatch(/unsafe-eval/);
    expect(csp).toMatch(/style-src[^;]*'unsafe-inline'/);
  });

  it('dev renderer webpack config still uses style-loader (HMR requirement)', () => {
    const config = require('../../webpack.renderer.config.js') as {
      module?: { rules?: { test?: RegExp; use?: (string | { loader?: string })[] }[] };
    };
    const cssRule = config.module?.rules?.find((r) => r.test?.toString() === /\.css$/.toString());
    expect(cssRule).toBeDefined();
    const loaderNames = (cssRule?.use ?? []).map((u) => (typeof u === 'string' ? u : u.loader));
    expect(loaderNames).toContain('style-loader');
  });

  it('index.html template does not hardcode unsafe-eval or remote font hosts into the CSP meta tag', () => {
    const html = fs.readFileSync(
      path.join(__dirname, '../../src/renderer/index.html'),
      'utf8',
    );
    const match = html.match(
      /<meta http-equiv="Content-Security-Policy" content="([^"]*)"/,
    );
    expect(match).not.toBeNull();
    const cspAttr = match?.[1] ?? '';
    // The static template must delegate to templateParameters (webpack.csp.js)
    // rather than hardcoding a policy directly, so dev/prod cannot drift apart.
    expect(cspAttr).toMatch(/templateParameters/);
    expect(cspAttr).not.toMatch(/unsafe-eval/);
    expect(cspAttr).not.toMatch(/fonts\.googleapis\.com/);
    expect(cspAttr).not.toMatch(/fonts\.gstatic\.com/);
  });
});
