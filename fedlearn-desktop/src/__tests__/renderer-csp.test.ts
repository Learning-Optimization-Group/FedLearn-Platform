// DE-14: the packaged (production) renderer must not carry 'unsafe-eval' in
// its Content-Security-Policy, and must not depend on remote Google Fonts
// hosts (fonts.googleapis.com / fonts.gstatic.com). Dev builds may keep
// 'unsafe-eval' because webpack's development build uses the `eval` devtool.
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
  it('buildRendererCsp({ allowEval: false }) has no unsafe-eval and no remote font hosts', () => {
    const { buildRendererCsp } = require('../../webpack.csp');
    const csp = buildRendererCsp({ allowEval: false });
    expect(csp).not.toMatch(/unsafe-eval/);
    expect(csp).not.toMatch(/googleapis/);
    expect(csp).not.toMatch(/gstatic/);
    expect(csp).toMatch(/script-src 'self'/);
  });

  it('buildRendererCsp({ allowEval: true }) still allows unsafe-eval for the dev eval devtool', () => {
    const { buildRendererCsp } = require('../../webpack.csp');
    const csp = buildRendererCsp({ allowEval: true });
    expect(csp).toMatch(/script-src[^;]*'unsafe-eval'/);
  });

  it('prod webpack config bakes the strict CSP into HtmlWebpackPlugin templateParameters', () => {
    const configs = require('../../webpack.prod.config.js') as ConfigLike[];
    const renderer = configs.find((c) => c.name === 'renderer');
    expect(renderer).toBeDefined();
    const csp = findHtmlPluginCsp(renderer?.plugins);
    expect(csp).toBeDefined();
    expect(csp).not.toMatch(/unsafe-eval/);
    expect(csp).not.toMatch(/googleapis|gstatic/);
  });

  it('dev renderer webpack config bakes a permissive (unsafe-eval) CSP into HtmlWebpackPlugin templateParameters', () => {
    const config = require('../../webpack.renderer.config.js') as ConfigLike;
    const csp = findHtmlPluginCsp(config.plugins);
    expect(csp).toBeDefined();
    expect(csp).toMatch(/unsafe-eval/);
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
