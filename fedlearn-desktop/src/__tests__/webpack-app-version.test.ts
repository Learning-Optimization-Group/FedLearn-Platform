// DE-4: the app version shown in the UI is injected at build time from
// package.json via webpack's DefinePlugin (replacing the old hardcoded
// 'v1.0.0'). This test also doubles as a syntax check that both webpack
// configs still load.

/* eslint-disable @typescript-eslint/no-var-requires */
const expectedDefine = JSON.stringify(require('../../package.json').version);

interface DefineLike {
  definitions?: Record<string, unknown>;
}

function findAppVersionDefine(plugins: DefineLike[] | undefined): unknown {
  const plugin = (plugins ?? []).find(
    (p) => p && p.definitions && '__APP_VERSION__' in p.definitions,
  );
  return plugin?.definitions?.__APP_VERSION__;
}

describe('webpack __APP_VERSION__ injection (DE-4)', () => {
  it('renderer (dev) config defines __APP_VERSION__ from package.json', () => {
    const config = require('../../webpack.renderer.config.js');
    expect(findAppVersionDefine(config.plugins)).toBe(expectedDefine);
  });

  it('prod config defines __APP_VERSION__ on the renderer bundle', () => {
    const configs = require('../../webpack.prod.config.js') as Array<{
      name?: string;
      plugins?: DefineLike[];
    }>;
    const renderer = configs.find((c) => c.name === 'renderer');
    expect(renderer).toBeDefined();
    expect(findAppVersionDefine(renderer?.plugins)).toBe(expectedDefine);
  });
});
