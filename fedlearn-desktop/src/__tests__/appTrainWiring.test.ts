// Regression guard for the App → TrainSection status wiring.
//
// TrainSection derives its setup/running/completed/error phases from the FULL
// container status. If App collapses that to the legacy `isRunning` boolean,
// the completed/error outcome banners and the run-finished notifications
// become unreachable, and 'restarting'/'paused' fall back to the setup card
// mid-run. That regression once shipped silently because no test looked at the
// App-level wiring.
//
// The desktop jest suite runs in the node environment (no jsdom), and App
// cannot mount past its auth-check effect under react-dom/server (effects do
// not run there), so — following the file-content pattern of
// renderer-csp.test.ts — the wiring is locked down at the source level. The
// phase behavior behind the prop is covered component-side in
// sectionsRender.test.ts and trainFlow.test.ts.

import * as fs from 'fs';
import * as path from 'path';

describe('App → TrainSection wiring', () => {
  const appSource = fs.readFileSync(path.join(__dirname, '../renderer/App.tsx'), 'utf8');
  const trainSectionJsx = appSource.match(/<TrainSection[\s\S]*?\/>/)?.[0] ?? '';

  it('renders TrainSection exactly once', () => {
    expect(appSource.match(/<TrainSection/g)).toHaveLength(1);
  });

  it('passes the full container status through, not only a collapsed boolean', () => {
    expect(trainSectionJsx).toContain('status={containerStatus}');
  });

  it('does not resurrect an isRunning-only mapping on the TrainSection element', () => {
    expect(trainSectionJsx).not.toContain('isRunning=');
  });
});
