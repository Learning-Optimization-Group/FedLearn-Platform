// Static-markup structure tests for the redesigned renderer sections.
//
// The desktop jest suite runs in the node environment (no jsdom/RTL), so these
// tests render initial markup via react-dom/server — effects (IPC fetches,
// scrolling, notifications) intentionally do not run; the async state-flow
// logic behind them is covered unit-style in trainFlow/logView/runNotifications
// tests. What this file locks down is the two-state layout swap, the readiness
// gating of the primary button, and the design-system contract of one primary
// action per view.

import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import TrainSection, { type TrainSectionProps } from '../renderer/components/TrainSection';
import SettingsSection from '../renderer/components/SettingsSection';
import LogPanel from '../renderer/components/LogPanel';
import { HardwareProfilePicker } from '../renderer/components/HardwareSelector';

const noop = () => undefined;

function renderTrain(overrides: Partial<TrainSectionProps> = {}): string {
  return renderToStaticMarkup(
    createElement(TrainSection, {
      status: 'idle',
      logs: [],
      onStart: noop,
      onStop: noop,
      ...overrides,
    }),
  );
}

function countOccurrences(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

describe('TrainSection — setup state', () => {
  it('renders the guided setup card with a disabled Start while checks are pending', () => {
    const html = renderTrain();
    expect(html).toContain('Set up training');
    expect(html).toContain('Model to train');
    expect(html).toContain('Dataset folder');
    expect(html).toContain('Readiness');
    // Before any IPC resolves every readiness row is pending → Start disabled.
    expect(html).toContain('Contacting server…');
    expect(html).toMatch(/<button[^>]*id="start-training-button"[^>]*disabled/);
    expect(html).not.toContain('id="stop-training-button"');
  });

  it('hides the log pane when there are no logs and shows it when there are', () => {
    expect(renderTrain()).not.toContain('Last run log');
    expect(renderTrain({ logs: ['previous run line\n'] })).toContain('Last run log');
  });

  it('exposes the Advanced disclosure with the hardware profile override', () => {
    const html = renderTrain();
    expect(html).toContain('Advanced');
    expect(html).toContain('id="profile-cpu"');
    expect(html).toContain('id="profile-jetson"');
  });

  it('has exactly one primary action', () => {
    expect(countOccurrences(renderTrain(), 'btn btn-primary')).toBe(1);
  });
});

describe('TrainSection — running state swap', () => {
  it('makes logs the dominant surface with a compact run header and Stop', () => {
    const html = renderTrain({ status: 'running' });
    expect(html).toContain('run-header');
    expect(html).toMatch(/<button[^>]*id="stop-training-button"/);
    expect(html).toContain('Activity log');
    expect(html).not.toContain('id="start-training-button"');
    expect(html).not.toContain('Readiness');
  });

  it('treats pulling as an active run and labels it Starting', () => {
    const html = renderTrain({ status: 'pulling' });
    expect(html).toContain('Starting');
    expect(html).toMatch(/<button[^>]*id="stop-training-button"/);
  });

  it.each(['restarting', 'paused'] as const)(
    'treats %s as an active run — running layout, never the setup card',
    (status) => {
      const html = renderTrain({ status });
      expect(html).toContain('run-header');
      expect(html).toMatch(/<button[^>]*id="stop-training-button"/);
      expect(html).not.toContain('id="start-training-button"');
      expect(html).not.toContain('Readiness');
    },
  );

  it('accepts the legacy isRunning boolean when no status is supplied', () => {
    const running = renderTrain({ status: undefined, isRunning: true });
    expect(running).toMatch(/<button[^>]*id="stop-training-button"/);
    const idle = renderTrain({ status: undefined, isRunning: false });
    expect(idle).toMatch(/<button[^>]*id="start-training-button"/);
  });
});

describe('TrainSection — outcome states', () => {
  it('shows a success banner and relabels the primary to Run again on completed', () => {
    const html = renderTrain({ status: 'completed', logs: ['done\n'] });
    expect(html).toContain('run-banner-success');
    expect(html).toContain('Run again');
    expect(html).toContain('Last run log'); // logs retained
  });

  it('shows a danger banner on error', () => {
    const html = renderTrain({ status: 'error', logs: ['ERROR: boom\n'] });
    expect(html).toContain('run-banner-danger');
    expect(html).toContain('Run again');
  });
});

describe('LogPanel', () => {
  it('keeps the empty state when there is no output', () => {
    const html = renderToStaticMarkup(createElement(LogPanel, { logs: [] }));
    expect(html).toContain('No output yet');
    expect(html).not.toContain('log-toolbar');
  });

  it('renders per-line severity classes, timestamps, and the filter input', () => {
    const html = renderToStaticMarkup(
      createElement(LogPanel, {
        logs: ['Round 1 complete\n', 'ERROR: aborted\n', 'WARNING: slow heartbeat\n'],
      }),
    );
    expect(html).toContain('log-line-error');
    expect(html).toContain('log-line-warn');
    expect(html).toContain('log-time');
    expect(html).toContain('aria-label="Filter log lines"');
    // Plain-text rendering only — log content must never be interpreted as HTML.
    expect(html).toContain('ERROR: aborted');
  });

  it('escapes HTML-shaped log content (XSS guard)', () => {
    const html = renderToStaticMarkup(
      createElement(LogPanel, { logs: ['<img src=x onerror=alert(1)>\n'] }),
    );
    expect(html).not.toContain('<img');
    expect(html).toContain('&lt;img');
  });
});

describe('SettingsSection', () => {
  it('renders server, updates, and about cards with one primary action', () => {
    const html = renderToStaticMarkup(createElement(SettingsSection, {}));
    expect(html).toContain('id="settings-section-title"');
    expect(html).toContain('Server URL');
    expect(html).toContain('Address of the FedLearn server this app connects to.');
    expect(html).toContain('id="settings-check-updates-button"');
    expect(html).toContain('App version');
    // __APP_VERSION__ is webpack-injected; under jest the typeof guard yields "dev".
    expect(html).toContain('vdev');
    expect(countOccurrences(html, 'btn btn-primary')).toBe(1);
  });

  it('starts with no warning or outcome banners', () => {
    const html = renderToStaticMarkup(createElement(SettingsSection, {}));
    expect(html).not.toContain('Use HTTP anyway');
    expect(html).not.toContain('validation-error');
    expect(html).not.toContain('validation-success');
  });
});

describe('HardwareProfilePicker', () => {
  it('renders all four profiles and marks the selected one', () => {
    const html = renderToStaticMarkup(
      createElement(HardwareProfilePicker, { value: 'mps', onChange: noop }),
    );
    for (const id of ['discrete', 'jetson', 'mps', 'cpu']) {
      expect(html).toContain(`id="profile-${id}"`);
    }
    expect(html).toMatch(/<button[^>]*id="profile-mps"[^>]*aria-pressed="true"/);
    expect(html).toMatch(/<button[^>]*id="profile-cpu"[^>]*aria-pressed="false"/);
  });

  it('disables every card when disabled', () => {
    const html = renderToStaticMarkup(
      createElement(HardwareProfilePicker, { value: 'cpu', onChange: noop, disabled: true }),
    );
    expect(countOccurrences(html, 'disabled=""')).toBe(4);
  });
});
