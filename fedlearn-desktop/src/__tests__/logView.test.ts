// Unit tests for the LogPanel display helpers (pure).

import {
  classifyLogSeverity,
  filterLogLines,
  formatLogTime,
  splitLogEntries,
} from '../renderer/components/logView';

describe('classifyLogSeverity', () => {
  it.each<[string, string]>([
    ['[System] Failed to start: boom', 'error'],
    ['ERROR: CUDA out of memory', 'error'],
    ['Traceback (most recent call last):', 'error'],
    ['grpc exception during round 3', 'error'],
    ['WARNING: flwr_datasets is deprecated', 'warn'],
    ['warn: heartbeat slow', 'warn'],
    ['Epoch 1/5 — loss=0.31 acc=0.88', 'info'],
    ['[System] Starting training container...', 'info'],
  ])('classifies %j as %s', (line, severity) => {
    expect(classifyLogSeverity(line)).toBe(severity);
  });

  it('requires word boundaries — substrings inside words do not match', () => {
    expect(classifyLogSeverity('no terrors here')).toBe('info');
    expect(classifyLogSeverity('unwarned metrics')).toBe('info');
  });

  it('ranks error above warn when a line matches both', () => {
    expect(classifyLogSeverity('WARNING: previous ERROR repeated')).toBe('error');
  });
});

describe('splitLogEntries', () => {
  it('flattens multi-line entries, keeping the source entry index', () => {
    const lines = splitLogEntries(['a\nb\n', 'c']);
    expect(lines.map((l) => [l.text, l.entryIndex])).toEqual([
      ['a', 0],
      ['b', 0],
      ['c', 1],
    ]);
  });

  it('drops blank segments and blank-only entries', () => {
    expect(splitLogEntries(['\n', '   \n', 'x\n\ny\n'])).toHaveLength(2);
  });

  it('attaches a severity to each line', () => {
    const lines = splitLogEntries(['ok\nERROR: nope\n']);
    expect(lines[0].severity).toBe('info');
    expect(lines[1].severity).toBe('error');
  });

  it('returns an empty list for an empty buffer', () => {
    expect(splitLogEntries([])).toEqual([]);
  });
});

describe('filterLogLines', () => {
  const lines = splitLogEntries(['Round 1 complete\n', 'ERROR: Round 2 aborted\n', 'heartbeat ok\n']);

  it('filters case-insensitively by substring', () => {
    expect(filterLogLines(lines, 'round').map((l) => l.text)).toEqual([
      'Round 1 complete',
      'ERROR: Round 2 aborted',
    ]);
  });

  it('keeps everything for an empty or whitespace query', () => {
    expect(filterLogLines(lines, '')).toHaveLength(3);
    expect(filterLogLines(lines, '   ')).toHaveLength(3);
  });

  it('returns no lines when nothing matches', () => {
    expect(filterLogLines(lines, 'no-such-token')).toEqual([]);
  });
});

describe('formatLogTime', () => {
  it('renders local wall-clock HH:MM:SS with zero padding', () => {
    const epoch = new Date(2026, 0, 2, 3, 4, 5).getTime(); // local 03:04:05
    expect(formatLogTime(epoch)).toBe('03:04:05');
  });

  it('pads all fields', () => {
    const epoch = new Date(2026, 5, 6, 0, 0, 0).getTime();
    expect(formatLogTime(epoch)).toBe('00:00:00');
  });
});
