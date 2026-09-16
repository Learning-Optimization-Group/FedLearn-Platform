// Unit tests for the LogPanel display helpers (pure).

import {
  classifyLogSeverity,
  createLogLineCache,
  filterLogLines,
  formatLogTime,
  splitLogEntries,
  updateLogLineCache,
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

  it('numbers lines with a stable, contiguous lineIndex', () => {
    expect(splitLogEntries(['a\nb\n', '\n', 'c']).map((l) => l.lineIndex)).toEqual([0, 1, 2]);
  });
});

describe('updateLogLineCache (incremental split)', () => {
  it('parses appended entries only, preserving earlier line objects and their keys', () => {
    const cache = createLogLineCache();
    const first = updateLogLineCache(cache, ['a\nb\n']);
    expect(first.map((l) => [l.text, l.entryIndex, l.lineIndex])).toEqual([
      ['a', 0, 0],
      ['b', 0, 1],
    ]);

    const second = updateLogLineCache(cache, ['a\nb\n', 'ERROR: c\n']);
    expect(second.map((l) => [l.text, l.entryIndex, l.lineIndex])).toEqual([
      ['a', 0, 0],
      ['b', 0, 1],
      ['ERROR: c', 1, 2],
    ]);
    expect(second[2].severity).toBe('error');
    // Already-parsed lines keep object identity (React.memo rows skip re-render)…
    expect(second[0]).toBe(first[0]);
    expect(second[1]).toBe(first[1]);
    // …while the array reference changes because content changed.
    expect(second).not.toBe(first);
  });

  it('returns the same array reference when nothing was appended', () => {
    const cache = createLogLineCache();
    const entries = ['a\n', 'b\n'];
    const first = updateLogLineCache(cache, entries);
    expect(updateLogLineCache(cache, entries)).toBe(first);
  });

  it('resets on clear (shorter buffer) and re-numbers from zero', () => {
    const cache = createLogLineCache();
    updateLogLineCache(cache, ['a\n', 'b\n']);
    const cleared = updateLogLineCache(cache, []);
    expect(cleared).toEqual([]);
    const fresh = updateLogLineCache(cache, ['new run\n']);
    expect(fresh.map((l) => [l.text, l.entryIndex, l.lineIndex])).toEqual([['new run', 0, 0]]);
  });

  it("re-splits in full when the buffer's head was trimmed (App's 10K cap)", () => {
    const cache = createLogLineCache();
    updateLogLineCache(cache, ['a\n', 'b\n', 'c\n']);
    // Same length, but the oldest entry was dropped and a new one appended.
    const trimmed = updateLogLineCache(cache, ['b\n', 'c\n', 'd\n']);
    expect(trimmed.map((l) => [l.text, l.entryIndex, l.lineIndex])).toEqual([
      ['b', 0, 0],
      ['c', 1, 1],
      ['d', 2, 2],
    ]);
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
