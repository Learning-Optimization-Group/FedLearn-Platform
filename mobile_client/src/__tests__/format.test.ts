// Display formatters for contribution numbers (Home / ledger rows).
import { formatBytes, formatDurationMs } from '../lib/format';

describe('formatBytes', () => {
  it('formats each unit tier', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(2048)).toBe('2.0 KB');
    expect(formatBytes(5_400_000)).toBe('5.1 MB'); // 5,400,000 / 1024² = 5.15 MiB
    expect(formatBytes(3 * 1024 ** 3)).toBe('3.0 GB');
  });

  it('renders invalid input as an em dash', () => {
    expect(formatBytes(-1)).toBe('—');
    expect(formatBytes(Number.NaN)).toBe('—');
  });
});

describe('formatDurationMs', () => {
  it('formats sub-minute, minute, and hour scales', () => {
    expect(formatDurationMs(900)).toBe('0.9s');
    expect(formatDurationMs(65_000)).toBe('1m 5s');
    expect(formatDurationMs(3_720_000)).toBe('1h 2m');
  });

  it('renders invalid input as an em dash', () => {
    expect(formatDurationMs(-5)).toBe('—');
    expect(formatDurationMs(Number.NaN)).toBe('—');
  });
});
