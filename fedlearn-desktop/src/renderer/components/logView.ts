// =============================================================================
// FedLearn Desktop — Log view helpers (pure, UI-free)
// =============================================================================
// Line splitting, severity classification, filtering, and timestamp formatting
// for LogPanel. The log *data flow* is untouched — App still owns the string[]
// buffer; these helpers only shape it for display. Kept free of React/DOM so
// the node-env jest suite can exercise every branch directly.
// =============================================================================

export type LogSeverity = 'error' | 'warn' | 'info';

export interface DisplayLogLine {
  text: string;
  /** Index of the App-buffer entry this line came from (for arrival times). */
  entryIndex: number;
  /** Stable position in the flattened line list (used as the React key). */
  lineIndex: number;
  severity: LogSeverity;
}

// Keyword heuristics over untrusted plain text. Order matters: error outranks
// warn when a line matches both.
const ERROR_RE = /\b(error|err|exception|traceback|fatal|critical|fail|failed|failure)\b/i;
const WARN_RE = /\b(warn|warning|deprecated)\b/i;

export function classifyLogSeverity(text: string): LogSeverity {
  if (ERROR_RE.test(text)) return 'error';
  if (WARN_RE.test(text)) return 'warn';
  return 'info';
}

/**
 * Incremental split cache. App's log buffer grows append-only (except on a
 * clear, and the rare head trim when the 10K entry cap is hit), so LogPanel
 * keeps one of these per mount and only the entries appended since the last
 * render are split/classified per batch — never the whole buffer.
 */
export interface LogLineCache {
  /** Number of buffer entries already parsed into `lines`. */
  parsedEntries: number;
  /** Probe values used to detect a non-append-only buffer change. */
  firstEntry: string | undefined;
  lastParsedEntry: string | undefined;
  /** Flattened display lines for the first `parsedEntries` entries. */
  lines: DisplayLogLine[];
}

export function createLogLineCache(): LogLineCache {
  return { parsedEntries: 0, firstEntry: undefined, lastParsedEntry: undefined, lines: [] };
}

/**
 * Flatten App's log-entry buffer (entries may hold several '\n'-separated
 * lines) into displayable lines, incrementally: only entries beyond
 * `cache.parsedEntries` are split per call. Empty segments (trailing newlines,
 * blank lines) are dropped. A shrunken buffer (clear) or a changed head/tail
 * probe (App's 10K cap trimmed the oldest entries) resets the cache and
 * re-splits in full — correctness over speed in that rare regime. (Like the
 * arrival-time stamps, the probes compare string values, so an all-identical
 * buffer trimmed in place can go undetected — an accepted approximation.)
 *
 * Identity contract: the returned array is `cache.lines`, and its reference
 * changes only when its content changed — appended lines land in a fresh
 * array — so callers can use it directly as a memo/effect dependency, and
 * unchanged `DisplayLogLine` objects keep their identity for React.memo rows.
 */
export function updateLogLineCache(cache: LogLineCache, entries: string[]): DisplayLogLine[] {
  const appendOnly =
    entries.length >= cache.parsedEntries
    && (cache.parsedEntries === 0
      || (entries[0] === cache.firstEntry
        && entries[cache.parsedEntries - 1] === cache.lastParsedEntry));
  if (!appendOnly) {
    cache.parsedEntries = 0;
    cache.firstEntry = undefined;
    cache.lastParsedEntry = undefined;
    cache.lines = [];
  }
  if (entries.length === cache.parsedEntries) return cache.lines;

  const lines = cache.lines.slice();
  for (let i = cache.parsedEntries; i < entries.length; i++) {
    for (const seg of entries[i].split('\n')) {
      if (seg.trim() === '') continue;
      lines.push({ text: seg, entryIndex: i, lineIndex: lines.length, severity: classifyLogSeverity(seg) });
    }
  }
  cache.parsedEntries = entries.length;
  cache.firstEntry = entries[0];
  cache.lastParsedEntry = entries[entries.length - 1];
  cache.lines = lines;
  return lines;
}

/** One-shot (non-cached) flatten — the incremental path with a throwaway cache. */
export function splitLogEntries(entries: string[]): DisplayLogLine[] {
  return updateLogLineCache(createLogLineCache(), entries);
}

/** Case-insensitive substring filter. An empty/whitespace query keeps all. */
export function filterLogLines(lines: DisplayLogLine[], query: string): DisplayLogLine[] {
  const q = query.trim().toLowerCase();
  if (q === '') return lines;
  return lines.filter((l) => l.text.toLowerCase().includes(q));
}

/** Local wall-clock "HH:MM:SS" for a line's arrival time. */
export function formatLogTime(epochMs: number): string {
  const d = new Date(epochMs);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  return `${hh}:${mm}:${ss}`;
}
