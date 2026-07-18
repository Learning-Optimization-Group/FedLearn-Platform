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
 * Flatten App's log-entry buffer (entries may hold several '\n'-separated
 * lines) into displayable lines. Empty segments (trailing newlines, blank
 * lines) are dropped.
 */
export function splitLogEntries(entries: string[]): DisplayLogLine[] {
  const lines: DisplayLogLine[] = [];
  for (let i = 0; i < entries.length; i++) {
    for (const seg of entries[i].split('\n')) {
      if (seg.trim() === '') continue;
      lines.push({ text: seg, entryIndex: i, severity: classifyLogSeverity(seg) });
    }
  }
  return lines;
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
