// =============================================================================
// FedLearn Desktop — LogPanel Component
// =============================================================================
// SECURITY: Renders plain text only — no HTML from container output.
// No dangerouslySetInnerHTML. No innerHTML. All log lines are rendered
// as React text nodes to guarantee XSS safety.
//
// Display upgrades (data flow untouched — App still owns the string[] buffer):
// - follow-tail that pauses when the user scrolls up, with a "Jump to latest"
//   pill to resume;
// - per-line severity coloring via tokens (ERROR → danger, WARN → warning);
// - a small case-insensitive filter input;
// - arrival timestamps per buffer entry (stamped renderer-side on receipt).
//
// PERFORMANCE: Per-line severity/timestamps need per-line nodes, so rendering
// is capped at the most recent MAX_RENDERED_LINES lines (the full buffer stays
// in App and remains searchable — filtering runs over everything). Line parsing
// is INCREMENTAL: a per-mount LogLineCache splits/classifies only the entries
// appended since the last batch (App's buffer is append-only except on clear,
// which resets the cache), line objects keep their identity, and rows are
// memoized with stable keys so React reuses the untouched DOM nodes instead of
// re-rendering up to MAX_RENDERED_LINES spans per batch.
// =============================================================================

import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { ScrollText, Search, ArrowDown } from 'lucide-react';
import {
  createLogLineCache,
  updateLogLineCache,
  filterLogLines,
  formatLogTime,
  type DisplayLogLine,
} from './logView';
import './sections.css';

interface LogPanelProps {
  logs: string[];
}

// Memoized row: with cached line objects (stable identity) and a stable
// arrival time, appending a batch re-renders only the new rows.
const LogLineRow = React.memo<{ line: DisplayLogLine; arrivedAt: number }>(
  ({ line, arrivedAt }) => (
    <span
      className={
        line.severity === 'error'
          ? 'log-line log-line-error'
          : line.severity === 'warn'
            ? 'log-line log-line-warn'
            : 'log-line'
      }
    >
      <span className="log-time">{formatLogTime(arrivedAt)}</span>
      {line.text}
    </span>
  ),
);
LogLineRow.displayName = 'LogLineRow';

/** Upper bound on DOM log lines; the App-side buffer (10K entries) is larger. */
const MAX_RENDERED_LINES = 2000;

/** "At bottom" tolerance in px for the follow-tail detector. */
const FOLLOW_EPSILON_PX = 50;

const LogPanel: React.FC<LogPanelProps> = ({ logs }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [following, setFollowing] = useState(true);
  const [query, setQuery] = useState('');

  // Arrival timestamps per buffer entry, stamped when an entry first appears.
  // A shrinking buffer means a new run cleared the logs — reset the clock map.
  // (If App's 10K cap trims the head while full, older stamps drift one entry —
  // an accepted approximation; stamps are renderer-side arrival times anyway.)
  const entryTimesRef = useRef<number[]>([]);
  if (logs.length < entryTimesRef.current.length) {
    entryTimesRef.current = [];
  }
  while (entryTimesRef.current.length < logs.length) {
    entryTimesRef.current.push(Date.now());
  }

  // Incremental split: only entries appended since the last render are parsed.
  // updateLogLineCache returns a new array reference only when content changed
  // (and resets itself when the buffer shrinks on clear), so it is safe as a
  // useMemo dependency despite living in a ref.
  const lineCacheRef = useRef(createLogLineCache());
  const allLines = updateLogLineCache(lineCacheRef.current, logs);
  const filtered = useMemo(() => filterLogLines(allLines, query), [allLines, query]);
  const isFiltering = query.trim() !== '';

  const hiddenCount = Math.max(0, filtered.length - MAX_RENDERED_LINES);
  const visible = hiddenCount > 0 ? filtered.slice(hiddenCount) : filtered;

  // Follow the tail while the user hasn't scrolled up.
  useEffect(() => {
    if (following && containerRef.current) {
      const el = containerRef.current;
      el.scrollTop = el.scrollHeight;
    }
  }, [visible, following]);

  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < FOLLOW_EPSILON_PX;
    setFollowing(atBottom);
  }, []);

  const jumpToLatest = useCallback(() => {
    const el = containerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
    setFollowing(true);
  }, []);

  if (logs.length === 0) {
    return (
      <div className="log-panel log-panel-empty" ref={containerRef}>
        <div className="log-empty-state">
          <span className="log-empty-icon"><ScrollText strokeWidth={1.5} size={28} /></span>
          <p className="log-empty-title">No output yet</p>
          <p className="log-empty-text">
            Start a training session to see live logs here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="log-panel-wrap">
      <div className="log-toolbar">
        <div className="log-search">
          <Search strokeWidth={1.5} size={14} aria-hidden="true" />
          <input
            type="text"
            className="log-search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter logs…"
            aria-label="Filter log lines"
          />
        </div>
        {isFiltering && (
          <span className="log-match-count" role="status">
            {filtered.length} of {allLines.length} lines
          </span>
        )}
      </div>

      <div className="log-panel" ref={containerRef} onScroll={handleScroll}>
        {hiddenCount > 0 && (
          <div className="log-truncated-note">
            Showing the last {MAX_RENDERED_LINES.toLocaleString()} of {filtered.length.toLocaleString()} lines
          </div>
        )}
        <pre className="log-content">
          {/*
            SECURITY: Every piece of log output below is a plain React text
            node — React escapes all content, so no HTML from container output
            is ever interpreted.
          */}
          {visible.map((line) => (
            <LogLineRow
              key={line.lineIndex}
              line={line}
              arrivedAt={entryTimesRef.current[line.entryIndex] ?? 0}
            />
          ))}
        </pre>
      </div>

      {!following && (
        <button type="button" className="log-jump-pill" onClick={jumpToLatest}>
          <ArrowDown strokeWidth={1.5} size={14} aria-hidden="true" />
          Jump to latest
        </button>
      )}
    </div>
  );
};

export default LogPanel;
