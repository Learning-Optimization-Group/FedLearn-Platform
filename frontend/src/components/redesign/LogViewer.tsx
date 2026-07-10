// =============================================================================
// FedLearn Frontend — Redesigned LogViewer (Live Telemetry Dashboard)
// =============================================================================
// Wired to real WebSocket log stream from the backend, backed by logStore so
// closing & reopening the modal preserves prior session output.

import { useState, useEffect, useRef } from 'react';
import { Play, Pause, Filter, TerminalSquare, Trash2, Activity } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line } from 'recharts';
import { cn } from '../../lib/utils';
import * as api from '../../services/apiServices';
import { logStore, StoredLogEntry } from '../../services/logStore';
import { Button, LogConsole, StatusPill } from '../ui';
import { WS_BROKER_URL } from '../../lib/serverConfig';
import { useStompClient, type StompSubscriptionSpec } from '../../hooks/useStompClient';
import { describeStompConnection, type StompConnectionSnapshot } from '../../lib/connectionStatus';

interface TelemetryEntry {
  timestamp: string;
  round?: number;
  loss: number;
  accuracy: number;
}

interface LogViewerProps {
  projectId: string;
  onClose: () => void;
}

// Telemetry cache (parallel to logStore) — preserves loss/accuracy sparklines
// across modal re-opens, keyed by projectId.
const telemetryCache = new Map<string, TelemetryEntry[]>();

function normalizeLevel(level?: string): 'INFO' | 'ERROR' | 'WARN' | 'DEBUG' {
  const upper = (level ?? 'INFO').toUpperCase();
  if (upper === 'ERROR' || upper === 'WARN' || upper === 'DEBUG') return upper;
  return 'INFO';
}

// Map the connection state to the one status-semantics scale: paused wins
// (explicit user action), then the honest STOMP phase — streaming -> running
// (accent), dropped-and-retrying -> pending (warning), never-yet-connected ->
// idle or error depending on whether a failure was actually observed.
function connectionStatus(isPaused: boolean, conn: StompConnectionSnapshot): {
  kind: ReturnType<typeof describeStompConnection>['kind'];
  label: string;
} {
  if (isPaused) return { kind: 'pending', label: 'Paused' };
  return describeStompConnection(conn, {
    live: 'Live Streaming',
    connecting: 'Connecting…',
    reconnecting: 'Reconnecting…',
    error: 'Connection lost',
  });
}

export function LogViewerV2({ projectId, onClose }: LogViewerProps) {
  const [logs, setLogs] = useState<StoredLogEntry[]>(() => logStore.get(projectId));
  const [telemetry, setTelemetry] = useState<TelemetryEntry[]>(
    () => telemetryCache.get(projectId) ?? []
  );
  const [isPaused, setIsPaused] = useState(false);
  const [filterError, setFilterError] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);
  // No pausedRef anymore — we always append to the store regardless of
  // pause state. "Pause" only affects auto-scroll (see effect below) so
  // the user can read history without losing in-flight messages.

  // Hydrate + subscribe to the shared log cache.
  useEffect(() => {
    setLogs(logStore.get(projectId));
    const unsubscribe = logStore.subscribe(projectId, (next) => setLogs(next));
    return unsubscribe;
  }, [projectId]);

  // Fetch historical logs once per project.
  useEffect(() => {
    if (logStore.hasLoadedHistorical(projectId)) return;
    api.fetchProjectLogs(projectId)
      .then((response) => {
        if (Array.isArray(response.data) && response.data.length > 0) {
          // ids are assigned by the store at merge time.
          const formatted = response.data.map((entry: any) => ({
            level: entry.level,
            message: entry.message,
            timestamp: entry.timestamp,
            stackTrace: entry.stackTrace,
          }));
          logStore.mergeHistorical(projectId, formatted);
        } else {
          logStore.markHistoricalLoaded(projectId);
        }
      })
      .catch((err) => console.error('Failed to fetch historical logs:', err));
  }, [projectId]);

  // Live WebSocket stream — lifecycle owned by useStompClient; this surface
  // only supplies the log-line + telemetry parsing.
  const logSubscriptions: StompSubscriptionSpec[] = [
    {
      topic: `/topic/logs/${projectId}`,
      onMessage: (message) => {
        // Always append — even while paused. "Pause" only freezes the
        // auto-scroll viewport (see effect below); messages received in
        // the paused window remain in the store and become visible again
        // when the user scrolls or resumes.
        try {
          const parsed = JSON.parse(message.body);
          const timeStr =
            parsed.timestamp || new Date().toLocaleTimeString('en-US', { hour12: false });

          logStore.append(projectId, {
            level: normalizeLevel(parsed.level),
            message: parsed.message ?? message.body,
            timestamp: timeStr,
            stackTrace: parsed.stackTrace,
          });

          // Check if payload is a RoundResultDto
          if (parsed.loss !== undefined && parsed.accuracy !== undefined) {
            const prev = telemetryCache.get(projectId) ?? [];
            const next = [
              ...prev.slice(-30),
              { timestamp: timeStr, round: parsed.serverRound, loss: parsed.loss, accuracy: parsed.accuracy },
            ];
            telemetryCache.set(projectId, next);
            setTelemetry(next);
          }
        } catch {
          logStore.append(projectId, {
            level: 'INFO',
            message: message.body,
            timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
          });
        }
      },
    },
  ];

  const connection = useStompClient({
    brokerURL: WS_BROKER_URL,
    subscriptions: logSubscriptions,
  });

  // Auto-scroll to newest entry (unless user paused).
  useEffect(() => {
    if (!isPaused && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, isPaused]);

  const filteredLogs = filterError
    ? logs.filter((l) => normalizeLevel(l.level) === 'ERROR')
    : logs;

  const handleClear = () => logStore.clear(projectId);

  const status = connectionStatus(isPaused, connection);
  const latest = telemetry.length > 0 ? telemetry[telemetry.length - 1] : null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md font-sans">
      <div className="bg-surface-1 border border-line w-full max-w-7xl h-[85vh] rounded-card flex flex-col overflow-hidden text-fg shadow-[0_30px_90px_-24px_rgba(0,0,0,0.95)]">

        {/* Header */}
        <div className="h-[60px] border-b border-hairline flex items-center justify-between px-6 bg-surface-1">
          <div className="flex items-center gap-3">
            <TerminalSquare strokeWidth={1.5} className="w-[18px] h-[18px] text-accent" />
            <h2 className="text-h4 font-display text-fg">Training activity</h2>
            <div className="w-px h-[18px] bg-hairline mx-2" />
            <StatusPill status={status.kind}>{status.label}</StatusPill>
          </div>
          <Button variant="secondary" size="sm" onClick={onClose}>
            Done
          </Button>
        </div>

        {/* Content */}
        <div className="flex-1 flex overflow-hidden">

          {/* Logs Pane */}
          <div className="flex-1 flex flex-col border-r border-hairline bg-canvas">
            {/* Toolbar */}
            <div className="h-[52px] border-b border-hairline flex items-center justify-between px-6">
              <div className="flex items-center gap-3">
                <Button
                  variant={isPaused ? 'primary' : 'secondary'}
                  size="sm"
                  onClick={() => setIsPaused(!isPaused)}
                >
                  {isPaused ? <Play strokeWidth={1.5} className="w-3.5 h-3.5 fill-current" /> : <Pause strokeWidth={1.5} className="w-3.5 h-3.5 fill-current" />}
                  {isPaused ? 'Resume' : 'Pause'}
                </Button>
                <Button
                  variant={filterError ? 'danger' : 'secondary'}
                  size="sm"
                  onClick={() => setFilterError(!filterError)}
                >
                  <Filter strokeWidth={1.5} className="w-3.5 h-3.5" />
                  Errors Only
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={handleClear}
                  title="Clear cached log entries"
                >
                  <Trash2 strokeWidth={1.5} className="w-3.5 h-3.5" />
                  Clear
                </Button>
              </div>
              <span className="text-label text-fg-subtle font-mono tabular-nums">{logs.length} events</span>
            </div>

            {/* Terminal Window */}
            <LogConsole className="flex-1 m-4 scroll-smooth">
              {filteredLogs.length === 0 && (
                <div className="flex items-center justify-center h-full text-fg-subtle">
                  Waiting for logs from project {projectId}…
                </div>
              )}
              {filteredLogs.map((log) => {
                const level = normalizeLevel(log.level);
                return (
                  // Keying by store-assigned id so prepended historical
                  // entries don't shift array indexes and confuse React.
                  <div key={log.id} className="flex hover:bg-surface-1 py-[2px] px-2 -mx-2 rounded-sm transition-colors font-mono">
                    <span className="text-fg-subtle w-[90px] shrink-0 select-none tabular-nums">{log.timestamp}</span>
                    <span className={cn(
                      "w-[60px] shrink-0 font-medium select-none",
                      level === 'INFO' && "text-accent",
                      level === 'ERROR' && "text-danger",
                      level === 'WARN' && "text-warning",
                      level === 'DEBUG' && "text-fg-muted"
                    )}>
                      {level}
                    </span>
                    <span className={cn(
                      "flex-1 break-words",
                      level === 'ERROR' ? "text-danger" : "text-fg",
                      level === 'WARN' && "text-warning"
                    )}>
                      {log.message}
                    </span>
                  </div>
                );
              })}
              <div ref={logsEndRef} />
            </LogConsole>
          </div>

          {/* Telemetry Pane */}
          <div className="w-[360px] flex flex-col shrink-0 bg-surface-1 border-l border-hairline">
            <div className="p-6 flex flex-col gap-6 h-full overflow-y-auto">
              <div>
                <h3 className="text-caption font-semibold uppercase tracking-widest text-fg-muted mb-1">Live numbers</h3>
              </div>

              <div className="bg-surface-2 rounded-card p-5 flex flex-col gap-4 border border-hairline">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2 text-fg-muted">
                    <Activity strokeWidth={1.5} className="w-4 h-4" />
                    <span className="text-label font-medium">Loss</span>
                  </div>
                  <span className="font-mono text-h4 tabular-nums text-fg">
                    {latest ? latest.loss.toFixed(4) : '---'}
                  </span>
                </div>
                <div className="h-[100px] w-full">
                  <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                    <LineChart data={telemetry}>
                      <Line type="stepAfter" dataKey="loss" stroke="var(--color-danger)" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-surface-2 rounded-card p-5 flex flex-col gap-4 border border-hairline">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2 text-fg-muted">
                    <Activity strokeWidth={1.5} className="w-4 h-4" />
                    <span className="text-label font-medium">Accuracy</span>
                  </div>
                  <span className="font-mono text-h4 tabular-nums text-fg">
                    {latest ? (latest.accuracy * 100).toFixed(2) + '%' : '---'}
                  </span>
                </div>
                <div className="h-[100px] w-full">
                  <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                    <LineChart data={telemetry}>
                      <Line type="monotone" dataKey="accuracy" stroke="var(--color-series-1)" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
