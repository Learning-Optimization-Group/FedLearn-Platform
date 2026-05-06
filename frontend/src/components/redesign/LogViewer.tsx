// =============================================================================
// FedLearn Frontend — Redesigned LogViewer (Live Telemetry Dashboard)
// =============================================================================
// Wired to real WebSocket log stream from the backend, backed by logStore so
// closing & reopening the modal preserves prior session output.

import { useState, useEffect, useRef } from 'react';
import { Play, Pause, Filter, TerminalSquare, Trash2 } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line } from 'recharts';
import { cn } from '../../lib/utils';
import { Client as StompClient } from '@stomp/stompjs';
import { Activity } from 'lucide-react';
import * as api from '../../services/apiServices';
import { logStore, StoredLogEntry } from '../../services/logStore';

interface TelemetryEntry {
  timestamp: string;
  round?: number;
  loss: number;
  accuracy: number;
}

interface LogViewerProps {
  projectId: string;
  serverUrl: string;
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

export function LogViewerV2({ projectId, serverUrl, onClose }: LogViewerProps) {
  const [logs, setLogs] = useState<StoredLogEntry[]>(() => logStore.get(projectId));
  const [telemetry, setTelemetry] = useState<TelemetryEntry[]>(
    () => telemetryCache.get(projectId) ?? []
  );
  const [isPaused, setIsPaused] = useState(false);
  const [filterError, setFilterError] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
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

  // Live WebSocket stream.
  useEffect(() => {
    const wsUrl = serverUrl.replace(/^http/, 'ws');
    const client = new StompClient({
      brokerURL: `${wsUrl}/ws-logs`,
      reconnectDelay: 5000,
    });

    client.onConnect = () => {
      setIsConnected(true);
      client.subscribe(`/topic/logs/${projectId}`, (message) => {
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
      });
    };
    client.onDisconnect = () => setIsConnected(false);

    client.activate();
    return () => {
      if (client.active) client.deactivate();
    };
  }, [projectId, serverUrl]);

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

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/60 backdrop-blur-sm font-sans">
      <div className="bg-slate-900 border border-slate-700 w-full max-w-7xl h-[85vh] rounded-md shadow-2xl shadow-cyan-900/10 flex flex-col overflow-hidden text-slate-200">

        {/* Header */}
        <div className="h-[60px] border-b border-slate-800 flex items-center justify-between px-6 bg-slate-900/50">
          <div className="flex items-center gap-3">
            <TerminalSquare className="w-[18px] h-[18px] text-slate-400" />
            <h2 className="text-[17px] font-semibold tracking-tight text-slate-100">Telemetry Dashboard</h2>
            <div className="w-px h-[18px] bg-slate-700 mx-2" />
            <span className="flex h-2 w-2 relative">
              <span className={cn(
                "animate-ping absolute inline-flex h-2 w-2 rounded-full opacity-75",
                isPaused ? "bg-amber-500" : isConnected ? "bg-green-500" : "bg-rose-500"
              )} />
              <span className={cn(
                "relative inline-flex rounded-full h-2 w-2",
                isPaused ? "bg-amber-500" : isConnected ? "bg-green-500" : "bg-rose-500"
              )} />
            </span>
            <span className="text-[13px] text-slate-400 font-medium tracking-tight">
              {isPaused ? 'Paused' : isConnected ? 'Live Streaming' : 'Connecting…'}
            </span>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-100 bg-slate-800 hover:bg-slate-700 rounded-sm px-4 py-1.5 text-[13px] font-medium transition-colors border border-slate-700">
            Done
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 flex overflow-hidden">

          {/* Logs Pane */}
          <div className="flex-1 flex flex-col border-r border-slate-800 bg-slate-950/30">
            {/* Toolbar */}
            <div className="h-[52px] border-b border-slate-800 flex items-center justify-between px-6">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setIsPaused(!isPaused)}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 rounded-sm text-[13px] font-medium transition-colors border border-transparent",
                    isPaused ? "bg-amber-500/20 text-amber-500 border-amber-500/30" : "bg-slate-800 text-slate-200 hover:bg-slate-700 border-slate-700"
                  )}
                >
                  {isPaused ? <Play className="w-3.5 h-3.5 fill-current" /> : <Pause className="w-3.5 h-3.5 fill-current" />}
                  {isPaused ? 'Resume' : 'Pause'}
                </button>
                <button
                  onClick={() => setFilterError(!filterError)}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 rounded-sm text-[13px] font-medium transition-colors border border-transparent",
                    filterError ? "bg-rose-500/20 text-rose-500 border-rose-500/30" : "bg-slate-800 text-slate-200 hover:bg-slate-700 border-slate-700"
                  )}
                >
                  <Filter className="w-3.5 h-3.5" />
                  Errors Only
                </button>
                <button
                  onClick={handleClear}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-sm text-[13px] font-medium bg-slate-800 text-slate-200 hover:bg-slate-700 transition-colors border border-slate-700"
                  title="Clear cached log entries"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  Clear
                </button>
              </div>
              <span className="text-[13px] text-slate-500 font-mono tracking-tight">{logs.length} events</span>
            </div>

            {/* Terminal Window */}
            <div className="flex-1 overflow-y-auto bg-slate-950 m-4 rounded-sm p-5 font-mono text-[13px] leading-relaxed relative scroll-smooth border border-slate-800 shadow-inner">
              {filteredLogs.length === 0 && (
                <div className="flex items-center justify-center h-full text-slate-500">
                  Waiting for logs from project {projectId}…
                </div>
              )}
              {filteredLogs.map((log) => {
                const level = normalizeLevel(log.level);
                return (
                  // Keying by store-assigned id so prepended historical
                  // entries don't shift array indexes and confuse React.
                  <div key={log.id} className="flex hover:bg-slate-900/50 py-[2px] px-2 -mx-2 rounded transition-colors font-mono">
                    <span className="text-slate-500 w-[90px] shrink-0 select-none">{log.timestamp}</span>
                    <span className={cn(
                      "w-[60px] shrink-0 font-medium select-none",
                      level === 'INFO' && "text-cyan-400",
                      level === 'ERROR' && "text-rose-400",
                      level === 'WARN' && "text-amber-400",
                      level === 'DEBUG' && "text-slate-400"
                    )}>
                      {level}
                    </span>
                    <span className={cn(
                      "flex-1 break-words tracking-tight",
                      level === 'ERROR' ? "text-rose-400" : "text-slate-300",
                      level === 'WARN' && "text-amber-400"
                    )}>
                      {log.message}
                    </span>
                  </div>
                );
              })}
              <div ref={logsEndRef} />
            </div>
          </div>

          {/* Telemetry Pane */}
          <div className="w-[360px] flex flex-col shrink-0 bg-slate-900 border-l border-slate-800">
            <div className="p-6 flex flex-col gap-6 h-full overflow-y-auto">
              <div>
                <h3 className="text-[11px] font-semibold uppercase tracking-widest text-slate-400 mb-1">Live Metrics</h3>
              </div>

              <div className="bg-slate-950/50 rounded-md p-5 flex flex-col gap-4 border border-slate-800">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2 text-slate-400">
                    <Activity className="w-4 h-4" />
                    <span className="text-[13px] font-medium">Global Loss</span>
                  </div>
                  <span className="font-mono text-[20px] tracking-tighter font-semibold text-slate-100">
                    {telemetry.length > 0 ? telemetry[telemetry.length - 1].loss.toFixed(4) : '---'}
                  </span>
                </div>
                <div className="h-[100px] w-full">
                  <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                    <LineChart data={telemetry}>
                      <Line type="stepAfter" dataKey="loss" stroke="#f43f5e" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-slate-950/50 rounded-md p-5 flex flex-col gap-4 border border-slate-800">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2 text-slate-400">
                    <Activity className="w-4 h-4" />
                    <span className="text-[13px] font-medium">Global Accuracy</span>
                  </div>
                  <span className="font-mono text-[20px] tracking-tighter font-semibold text-slate-100">
                    {telemetry.length > 0 ? (telemetry[telemetry.length - 1].accuracy * 100).toFixed(2) + '%' : '---'}
                  </span>
                </div>
                <div className="h-[100px] w-full">
                  <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                    <LineChart data={telemetry}>
                      <Line type="monotone" dataKey="accuracy" stroke="#22c55e" strokeWidth={2.5} dot={false} isAnimationActive={false} />
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
