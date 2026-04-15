// =============================================================================
// FedLearn Frontend — Redesigned LogViewer (Live Telemetry Dashboard)
// =============================================================================
// Wired to real WebSocket log stream from the backend.

import { useState, useEffect, useRef } from 'react';
import { Play, Pause, Filter, TerminalSquare } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line } from 'recharts';
import { cn } from '../../lib/utils';
import { Client as StompClient } from '@stomp/stompjs';
import { Activity } from 'lucide-react';

interface LogEntry {
  id: number;
  level: 'INFO' | 'ERROR' | 'WARN' | 'DEBUG';
  message: string;
  timestamp: string;
}

interface TelemetryEntry {
  timestamp: string;
  loss: number;
  accuracy: number;
}

interface LogViewerProps {
  projectId: string;
  serverUrl: string;
  onClose: () => void;
}

export function LogViewerV2({ projectId, serverUrl, onClose }: LogViewerProps) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [telemetry, setTelemetry] = useState<TelemetryEntry[]>([]);
  const [isPaused, setIsPaused] = useState(false);
  const [filterError, setFilterError] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const idCounterRef = useRef(0);

  // Connect to real WebSocket log stream
  useEffect(() => {
    const wsUrl = serverUrl.replace(/^http/, 'ws');
    const client = new StompClient({
      brokerURL: `${wsUrl}/ws-logs`,
      reconnectDelay: 5000,
    });

    client.onConnect = () => {
      client.subscribe(`/topic/logs/${projectId}`, (message) => {
        if (isPaused) return;
        try {
          const logData = JSON.parse(message.body);
          idCounterRef.current++;
          const now = new Date();
          const timeStr = now.toLocaleTimeString('en-US', { hour12: false });

          const entry: LogEntry = {
            id: idCounterRef.current,
            level: logData.level || 'INFO',
            message: logData.message || message.body,
            timestamp: logData.timestamp || timeStr,
          };

          setLogs((prev) => [...prev.slice(-200), entry]);

          // Extract telemetry if the log contains metrics
          if (logData.loss !== undefined && logData.accuracy !== undefined) {
            setTelemetry((prev) => [
              ...prev.slice(-30),
              { timestamp: timeStr, loss: logData.loss, accuracy: logData.accuracy },
            ]);
          }
        } catch {
          // Raw text log
          idCounterRef.current++;
          setLogs((prev) => [
            ...prev.slice(-200),
            {
              id: idCounterRef.current,
              level: 'INFO',
              message: message.body,
              timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
            },
          ]);
        }
      });
    };

    client.activate();
    return () => {
      if (client.active) client.deactivate();
    };
  }, [projectId, serverUrl, isPaused]);

  // Auto-scroll
  useEffect(() => {
    if (!isPaused && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, isPaused]);

  const filteredLogs = filterError ? logs.filter((l) => l.level === 'ERROR') : logs;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xl font-sans">
      <div className="bg-[rgba(28,28,30,0.85)] border border-[rgba(255,255,255,0.1)] w-full max-w-7xl h-[85vh] rounded-[32px] shadow-[0_20px_50px_rgba(0,0,0,0.5)] flex flex-col overflow-hidden text-[#f5f5f7]">

        {/* Header */}
        <div className="h-[60px] border-b border-[rgba(255,255,255,0.1)] flex items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <TerminalSquare className="w-[18px] h-[18px] text-[#86868b]" />
            <h2 className="text-[17px] font-semibold tracking-tight text-[#f5f5f7]">Telemetry Dashboard</h2>
            <div className="w-px h-[18px] bg-[rgba(255,255,255,0.1)] mx-2" />
            <span className="flex h-2 w-2">
              <span className={cn("animate-ping absolute inline-flex h-2 w-2 rounded-full opacity-75", isPaused ? "bg-[#ff9f0a]" : "bg-[#32d74b]")} />
              <span className={cn("relative inline-flex rounded-full h-2 w-2", isPaused ? "bg-[#ff9f0a]" : "bg-[#32d74b]")} />
            </span>
            <span className="text-[13px] text-[#86868b] font-medium tracking-tight">
              {isPaused ? 'Paused' : 'Live Streaming'}
            </span>
          </div>
          <button onClick={onClose} className="text-[#86868b] hover:text-[#f5f5f7] bg-[rgba(255,255,255,0.05)] hover:bg-[rgba(255,255,255,0.1)] rounded-full px-4 py-1.5 text-[13px] font-medium transition-colors">
            Done
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 flex overflow-hidden">

          {/* Logs Pane */}
          <div className="flex-1 flex flex-col border-r border-[rgba(255,255,255,0.1)]">
            {/* Toolbar */}
            <div className="h-[52px] border-b border-[rgba(255,255,255,0.05)] flex items-center justify-between px-6">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setIsPaused(!isPaused)}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 rounded-full text-[13px] font-medium transition-colors",
                    isPaused ? "bg-[#0a84ff]/20 text-[#0a84ff]" : "bg-[rgba(255,255,255,0.05)] text-[#f5f5f7] hover:bg-[rgba(255,255,255,0.1)]"
                  )}
                >
                  {isPaused ? <Play className="w-3.5 h-3.5 fill-current" /> : <Pause className="w-3.5 h-3.5 fill-current" />}
                  {isPaused ? 'Resume' : 'Pause'}
                </button>
                <button
                  onClick={() => setFilterError(!filterError)}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 rounded-full text-[13px] font-medium transition-colors",
                    filterError ? "bg-[#ff453a]/20 text-[#ff453a]" : "bg-[rgba(255,255,255,0.05)] text-[#f5f5f7] hover:bg-[rgba(255,255,255,0.1)]"
                  )}
                >
                  <Filter className="w-3.5 h-3.5" />
                  Errors Only
                </button>
              </div>
              <span className="text-[13px] text-[#86868b] font-mono tracking-tight">{logs.length} events</span>
            </div>

            {/* Terminal Window */}
            <div className="flex-1 overflow-y-auto bg-black m-4 rounded-[20px] p-5 font-mono text-[13px] leading-relaxed relative scroll-smooth border border-[rgba(255,255,255,0.1)]">
              {filteredLogs.length === 0 && (
                <div className="flex items-center justify-center h-full text-[#86868b]">
                  Waiting for logs from project {projectId}...
                </div>
              )}
              {filteredLogs.map((log) => (
                <div key={log.id} className="flex hover:bg-[rgba(255,255,255,0.05)] py-[2px] px-2 -mx-2 rounded-md transition-colors">
                  <span className="text-[#86868b] w-[90px] shrink-0 select-none">{log.timestamp}</span>
                  <span className={cn(
                    "w-[60px] shrink-0 font-medium select-none",
                    log.level === 'INFO' && "text-[#0a84ff]",
                    log.level === 'ERROR' && "text-[#ff453a]",
                    log.level === 'WARN' && "text-[#ff9f0a]",
                    log.level === 'DEBUG' && "text-[#86868b]"
                  )}>
                    {log.level}
                  </span>
                  <span className={cn(
                    "flex-1 break-words tracking-tight",
                    log.level === 'ERROR' ? "text-[#ff453a]" : "text-[#f5f5f7]",
                    log.level === 'WARN' && "text-[#ff9f0a]"
                  )}>
                    {log.message}
                  </span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>

          {/* Telemetry Pane */}
          <div className="w-[360px] flex flex-col shrink-0">
            <div className="p-6 flex flex-col gap-6 h-full overflow-y-auto">
              <div>
                <h3 className="text-[11px] font-semibold uppercase tracking-widest text-[#86868b] mb-1">Live Metrics</h3>
              </div>

              {/* Loss Sparkline */}
              <div className="bg-[rgba(0,0,0,0.3)] rounded-[20px] p-5 flex flex-col gap-4">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2 text-[#86868b]">
                    <Activity className="w-4 h-4" />
                    <span className="text-[13px] font-medium">Global Loss</span>
                  </div>
                  <span className="font-mono text-[20px] tracking-tighter font-semibold text-[#f5f5f7]">
                    {telemetry.length > 0 ? telemetry[telemetry.length - 1].loss.toFixed(4) : '---'}
                  </span>
                </div>
                <div className="h-[100px] w-full">
                  <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                    <LineChart data={telemetry}>
                      <Line type="stepAfter" dataKey="loss" stroke="#bf5af2" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Accuracy Sparkline */}
              <div className="bg-[rgba(0,0,0,0.3)] rounded-[20px] p-5 flex flex-col gap-4">
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2 text-[#86868b]">
                    <Activity className="w-4 h-4" />
                    <span className="text-[13px] font-medium">Global Accuracy</span>
                  </div>
                  <span className="font-mono text-[20px] tracking-tighter font-semibold text-[#f5f5f7]">
                    {telemetry.length > 0 ? (telemetry[telemetry.length - 1].accuracy * 100).toFixed(2) + '%' : '---'}
                  </span>
                </div>
                <div className="h-[100px] w-full">
                  <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                    <LineChart data={telemetry}>
                      <Line type="monotone" dataKey="accuracy" stroke="#32d74b" strokeWidth={2.5} dot={false} isAnimationActive={false} />
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
