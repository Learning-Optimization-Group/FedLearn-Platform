import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ResponsiveContainer, LineChart, Line, YAxis } from 'recharts';
import { Activity, Server, Trash2, Copy, Check, Play, Square, ChartLine, TerminalSquare } from 'lucide-react';
import { cn } from '../../lib/utils';
import type { Project, ProjectResult } from '../../services/apiServices';

interface ProjectCardProps {
  project: Project;
  results?: ProjectResult[];
  onOpenResults: () => void;
  onOpenLogs: () => void;
  onToggleServer: () => void;
  onEditProject: () => void;
  onDeleteProject: () => void;
}

function CopyButton({ text, label }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <button
      onClick={handleCopy}
      className="inline-flex items-center gap-1.5 text-[#86868b] hover:text-[#f5f5f7] transition-colors group/copy"
      title={`Copy ${label || text}`}
    >
      {label && <span className="text-[12px] font-medium">{label}:</span>}
      <code className="text-[12px] font-mono bg-[#2c2c2e] px-2 py-0.5 rounded-md text-[#f5f5f7] max-w-[120px] truncate">
        {text}
      </code>
      {copied ? (
        <Check className="w-3.5 h-3.5 text-[#32d74b]" />
      ) : (
        <Copy className="w-3.5 h-3.5 opacity-0 group-hover/copy:opacity-100 transition-opacity" />
      )}
    </button>
  );
}

export function ProjectCard({
  project,
  results = [],
  onOpenResults,
  onOpenLogs,
  onToggleServer,
  onEditProject,
  onDeleteProject,
}: ProjectCardProps) {
  const [confirmDelete, setConfirmDelete] = useState(false);

  const isRunning = project.status === 'RUNNING';
  const isCompleted = project.status === 'COMPLETED';
  const isFailed = project.status === 'FAILED';

  const accuracyTrend = results.slice(-10).map((r) => ({
    round: r.serverRound,
    accuracy: r.accuracy,
  }));

  const latestRound = results.length > 0 ? results[results.length - 1].serverRound : 0;
  const latestAccuracy = results.length > 0 ? results[results.length - 1].accuracy : 0;
  const latestLoss = results.length > 0 ? results[results.length - 1].loss : 0;

  const statusColor = isRunning
    ? 'text-[#0a84ff]'
    : isCompleted
      ? 'text-[#32d74b]'
      : isFailed
        ? 'text-[#ff453a]'
        : 'text-[#86868b]';

  const handleDelete = () => {
    setConfirmDelete(true);
    setTimeout(() => setConfirmDelete(false), 3000); // Reset after 3s
  };

  return (
    <div
      className="rounded-3xl p-6 flex flex-col gap-5 w-full transition-all duration-300 group"
      style={{
        background: 'var(--background-card)',
        border: '1px solid var(--border-color)',
        boxShadow: 'var(--shadow-soft)',
      }}
    >
      <div className="flex justify-between items-start">
        <div className="flex-1 min-w-0">
          <h3 className="text-[20px] font-semibold tracking-tight text-(--text-primary)">{project.name}</h3>
          <div className="flex items-center gap-2 mt-1">
            <span className={cn("inline-flex items-center gap-[6px] text-[13px] font-medium tracking-tight", statusColor)}>
              <span className={cn(
                "w-1.5 h-1.5 rounded-full",
                isRunning && "bg-[#0a84ff] animate-pulse",
                isCompleted && "bg-[#32d74b]",
                isFailed && "bg-[#ff453a]",
                !isRunning && !isCompleted && !isFailed && "bg-[#86868b]"
              )} />
              {project.status}
            </span>
            {project.myRelationship && (
              <span className={cn(
                'inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold uppercase tracking-wider',
                project.myRelationship === 'OWNER'
                  ? 'bg-blue-500/10 text-blue-500 border border-blue-500/20'
                  : project.myRelationship === 'MEMBER'
                    ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20'
                    : 'bg-purple-500/10 text-purple-500 border border-purple-500/20'
              )}>
                {project.myRelationship}
              </span>
            )}
          </div>
        </div>

        <div className="text-right">
          <div className="text-xs uppercase tracking-[0.18em] text-(--text-secondary)">Round</div>
          <div className="text-2xl font-semibold text-(--text-primary)">{latestRound || '—'}</div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <CopyButton text={project.id} label="ID" />
        {isRunning && project.serverPort && (
          <CopyButton text={String(project.serverPort)} label="Port" />
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div
          className="rounded-2xl p-4 flex flex-col justify-between gap-3"
          style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}
        >
          <div className="flex items-center text-(--text-secondary) gap-1.5">
            <Server className="w-[14px] h-[14px]" />
            <span className="text-[11px] font-semibold uppercase tracking-wider">Model</span>
          </div>
          <div className="text-[14px] font-medium text-(--text-primary) tracking-tight truncate">
            {project.modelName}
          </div>
          <div className="text-[12px] text-(--text-secondary) tracking-tight">
            {project.modelType} · {project.optimizer}
          </div>
        </div>

        <div
          className="rounded-2xl p-4 flex flex-col justify-between gap-2 relative overflow-hidden"
          style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}
        >
          <div className="flex items-center justify-between text-(--text-secondary)">
            <div className="flex items-center gap-1.5">
              <Activity className="w-[14px] h-[14px]" />
              <span className="text-[11px] font-semibold uppercase tracking-wider">Accuracy</span>
            </div>
            {accuracyTrend.length > 0 && (
              <span className="text-[13px] font-semibold tracking-tight text-(--text-primary)">
                {(accuracyTrend[accuracyTrend.length - 1].accuracy * 100).toFixed(1)}%
              </span>
            )}
          </div>
          <div className="h-8 w-full mt-auto">
            {accuracyTrend.length > 1 ? (
              <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                <LineChart data={accuracyTrend}>
                  <YAxis domain={['auto', 'auto']} hide />
                  <Line type="monotone" dataKey="accuracy" stroke="var(--accent-primary)" strokeWidth={2.5} dot={false} isAnimationActive={true} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-[11px] text-(--text-secondary)">
                No data yet
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div
          className="rounded-xl px-3 py-2"
          style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}
        >
          <div className="text-xs text-(--text-secondary)">Latest Accuracy</div>
          <div className="font-semibold text-(--text-primary)">{latestAccuracy ? `${(latestAccuracy * 100).toFixed(2)}%` : '—'}</div>
        </div>
        <div
          className="rounded-xl px-3 py-2"
          style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}
        >
          <div className="text-xs text-(--text-secondary)">Latest Loss</div>
          <div className="font-semibold text-(--text-primary)">{latestLoss ? latestLoss.toFixed(4) : '—'}</div>
        </div>
      </div>

      <div className="flex gap-3 mt-1">
        <button
          onClick={onOpenResults}
          className="flex-1 py-2.5 px-4 rounded-xl text-[14px] font-medium tracking-tight transition-colors border inline-flex items-center justify-center gap-2"
          style={{ backgroundColor: 'var(--background-secondary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
        >
          <ChartLine className="w-4 h-4" />
          View Results
        </button>
        <button
          onClick={onOpenLogs}
          className="flex-1 py-2.5 px-4 rounded-xl text-[14px] font-medium border tracking-tight transition-colors inline-flex items-center justify-center gap-2"
          style={{ backgroundColor: 'var(--background-secondary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
        >
          <TerminalSquare className="w-4 h-4" />
          View Logs
        </button>
        <button onClick={onToggleServer} disabled={isFailed} className={cn(
          "py-2.5 px-5 rounded-xl text-[14px] font-semibold tracking-tight transition-all border inline-flex items-center gap-2",
          isFailed
            ? "bg-slate-800/30 border-slate-700/50 text-slate-500 cursor-not-allowed"
            : isRunning
              ? "bg-rose-500/10 border-rose-500/30 text-rose-500 hover:bg-rose-500/20"
              : "bg-cyan-500/10 border-cyan-500/30 text-cyan-500 hover:bg-cyan-500/20"
        )}>
          {isRunning ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {isRunning ? 'Stop' : 'Start'}
        </button>
        <button
          onClick={confirmDelete ? onDeleteProject : handleDelete}
          className={cn(
            "py-2.5 px-4 rounded-xl text-[14px] font-semibold tracking-tight transition-all border inline-flex items-center gap-2",
            confirmDelete
              ? "bg-rose-500/20 border-rose-500/40 text-rose-500"
              : "bg-transparent border-(--border-color) text-(--text-secondary) hover:text-rose-500 hover:border-rose-400/40"
          )}
        >
          <Trash2 className="w-4 h-4" />
          {confirmDelete ? 'Confirm' : 'Delete'}
        </button>
      </div>

      <div className="flex items-center justify-between mt-1">
        <Link
          to={`/projects/${project.id}`}
          className="text-xs font-medium text-(--text-secondary) hover:text-(--accent-primary) transition-colors"
        >
          View Details →
        </Link>
        <button
          onClick={onEditProject}
          className="text-xs font-medium text-(--text-secondary) hover:text-(--accent-primary)"
        >
          Edit Project Details
        </button>
      </div>
    </div>
  );
}
