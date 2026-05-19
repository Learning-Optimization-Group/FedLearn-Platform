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
    ? 'var(--signal)'
    : isCompleted
      ? 'var(--info)'
      : isFailed
        ? 'var(--danger)'
        : 'var(--fg-3)';

  const handleDelete = () => {
    setConfirmDelete(true);
    setTimeout(() => setConfirmDelete(false), 3000); // Reset after 3s
  };

  return (
    <div
      className="rounded-[16px] p-[20px] flex flex-col w-full transition-all duration-300 relative group"
      style={{
        background: 'var(--background-card)',
        border: '1px solid var(--border-color)',
        minHeight: 200,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = 'var(--border-strong)';
        e.currentTarget.style.transform = 'translateY(-2px)';
        e.currentTarget.style.boxShadow = '0 8px 28px oklch(0 0 0 / 0.32)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = 'var(--border-color)';
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = 'none';
      }}
    >
      <div className="flex justify-between items-start mb-4">
        <div className="flex-1 min-w-0 pr-4">
          <div className="flex items-center gap-1.5 mb-1.5">
            <span className="font-mono text-[10.5px] font-medium tracking-[0.12em] uppercase text-(--text-secondary)">
              {project.myRelationship || 'PROJECT'}
            </span>
          </div>
          <h3 className="text-[17px] font-medium tracking-tight text-(--text-primary) m-0">{project.name}</h3>
          <div className="flex flex-wrap items-center gap-3 mt-3">
             <CopyButton text={project.id} label="ID" />
             {isRunning && project.serverPort && (
               <CopyButton text={String(project.serverPort)} label="PORT" />
             )}
          </div>
        </div>
        
        <div 
          className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-[11px] font-medium tracking-wide uppercase border"
          style={{
             borderColor: `color-mix(in srgb, ${statusColor} 30%, transparent)`,
             backgroundColor: `color-mix(in srgb, ${statusColor} 10%, transparent)`,
             color: statusColor
          }}
        >
          <span className={cn("w-1.5 h-1.5 rounded-full", isRunning && "animate-pulse")} style={{ backgroundColor: statusColor }} />
          {project.status}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mt-auto">
        <div className="flex flex-col gap-1.5">
          <div className="font-mono text-[10.5px] font-medium tracking-[0.12em] uppercase text-(--text-secondary)">ACCURACY TREND</div>
          <div className="h-[40px] w-full mt-1">
            {accuracyTrend.length > 1 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={accuracyTrend}>
                  <YAxis domain={['auto', 'auto']} hide />
                  <Line type="stepAfter" dataKey="accuracy" stroke="var(--accent-primary)" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center text-[11px] text-(--text-secondary) h-full font-mono">No data</div>
            )}
          </div>
        </div>

        <div className="flex flex-col items-end gap-1.5 text-right">
          <div className="font-mono text-[10.5px] font-medium tracking-[0.12em] uppercase text-(--text-secondary)">LATEST RESULTS</div>
          <div className="font-mono text-[16px] font-medium text-(--text-primary)">
            {latestAccuracy ? `${(latestAccuracy * 100).toFixed(2)}%` : '—'}
          </div>
          <div className="font-mono text-[11px] text-(--text-secondary)">
            Loss: {latestLoss ? latestLoss.toFixed(4) : '—'}
          </div>
        </div>
      </div>

      <div className="mt-5 pt-4 border-t border-(--border-color) flex items-center justify-between font-mono text-[11px] text-(--text-secondary)">
        <span>{project.modelName}</span>
        <span>Round {latestRound || 0}</span>
      </div>

      <div className="flex gap-2 mt-4">
        <button
          onClick={onOpenResults}
          className="flex-1 py-2 rounded-lg text-[12px] font-medium tracking-tight transition-colors border inline-flex items-center justify-center gap-1.5"
          style={{ backgroundColor: 'var(--background-secondary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
        >
          <ChartLine className="w-3.5 h-3.5" />
          Results
        </button>
        <button
          onClick={onOpenLogs}
          className="flex-1 py-2 rounded-lg text-[12px] font-medium tracking-tight transition-colors border inline-flex items-center justify-center gap-1.5"
          style={{ backgroundColor: 'var(--background-secondary)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
        >
          <TerminalSquare className="w-3.5 h-3.5" />
          Logs
        </button>
        <button onClick={onToggleServer} disabled={isFailed} className={cn(
          "flex-1 py-2 rounded-lg text-[12px] font-medium tracking-tight transition-colors border inline-flex items-center justify-center gap-1.5",
          isFailed
            ? "bg-transparent border-(--border-color) text-(--text-secondary) opacity-50 cursor-not-allowed"
            : isRunning
              ? "bg-(--destructive) text-white border-transparent"
              : "bg-(--accent-primary) text-(--primary-foreground) border-transparent hover:brightness-110"
        )}>
          {isRunning ? <Square className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          {isRunning ? 'Stop' : 'Start'}
        </button>
      </div>

      <div className="flex items-center justify-between mt-3 pt-3 border-t border-(--border-color)">
        <div className="flex gap-4">
          <Link
            to={`/projects/${project.id}`}
            className="text-[11px] font-medium text-(--text-secondary) hover:text-(--accent-primary) transition-colors"
          >
            Details →
          </Link>
          <button
            onClick={onEditProject}
            className="text-[11px] font-medium text-(--text-secondary) hover:text-(--accent-primary) transition-colors"
          >
            Edit
          </button>
        </div>
        <button
          onClick={confirmDelete ? onDeleteProject : handleDelete}
          className={cn(
            "text-[11px] font-medium transition-colors inline-flex items-center gap-1",
            confirmDelete
              ? "text-(--destructive)"
              : "text-(--text-secondary) hover:text-(--destructive)"
          )}
        >
          <Trash2 className="w-3 h-3" />
          {confirmDelete ? 'Confirm' : 'Delete'}
        </button>
      </div>
    </div>
  );
}
