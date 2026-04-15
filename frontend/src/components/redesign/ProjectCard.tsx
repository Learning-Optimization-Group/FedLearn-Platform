// =============================================================================
// FedLearn Frontend — Redesigned ProjectCard (Apple-inspired)
// =============================================================================
// Full feature parity: delete, copy ID, copy port, start/stop, results, logs.

import { useState } from 'react';
import { ResponsiveContainer, LineChart, Line, YAxis } from 'recharts';
import { Activity, Server, Trash2, Copy, Check, MoreHorizontal } from 'lucide-react';
import { cn } from '../../lib/utils';
import type { Project, ProjectResult } from '../../services/apiServices';

interface ProjectCardProps {
  project: Project;
  results?: ProjectResult[];
  onOpenResults: () => void;
  onOpenLogs: () => void;
  onToggleServer: () => void;
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
  onDeleteProject,
}: ProjectCardProps) {
  const [showMenu, setShowMenu] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const isRunning = project.status === 'RUNNING';
  const isCompleted = project.status === 'COMPLETED';
  const isFailed = project.status === 'FAILED';

  // Build accuracy trend from real results
  const accuracyTrend = results.slice(-10).map((r) => ({
    round: r.serverRound,
    accuracy: r.accuracy,
  }));

  // Progress calculation
  const latestRound = results.length > 0 ? results[results.length - 1].serverRound : 0;
  const totalRounds = 100;
  const progress = Math.min((latestRound / totalRounds) * 100, 100);

  // Circular progress ring
  const radius = 26;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  const statusColor = isRunning
    ? 'text-[#0a84ff]'
    : isCompleted
      ? 'text-[#32d74b]'
      : isFailed
        ? 'text-[#ff453a]'
        : 'text-[#86868b]';

  const ringColor = isRunning
    ? 'text-[#0a84ff]'
    : isCompleted
      ? 'text-[#32d74b]'
      : 'text-[#ff453a]';

  const handleDelete = () => {
    if (confirmDelete) {
      onDeleteProject();
      setConfirmDelete(false);
      setShowMenu(false);
    } else {
      setConfirmDelete(true);
      setTimeout(() => setConfirmDelete(false), 3000); // Reset after 3s
    }
  };

  return (
    <div className="bg-[#1c1c1e] rounded-[24px] p-6 flex flex-col gap-5 text-[#f5f5f7] w-full font-sans transition-all hover:bg-[#2c2c2e]/60 duration-300 group relative">
      {/* Header Row */}
      <div className="flex justify-between items-start">
        <div className="flex-1 min-w-0">
          <h3 className="text-[20px] font-semibold tracking-tight">{project.name}</h3>
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
          </div>
        </div>

        {/* Actions Menu */}
        <div className="relative">
          <button
            onClick={() => setShowMenu(!showMenu)}
            className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-[#3a3a3c] text-[#86868b] hover:text-[#f5f5f7] transition-colors"
          >
            <MoreHorizontal className="w-4 h-4" />
          </button>
          {showMenu && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => { setShowMenu(false); setConfirmDelete(false); }} />
              <div className="absolute right-0 top-10 z-20 bg-[#2c2c2e] border border-[rgba(255,255,255,0.1)] rounded-2xl py-2 w-48 shadow-[0_10px_30px_rgba(0,0,0,0.5)]">
                <button
                  onClick={handleDelete}
                  className={cn(
                    "w-full px-4 py-2.5 text-left text-[14px] font-medium transition-colors flex items-center gap-2",
                    confirmDelete
                      ? "text-[#ff453a] bg-[#ff453a]/10"
                      : "text-[#ff453a] hover:bg-[rgba(255,255,255,0.05)]"
                  )}
                >
                  <Trash2 className="w-4 h-4" />
                  {confirmDelete ? 'Confirm Delete?' : 'Delete Project'}
                </button>
              </div>
            </>
          )}
        </div>

        {/* Circular Progress Ring */}
        <div className="relative flex items-center justify-center w-14 h-14 ml-2">
          <svg className="w-full h-full transform -rotate-90">
            <circle cx="28" cy="28" r={radius} stroke="currentColor" strokeWidth="4.5" fill="transparent" className="text-[#2c2c2e]" />
            <circle
              cx="28" cy="28" r={radius} stroke="currentColor" strokeWidth="4.5" fill="transparent"
              strokeDasharray={circumference} strokeDashoffset={strokeDashoffset} strokeLinecap="round"
              className={cn("transition-all duration-1000 ease-out", ringColor)}
            />
          </svg>
          <div className="absolute flex flex-col items-center justify-center text-center">
            <span className="text-[12px] font-bold tracking-tighter text-[#f5f5f7]">{Math.round(progress)}%</span>
          </div>
        </div>
      </div>

      {/* Project ID & Port — Copyable */}
      <div className="flex flex-wrap items-center gap-3">
        <CopyButton text={project.id} label="ID" />
        {isRunning && project.serverPort && (
          <CopyButton text={String(project.serverPort)} label="Port" />
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-4">
        {/* Model Info */}
        <div className="bg-[#2c2c2e]/40 rounded-2xl p-4 flex flex-col justify-between gap-3">
          <div className="flex items-center text-[#86868b] gap-1.5">
            <Server className="w-[14px] h-[14px]" />
            <span className="text-[11px] font-semibold uppercase tracking-wider">Model</span>
          </div>
          <div className="text-[14px] font-medium text-[#f5f5f7] tracking-tight truncate">
            {project.modelName}
          </div>
          <div className="text-[12px] text-[#86868b] tracking-tight">
            {project.modelType} · {project.optimizer}
          </div>
        </div>

        {/* Accuracy Sparkline */}
        <div className="bg-[#2c2c2e]/40 rounded-2xl p-4 flex flex-col justify-between gap-2 relative overflow-hidden">
          <div className="flex items-center justify-between text-[#86868b]">
            <div className="flex items-center gap-1.5">
              <Activity className="w-[14px] h-[14px]" />
              <span className="text-[11px] font-semibold uppercase tracking-wider">Accuracy</span>
            </div>
            {accuracyTrend.length > 0 && (
              <span className="text-[13px] font-semibold tracking-tight text-[#f5f5f7]">
                {(accuracyTrend[accuracyTrend.length - 1].accuracy * 100).toFixed(1)}%
              </span>
            )}
          </div>
          <div className="h-8 w-full mt-auto">
            {accuracyTrend.length > 1 ? (
              <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                <LineChart data={accuracyTrend}>
                  <YAxis domain={['auto', 'auto']} hide />
                  <Line type="monotone" dataKey="accuracy" stroke="#0a84ff" strokeWidth={2.5} dot={false} isAnimationActive={true} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-[11px] text-[#86868b]">
                No data yet
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 mt-1">
        <button onClick={onOpenResults} className="flex-1 bg-[#2c2c2e] hover:bg-[#3a3a3c] text-[#f5f5f7] py-[11px] px-4 rounded-full text-[15px] font-medium tracking-tight transition-colors">
          View Results
        </button>
        <button onClick={onOpenLogs} disabled={!isRunning} className={cn(
          "flex-1 py-[11px] px-4 rounded-full text-[15px] font-medium tracking-tight transition-colors",
          isRunning ? "bg-[#2c2c2e] hover:bg-[#3a3a3c] text-[#f5f5f7]" : "bg-[#2c2c2e]/50 text-[#86868b] cursor-not-allowed"
        )}>
          Logs
        </button>
        <button onClick={onToggleServer} disabled={isFailed} className={cn(
          "py-[11px] px-5 rounded-full text-[15px] font-medium tracking-tight transition-all",
          isFailed
            ? "bg-[#3a3a3c]/30 text-[#86868b] cursor-not-allowed"
            : isRunning
              ? "bg-[#ff453a]/20 text-[#ff453a] hover:bg-[#ff453a]/30"
              : "bg-[#32d74b]/20 text-[#32d74b] hover:bg-[#32d74b]/30"
        )}>
          {isRunning ? 'Stop' : 'Start'}
        </button>
      </div>
    </div>
  );
}
