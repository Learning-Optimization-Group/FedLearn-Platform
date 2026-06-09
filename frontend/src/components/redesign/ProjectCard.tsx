// =============================================================================
// FedLearn Frontend — Redesigned ProjectCard (Instrument design system)
// =============================================================================
// Full feature parity: delete, copy ID, copy port, start/stop, results, logs.

import { useState } from 'react';
import { ResponsiveContainer, LineChart, Line, YAxis } from 'recharts';
import { Activity, Server, Trash2, Copy, Check, MoreHorizontal, Edit3 } from 'lucide-react';
import { cn } from '../../lib/utils';
import { Card, Button, StatusPill, ConfirmDialog, type StatusKind } from '../ui';
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

/** Map domain status -> the 5 Instrument status kinds. */
function toStatusKind(status: Project['status']): StatusKind {
  switch (status) {
    case 'RUNNING':
      return 'running';
    case 'COMPLETED':
      return 'completed';
    case 'FAILED':
      return 'error';
    default:
      return 'idle';
  }
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
      className="inline-flex items-center gap-1.5 text-fg-muted hover:text-fg transition-colors group/copy"
      title={`Copy ${label || text}`}
    >
      {label && <span className="text-caption font-medium">{label}:</span>}
      <code className="text-caption font-mono tabular-nums bg-surface-2 px-2 py-0.5 rounded-sm text-fg max-w-[120px] truncate">
        {text}
      </code>
      {copied ? (
        <Check className="w-3.5 h-3.5 text-success" strokeWidth={1.5} />
      ) : (
        <Copy className="w-3.5 h-3.5 opacity-0 group-hover/copy:opacity-100 transition-opacity" strokeWidth={1.5} />
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

  // Ring uses the status semantics: running -> accent, completed -> success,
  // failed -> danger, otherwise muted. (text-* drives the SVG currentColor.)
  const ringColor = isRunning
    ? 'text-accent'
    : isCompleted
      ? 'text-success'
      : isFailed
        ? 'text-danger'
        : 'text-fg-muted';

  return (
    <Card
      padding="lg"
      className="flex flex-col gap-5 text-fg w-full font-sans transition-colors duration-[240ms] hover:bg-surface-2 hover:border-line group relative"
    >
      {/* Header Row */}
      <div className="flex justify-between items-start">
        <div className="flex-1 min-w-0">
          <h3 className="text-h4 font-semibold tracking-tight truncate">{project.name}</h3>
          <div className="flex items-center gap-2 mt-1">
            <StatusPill status={toStatusKind(project.status)}>{project.status}</StatusPill>
          </div>
        </div>

        {/* Actions Menu */}
        <div className="relative">
          <button
            onClick={() => setShowMenu(!showMenu)}
            className="w-8 h-8 flex items-center justify-center rounded-pill hover:bg-surface-3 text-fg-muted hover:text-fg transition-colors"
          >
            <MoreHorizontal className="w-4 h-4" strokeWidth={1.5} />
          </button>
          {showMenu && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setShowMenu(false)} />
              <div className="absolute right-0 top-10 z-20 bg-surface-2 border border-hairline rounded-md py-1 w-48">
                <button
                  onClick={() => {
                    onEditProject();
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left text-body font-medium transition-colors flex items-center gap-2 text-fg hover:bg-surface-3"
                >
                  <Edit3 className="w-4 h-4" strokeWidth={1.5} />
                  Edit Project
                </button>
                <div className="h-px bg-hairline my-1" />
                <button
                  onClick={() => {
                    setConfirmDelete(true);
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left text-body font-medium transition-colors flex items-center gap-2 text-danger hover:bg-surface-3"
                >
                  <Trash2 className="w-4 h-4" strokeWidth={1.5} />
                  Delete Project
                </button>
              </div>
            </>
          )}
        </div>

        {/* Circular Progress Ring */}
        <div className="relative flex items-center justify-center w-14 h-14 ml-2">
          <svg className="w-full h-full transform -rotate-90">
            <circle cx="28" cy="28" r={radius} stroke="currentColor" strokeWidth="4.5" fill="transparent" className="text-surface-3" />
            <circle
              cx="28" cy="28" r={radius} stroke="currentColor" strokeWidth="4.5" fill="transparent"
              strokeDasharray={circumference} strokeDashoffset={strokeDashoffset} strokeLinecap="round"
              className={cn("transition-all duration-[240ms] ease-out", ringColor)}
            />
          </svg>
          <div className="absolute flex flex-col items-center justify-center text-center">
            <span className="text-caption font-mono tabular-nums font-bold text-fg">{Math.round(progress)}%</span>
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
        <div className="bg-surface-2 border border-hairline rounded-md p-4 flex flex-col justify-between gap-3">
          <div className="flex items-center text-fg-muted gap-1.5">
            <Server className="w-[14px] h-[14px]" strokeWidth={1.5} />
            <span className="text-caption font-semibold uppercase tracking-wider">Model</span>
          </div>
          <div className="text-body font-medium text-fg tracking-tight truncate">
            {project.modelName}
          </div>
          <div className="text-caption text-fg-muted tracking-tight">
            {project.modelType} · {project.optimizer}
          </div>
        </div>

        {/* Accuracy Sparkline */}
        <div className="bg-surface-2 border border-hairline rounded-md p-4 flex flex-col justify-between gap-2 relative overflow-hidden">
          <div className="flex items-center justify-between text-fg-muted">
            <div className="flex items-center gap-1.5">
              <Activity className="w-[14px] h-[14px]" strokeWidth={1.5} />
              <span className="text-caption font-semibold uppercase tracking-wider">Accuracy</span>
            </div>
            {accuracyTrend.length > 0 && (
              <span className="text-label font-mono tabular-nums font-semibold text-fg">
                {(accuracyTrend[accuracyTrend.length - 1].accuracy * 100).toFixed(1)}%
              </span>
            )}
          </div>
          <div className="h-8 w-full mt-auto">
            {accuracyTrend.length > 1 ? (
              <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                <LineChart data={accuracyTrend}>
                  <YAxis domain={['auto', 'auto']} hide />
                  <Line type="monotone" dataKey="accuracy" stroke="var(--color-series-1)" strokeWidth={2.5} dot={false} isAnimationActive={true} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-caption text-fg-muted">
                No data yet
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 mt-1">
        <Button variant="secondary" onClick={onOpenResults} className="flex-1">
          View Results
        </Button>
        <Button variant="secondary" onClick={onOpenLogs} className="flex-1">
          View Logs
        </Button>
        <Button
          variant={isRunning ? 'danger' : 'primary'}
          onClick={onToggleServer}
          disabled={isFailed}
        >
          {isRunning ? 'Stop' : 'Start'}
        </Button>
      </div>

      <ConfirmDialog
        open={confirmDelete}
        title="Delete project?"
        message={`This permanently deletes "${project.name}" and its results. This cannot be undone.`}
        confirmLabel="Delete"
        cancelLabel="Cancel"
        danger
        onConfirm={() => {
          onDeleteProject();
          setConfirmDelete(false);
        }}
        onCancel={() => setConfirmDelete(false)}
      />
    </Card>
  );
}
