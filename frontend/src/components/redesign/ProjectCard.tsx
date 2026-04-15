// =============================================================================
// FedLearn Frontend — Redesigned ProjectCard
// =============================================================================
// Displays project with circular progress ring + accuracy sparkline.
// Wired to real Project type from apiServices.

import { ResponsiveContainer, LineChart, Line, YAxis } from 'recharts';
import { Activity, Server } from 'lucide-react';
import { cn } from '../../lib/utils';
import type { Project, ProjectResult } from '../../services/apiServices';

interface ProjectCardProps {
  project: Project;
  results?: ProjectResult[];
  onOpenResults: () => void;
  onOpenLogs: () => void;
  onToggleServer: () => void;
}

export function ProjectCard({
  project,
  results = [],
  onOpenResults,
  onOpenLogs,
  onToggleServer,
}: ProjectCardProps) {
  const isRunning = project.status === 'RUNNING';
  const isCompleted = project.status === 'COMPLETED';
  const isFailed = project.status === 'FAILED';

  // Build accuracy trend from real results
  const accuracyTrend = results.slice(-10).map((r) => ({
    round: r.serverRound,
    accuracy: r.accuracy,
  }));

  // Progress calculation — use latest round from results if available
  const latestRound = results.length > 0 ? results[results.length - 1].serverRound : 0;
  const totalRounds = 100; // Default — will be dynamic when backend exposes it
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

  return (
    <div className="bg-[#1c1c1e] rounded-[24px] p-6 flex flex-col gap-6 text-[#f5f5f7] w-full font-sans transition-all hover:bg-[#2c2c2e]/60 duration-300 group">
      <div className="flex justify-between items-start">
        <div>
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

        {/* Circular Progress Ring */}
        <div className="relative flex items-center justify-center w-14 h-14">
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

        {/* Global Accuracy Sparkline */}
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

      <div className="flex gap-3 mt-2">
        <button onClick={onOpenResults} className="flex-1 bg-[#2c2c2e] hover:bg-[#3a3a3c] text-[#f5f5f7] py-[11px] px-4 rounded-full text-[15px] font-medium tracking-tight transition-colors">
          View Results
        </button>
        <button onClick={onOpenLogs} disabled={!isRunning} className={cn(
          "flex-1 py-[11px] px-4 rounded-full text-[15px] font-medium tracking-tight transition-colors",
          isRunning ? "bg-[#2c2c2e] hover:bg-[#3a3a3c] text-[#f5f5f7]" : "bg-[#2c2c2e]/50 text-[#86868b] cursor-not-allowed"
        )}>
          Logs
        </button>
        <button onClick={onToggleServer} className={cn(
          "py-[11px] px-5 rounded-full text-[15px] font-medium tracking-tight transition-all",
          isRunning
            ? "bg-[#ff453a]/20 text-[#ff453a] hover:bg-[#ff453a]/30"
            : "bg-[#32d74b]/20 text-[#32d74b] hover:bg-[#32d74b]/30"
        )}>
          {isRunning ? 'Stop' : 'Start'}
        </button>
      </div>
    </div>
  );
}
