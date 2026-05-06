// =============================================================================
// FedLearn Frontend — V2 Models View
// =============================================================================
// Aggregates model architectures currently in rotation across all projects.

import { useEffect, useMemo, useState } from 'react';
import { Boxes, Activity, Play, CheckCircle2, AlertCircle } from 'lucide-react';
import * as api from '../../services/apiServices';
import type { Project } from '../../services/apiServices';

interface ModelSummary {
  modelName: string;
  modelType: string;
  projectCount: number;
  running: number;
  completed: number;
  failed: number;
  optimizers: string[];
}

function summarize(projects: Project[]): ModelSummary[] {
  const map = new Map<string, ModelSummary>();
  for (const p of projects) {
    const key = `${p.modelType}::${p.modelName}`;
    const existing = map.get(key);
    if (existing) {
      existing.projectCount += 1;
      if (p.status === 'RUNNING') existing.running += 1;
      if (p.status === 'COMPLETED') existing.completed += 1;
      if (p.status === 'FAILED') existing.failed += 1;
      if (p.optimizer && !existing.optimizers.includes(p.optimizer)) {
        existing.optimizers.push(p.optimizer);
      }
    } else {
      map.set(key, {
        modelName: p.modelName,
        modelType: p.modelType,
        projectCount: 1,
        running: p.status === 'RUNNING' ? 1 : 0,
        completed: p.status === 'COMPLETED' ? 1 : 0,
        failed: p.status === 'FAILED' ? 1 : 0,
        optimizers: p.optimizer ? [p.optimizer] : [],
      });
    }
  }
  return Array.from(map.values()).sort((a, b) => b.projectCount - a.projectCount);
}

export function ModelsView() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    (async () => {
      try {
        const res = await api.fetchProjects();
        setProjects(Array.isArray(res.data) ? res.data : []);
      } catch {
        setError('Failed to fetch models.');
      } finally {
        setIsLoading(false);
      }
    })();
  }, []);

  const summaries = useMemo(() => summarize(projects), [projects]);

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-black text-[#f5f5f7] font-sans">
      <div className="h-24 flex items-center justify-between px-10 border-b border-[#2c2c2e] bg-[rgba(0,0,0,0.65)] backdrop-blur-3xl saturate-[1.8] sticky top-0 z-20">
        <div>
          <h1 className="text-[28px] font-semibold tracking-tight">Models</h1>
          <p className="text-[15px] text-[#86868b] mt-0.5 tracking-tight">
            Architectures currently running across federated projects.
          </p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-10 py-10 bg-black">
        {error && (
          <div className="mb-6 px-5 py-3 rounded-2xl bg-[#ff453a]/10 text-[#ff453a] text-[14px] font-medium">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center h-64 text-[#86868b]">
            Loading models…
          </div>
        ) : summaries.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-[#86868b] gap-2">
            <p className="text-[17px]">No models attached yet.</p>
            <p className="text-[14px]">Create a project to start tracking a model here.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {summaries.map((m) => (
              <div
                key={`${m.modelType}-${m.modelName}`}
                className="bg-[#1c1c1e] rounded-[24px] p-6 flex flex-col gap-4 border border-[rgba(255,255,255,0.05)] hover:bg-[#2c2c2e]/60 transition-all"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-[#0a84ff]/10 text-[#0a84ff] flex items-center justify-center">
                    <Boxes className="w-5 h-5" />
                  </div>
                  <div className="min-w-0">
                    <h3 className="text-[17px] font-semibold tracking-tight truncate">
                      {m.modelName}
                    </h3>
                    <p className="text-[13px] text-[#86868b] tracking-tight">
                      {m.modelType}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-[#2c2c2e]/40 rounded-xl p-3 flex flex-col gap-1">
                    <div className="flex items-center gap-1.5 text-[#0a84ff]">
                      <Activity className="w-3.5 h-3.5" />
                      <span className="text-[10px] uppercase tracking-wider font-semibold">
                        Total
                      </span>
                    </div>
                    <span className="text-[20px] font-semibold tracking-tight">
                      {m.projectCount}
                    </span>
                  </div>
                  <div className="bg-[#2c2c2e]/40 rounded-xl p-3 flex flex-col gap-1">
                    <div className="flex items-center gap-1.5 text-[#32d74b]">
                      <Play className="w-3.5 h-3.5" />
                      <span className="text-[10px] uppercase tracking-wider font-semibold">
                        Running
                      </span>
                    </div>
                    <span className="text-[20px] font-semibold tracking-tight">
                      {m.running}
                    </span>
                  </div>
                  <div className="bg-[#2c2c2e]/40 rounded-xl p-3 flex flex-col gap-1">
                    <div className="flex items-center gap-1.5 text-[#bf5af2]">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span className="text-[10px] uppercase tracking-wider font-semibold">
                        Done
                      </span>
                    </div>
                    <span className="text-[20px] font-semibold tracking-tight">
                      {m.completed}
                    </span>
                  </div>
                </div>

                {m.failed > 0 && (
                  <div className="flex items-center gap-2 text-[#ff453a] text-[13px] font-medium">
                    <AlertCircle className="w-4 h-4" />
                    {m.failed} failed run{m.failed > 1 ? 's' : ''}
                  </div>
                )}

                {m.optimizers.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {m.optimizers.map((o) => (
                      <span
                        key={o}
                        className="text-[12px] font-medium px-2.5 py-1 rounded-full bg-[#2c2c2e] text-[#f5f5f7]"
                      >
                        {o}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
