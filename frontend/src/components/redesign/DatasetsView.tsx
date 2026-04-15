// =============================================================================
// FedLearn Frontend — V2 Datasets View
// =============================================================================
// Aggregates dataset types (modelType) currently referenced by active projects.

import { useEffect, useMemo, useState } from 'react';
import { Database, Layers } from 'lucide-react';
import * as api from '../../services/apiServices';
import type { Project } from '../../services/apiServices';

interface DatasetSummary {
  modelType: string;
  projectCount: number;
  uniqueModels: number;
  runningCount: number;
}

function summarize(projects: Project[]): DatasetSummary[] {
  const byType = new Map<string, DatasetSummary & { models: Set<string> }>();
  for (const p of projects) {
    const existing = byType.get(p.modelType);
    if (existing) {
      existing.projectCount += 1;
      existing.models.add(p.modelName);
      if (p.status === 'RUNNING') existing.runningCount += 1;
      existing.uniqueModels = existing.models.size;
    } else {
      byType.set(p.modelType, {
        modelType: p.modelType,
        projectCount: 1,
        uniqueModels: 1,
        runningCount: p.status === 'RUNNING' ? 1 : 0,
        models: new Set([p.modelName]),
      });
    }
  }
  return Array.from(byType.values())
    .map(({ models, ...rest }) => rest)
    .sort((a, b) => b.projectCount - a.projectCount);
}

export function DatasetsView() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    (async () => {
      try {
        const res = await api.fetchProjects();
        setProjects(Array.isArray(res.data) ? res.data : []);
      } catch {
        setError('Failed to fetch datasets.');
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
          <h1 className="text-[28px] font-semibold tracking-tight">Datasets</h1>
          <p className="text-[15px] text-[#86868b] mt-0.5 tracking-tight">
            Data domains actively consumed by federated projects.
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
            Loading datasets…
          </div>
        ) : summaries.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-[#86868b] gap-2">
            <p className="text-[17px]">No dataset types registered.</p>
            <p className="text-[14px]">
              Create a project with a model type to start populating this view.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {summaries.map((s) => (
              <div
                key={s.modelType}
                className="bg-[#1c1c1e] rounded-[24px] p-6 flex flex-col gap-4 border border-[rgba(255,255,255,0.05)] hover:bg-[#2c2c2e]/60 transition-all"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-[#bf5af2]/10 text-[#bf5af2] flex items-center justify-center">
                    <Database className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-[17px] font-semibold tracking-tight">{s.modelType}</h3>
                    <p className="text-[13px] text-[#86868b] tracking-tight">
                      {s.projectCount} project{s.projectCount > 1 ? 's' : ''}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-[#2c2c2e]/40 rounded-xl p-3 flex flex-col gap-1">
                    <div className="flex items-center gap-1.5 text-[#0a84ff]">
                      <Layers className="w-3.5 h-3.5" />
                      <span className="text-[10px] uppercase tracking-wider font-semibold">
                        Unique Models
                      </span>
                    </div>
                    <span className="text-[20px] font-semibold tracking-tight">
                      {s.uniqueModels}
                    </span>
                  </div>
                  <div className="bg-[#2c2c2e]/40 rounded-xl p-3 flex flex-col gap-1">
                    <span className="text-[10px] uppercase tracking-wider font-semibold text-[#32d74b]">
                      Running
                    </span>
                    <span className="text-[20px] font-semibold tracking-tight">
                      {s.runningCount}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
