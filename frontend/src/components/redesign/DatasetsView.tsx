// =============================================================================
// FedLearn Frontend — V2 Datasets View
// =============================================================================
// Aggregates dataset types (modelType) currently referenced by active projects.

import { useEffect, useMemo, useState } from 'react';
import { Database, Layers } from 'lucide-react';
import * as api from '../../services/apiServices';
import type { Project } from '../../services/apiServices';
import { Card } from '../ui';

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
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
      <div className="h-24 flex items-center justify-between px-10 border-b border-hairline bg-canvas/65 backdrop-blur-xl sticky top-0 z-20">
        <div>
          <h1 className="text-h2 text-fg">Datasets</h1>
          <p className="text-body text-fg-muted mt-0.5">
            Data domains actively consumed by federated projects.
          </p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-10 py-10 bg-canvas">
        {error && (
          <div className="mb-6 px-5 py-3 rounded-card bg-surface-1 border border-hairline text-danger text-body font-medium">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center h-64 text-fg-muted">
            Loading datasets…
          </div>
        ) : summaries.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-fg-muted gap-2">
            <p className="text-h4">No dataset types registered.</p>
            <p className="text-body">
              Create a project with a model type to start populating this view.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {summaries.map((s) => (
              <Card
                key={s.modelType}
                padding="lg"
                className="flex flex-col gap-4 hover:bg-surface-2 transition-colors duration-[160ms]"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-pill bg-surface-2 text-accent flex items-center justify-center">
                    <Database strokeWidth={1.5} className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-h4 text-fg">{s.modelType}</h3>
                    <p className="text-label text-fg-muted">
                      {s.projectCount} project{s.projectCount > 1 ? 's' : ''}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-surface-2 rounded-sm p-3 flex flex-col gap-1">
                    <div className="flex items-center gap-1.5 text-fg-muted">
                      <Layers strokeWidth={1.5} className="w-3.5 h-3.5" />
                      <span className="text-caption uppercase tracking-wide font-semibold">
                        Unique Models
                      </span>
                    </div>
                    <span className="text-h4 font-mono tabular-nums text-fg">
                      {s.uniqueModels}
                    </span>
                  </div>
                  <div className="bg-surface-2 rounded-sm p-3 flex flex-col gap-1">
                    <span className="text-caption uppercase tracking-wide font-semibold text-accent">
                      Running
                    </span>
                    <span className="text-h4 font-mono tabular-nums text-fg">
                      {s.runningCount}
                    </span>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
