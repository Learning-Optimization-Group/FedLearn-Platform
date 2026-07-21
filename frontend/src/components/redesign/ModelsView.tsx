// =============================================================================
// FedLearn Frontend — Models View (Ledger design system)
// =============================================================================
// Aggregates model architectures currently in rotation across all projects.

import { useEffect, useMemo, useState } from 'react';
import { Boxes, AlertCircle } from 'lucide-react';
import * as api from '../../services/apiServices';
import type { Project } from '../../services/apiServices';
import { Card, MetricTile, Skeleton } from '../ui';
import { PageHeader } from './PageHeader';

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
        setError('Failed to load models.');
      } finally {
        setIsLoading(false);
      }
    })();
  }, []);

  const summaries = useMemo(() => summarize(projects), [projects]);

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
      <PageHeader title="Models" subtitle="The models you're training across your projects." />

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[1400px] px-6 py-6 md:px-10 reveal">
          {error && (
            <div className="mb-6 flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
              <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
              {error}
            </div>
          )}

          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[0, 1, 2].map((i) => (
                <Card key={i} padding="lg" className="flex flex-col gap-4">
                  <div className="flex items-center gap-3">
                    <Skeleton className="h-11 w-11 rounded-lg" />
                    <Skeleton className="h-5 w-32" />
                  </div>
                  <Skeleton className="h-16 w-full" />
                </Card>
              ))}
            </div>
          ) : summaries.length === 0 ? (
            <div className="flex flex-col items-center justify-center text-center gap-4 pt-16 md:pt-24">
              <div className="grid h-12 w-12 place-items-center rounded-pill bg-surface-2 text-fg-muted">
                <Boxes className="h-6 w-6" strokeWidth={1.5} />
              </div>
              <div className="max-w-sm">
                <p className="text-h4 font-semibold text-fg">No models yet</p>
                <p className="text-caption text-fg-muted mt-1">
                  Create a project and your models will show up here.
                </p>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {summaries.map((m) => (
                <Card key={`${m.modelType}-${m.modelName}`} padding="lg" className="flex flex-col gap-4">
                  <div className="flex items-center gap-3">
                    <span className="icon-tile flex-shrink-0">
                      <Boxes strokeWidth={1.5} className="w-5 h-5" />
                    </span>
                    <div className="min-w-0">
                      <h3 className="text-h4 font-semibold text-fg truncate">{m.modelName}</h3>
                      <p className="text-caption text-fg-muted">{m.modelType}</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4 border-t border-hairline pt-4">
                    <MetricTile label="Total" value={m.projectCount} />
                    <MetricTile label="Training" value={m.running} />
                    <MetricTile label="Completed" value={m.completed} />
                  </div>

                  {m.failed > 0 && (
                    <p className="flex items-center gap-2 text-label font-medium text-danger">
                      <AlertCircle strokeWidth={1.5} className="w-4 h-4" />
                      {m.failed} run{m.failed > 1 ? 's' : ''} had an error
                    </p>
                  )}

                  {m.optimizers.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {m.optimizers.map((o) => (
                        <span key={o} className="chip">
                          {o}
                        </span>
                      ))}
                    </div>
                  )}
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
