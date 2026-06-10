// =============================================================================
// FedLearn Frontend — Models View (Ember design system)
// =============================================================================
// Aggregates model architectures currently in rotation across all projects.

import { useEffect, useMemo, useState } from 'react';
import { Boxes, Activity, Play, CheckCircle2, AlertCircle } from 'lucide-react';
import * as api from '../../services/apiServices';
import type { Project } from '../../services/apiServices';
import { Card, Skeleton } from '../ui';
import { BrandMark } from '../brand';
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

      <div className="flex-1 overflow-y-auto px-6 md:px-10 py-8 bg-canvas">
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
                  <Skeleton className="h-11 w-11 rounded-xl" />
                  <Skeleton className="h-5 w-32" />
                </div>
                <Skeleton className="h-16 w-full" />
              </Card>
            ))}
          </div>
        ) : summaries.length === 0 ? (
          <div className="flex flex-col items-center justify-center text-center gap-5 mt-16 md:mt-24">
            <div className="grid h-20 w-20 place-items-center rounded-card border border-hairline bg-surface-1">
              <BrandMark size={48} />
            </div>
            <div className="max-w-sm">
              <p className="text-h4 font-display text-fg">No models yet</p>
              <p className="text-body text-fg-muted mt-1.5">
                Create a project and your models will show up here.
              </p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {summaries.map((m) => (
              <Card
                key={`${m.modelType}-${m.modelName}`}
                padding="lg"
                className="flex flex-col gap-4 transition-colors duration-[160ms] hover:bg-surface-2 hover:border-accent/25"
              >
                <div className="flex items-center gap-3">
                  <span className="icon-tile flex-shrink-0">
                    <Boxes strokeWidth={1.5} className="w-5 h-5" />
                  </span>
                  <div className="min-w-0">
                    <h3 className="text-h4 font-display text-fg truncate">{m.modelName}</h3>
                    <p className="text-label text-fg-muted">{m.modelType}</p>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-surface-2 border border-hairline rounded-lg p-3 flex flex-col gap-1">
                    <div className="flex items-center gap-1.5 text-fg-muted">
                      <Activity strokeWidth={1.5} className="w-3.5 h-3.5" />
                      <span className="text-caption uppercase tracking-wide font-semibold">Total</span>
                    </div>
                    <span className="text-h4 font-mono tabular-nums text-fg">{m.projectCount}</span>
                  </div>
                  <div className="bg-surface-2 border border-hairline rounded-lg p-3 flex flex-col gap-1">
                    <div className="flex items-center gap-1.5 text-accent">
                      <Play strokeWidth={1.5} className="w-3.5 h-3.5" />
                      <span className="text-caption uppercase tracking-wide font-semibold">Training</span>
                    </div>
                    <span className="text-h4 font-mono tabular-nums text-fg">{m.running}</span>
                  </div>
                  <div className="bg-surface-2 border border-hairline rounded-lg p-3 flex flex-col gap-1">
                    <div className="flex items-center gap-1.5 text-success">
                      <CheckCircle2 strokeWidth={1.5} className="w-3.5 h-3.5" />
                      <span className="text-caption uppercase tracking-wide font-semibold">Done</span>
                    </div>
                    <span className="text-h4 font-mono tabular-nums text-fg">{m.completed}</span>
                  </div>
                </div>

                {m.failed > 0 && (
                  <div className="flex items-center gap-2 text-danger text-label font-medium">
                    <AlertCircle strokeWidth={1.5} className="w-4 h-4" />
                    {m.failed} run{m.failed > 1 ? 's' : ''} had an error
                  </div>
                )}

                {m.optimizers.length > 0 && (
                  <div className="flex flex-wrap gap-2">
                    {m.optimizers.map((o) => (
                      <span
                        key={o}
                        className="text-caption font-medium px-2.5 py-1 rounded-pill bg-surface-2 border border-hairline text-fg-muted"
                      >
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
  );
}
