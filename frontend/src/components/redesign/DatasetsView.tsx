// =============================================================================
// FedLearn Frontend — Data View (Ledger design system)
// =============================================================================
// Aggregates dataset types (modelType) currently referenced by active projects.

import { useEffect, useMemo, useState } from 'react';
import { Database, AlertCircle } from 'lucide-react';
import * as api from '../../services/apiServices';
import type { Project } from '../../services/apiServices';
import { Card, MetricTile, Skeleton } from '../ui';
import { PageHeader } from './PageHeader';

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
    .map(({ models: _models, ...rest }) => rest) // strip the internal Set; keep the rest
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
        setError('Failed to load data.');
      } finally {
        setIsLoading(false);
      }
    })();
  }, []);

  const summaries = useMemo(() => summarize(projects), [projects]);

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
      <PageHeader title="Data" subtitle="The kinds of data your projects learn from — never shared, only learned from." />

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[1400px] px-6 py-6 md:px-10">
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
                    <Skeleton className="h-5 w-28" />
                  </div>
                  <Skeleton className="h-16 w-full" />
                </Card>
              ))}
            </div>
          ) : summaries.length === 0 ? (
            <div className="flex flex-col items-center justify-center text-center gap-4 pt-16 md:pt-24">
              <div className="grid h-12 w-12 place-items-center rounded-pill bg-surface-2 text-fg-muted">
                <Database className="h-6 w-6" strokeWidth={1.5} />
              </div>
              <div className="max-w-sm">
                <p className="text-h4 font-semibold text-fg">No data yet</p>
                <p className="text-caption text-fg-muted mt-1">
                  Create a project and the data it learns from will show up here.
                </p>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {summaries.map((s) => (
                <Card key={s.modelType} padding="lg" className="flex flex-col gap-4">
                  <div className="flex items-center gap-3">
                    <span className="icon-tile flex-shrink-0">
                      <Database strokeWidth={1.5} className="w-5 h-5" />
                    </span>
                    <div className="min-w-0">
                      <h3 className="text-h4 font-semibold text-fg truncate">{s.modelType}</h3>
                      <p className="text-caption text-fg-muted">
                        {s.projectCount} project{s.projectCount > 1 ? 's' : ''}
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 border-t border-hairline pt-4">
                    <MetricTile label="Models" value={s.uniqueModels} />
                    <MetricTile label="Training" value={s.runningCount} />
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
