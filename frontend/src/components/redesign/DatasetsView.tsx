import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Database, Layers, Radar, FolderCog } from 'lucide-react';
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
  const totalDomains = summaries.length;

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden">
      <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px)' }}>
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="font-display text-4xl font-semibold tracking-tight text-(--text-primary)">Data Domains</h1>
            <p className="text-sm text-(--text-secondary) mt-1">Track where training data categories are concentrated.</p>
          </div>
          <span className="inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm" style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}>
            <FolderCog className="w-4 h-4 text-(--accent-primary)" /> {totalDomains} active domains
          </span>
        </motion.div>
      </div>

      <div className="flex-1 overflow-y-auto px-8 py-8">
        {error && (
          <div className="mb-6 px-5 py-3 rounded-2xl text-sm font-medium" style={{ backgroundColor: 'color-mix(in srgb, #ef4444 12%, transparent)', color: '#ef4444', border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)' }}>
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center h-64 text-(--text-secondary)">Loading datasets...</div>
        ) : summaries.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-(--text-secondary) gap-2">
            <p className="text-lg text-(--text-primary)">No dataset domains registered.</p>
            <p className="text-sm">Create projects to populate this view.</p>
          </div>
        ) : (
          <motion.div
            initial="hidden"
            animate="visible"
            variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.06 } } }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {summaries.map((s) => (
              <motion.div key={s.modelType} variants={{ hidden: { opacity: 0, y: 12 }, visible: { opacity: 1, y: 0 } }}>
                <div className="rounded-3xl p-6 flex flex-col gap-4" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}>
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl inline-flex items-center justify-center" style={{ backgroundColor: 'color-mix(in srgb, #8b5cf6 12%, transparent)' }}>
                      <Database className="w-5 h-5 text-violet-500" />
                    </div>
                    <div>
                      <h3 className="text-[18px] font-semibold tracking-tight text-(--text-primary)">{s.modelType}</h3>
                      <p className="text-[13px] text-(--text-secondary)">{s.projectCount} linked project{s.projectCount > 1 ? 's' : ''}</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}>
                      <div className="flex items-center gap-1.5 text-(--text-secondary)">
                        <Layers className="w-3.5 h-3.5" />
                        <span className="text-[10px] uppercase tracking-wider font-semibold">Unique Models</span>
                      </div>
                      <span className="text-[20px] font-semibold tracking-tight text-(--text-primary)">{s.uniqueModels}</span>
                    </div>
                    <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}>
                      <div className="flex items-center gap-1.5 text-emerald-500">
                        <Radar className="w-3.5 h-3.5" />
                        <span className="text-[10px] uppercase tracking-wider font-semibold">Running</span>
                      </div>
                      <span className="text-[20px] font-semibold tracking-tight text-(--text-primary)">{s.runningCount}</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        )}
      </div>
    </div>
  );
}
