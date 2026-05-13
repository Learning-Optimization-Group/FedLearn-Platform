import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Boxes, Activity, Play, CheckCircle2, AlertCircle, Gauge, Brain } from 'lucide-react';
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
  const totalModels = summaries.length;
  const runningModels = summaries.filter((m) => m.running > 0).length;

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden">
      <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px)' }}>
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="font-display text-4xl font-semibold tracking-tight text-(--text-primary)">Model Portfolio</h1>
            <p className="text-sm text-(--text-secondary) mt-1">A cross-project view of architecture usage and stability.</p>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <span className="inline-flex items-center gap-2 rounded-full px-4 py-2" style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}>
              <Brain className="w-4 h-4 text-(--accent-primary)" /> {totalModels} model families
            </span>
            <span className="inline-flex items-center gap-2 rounded-full px-4 py-2" style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}>
              <Gauge className="w-4 h-4 text-emerald-500" /> {runningModels} actively training
            </span>
          </div>
        </motion.div>
      </div>

      <div className="flex-1 overflow-y-auto px-8 py-8">
        {error && (
          <div className="mb-6 px-5 py-3 rounded-2xl text-sm font-medium" style={{ backgroundColor: 'color-mix(in srgb, #ef4444 12%, transparent)', color: '#ef4444', border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)' }}>
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center h-64 text-(--text-secondary)">Loading models...</div>
        ) : summaries.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-(--text-secondary) gap-2">
            <p className="text-lg text-(--text-primary)">No models attached yet.</p>
            <p className="text-sm">Create a project to start model tracking.</p>
          </div>
        ) : (
          <motion.div
            initial="hidden"
            animate="visible"
            variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.06 } } }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {summaries.map((m) => (
              <motion.div key={`${m.modelType}-${m.modelName}`} variants={{ hidden: { opacity: 0, y: 12 }, visible: { opacity: 1, y: 0 } }}>
                <div className="rounded-3xl p-6 flex flex-col gap-4" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}>
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl inline-flex items-center justify-center" style={{ backgroundColor: 'color-mix(in srgb, var(--accent-primary) 10%, transparent)' }}>
                      <Boxes className="w-5 h-5 text-(--accent-primary)" />
                    </div>
                    <div className="min-w-0">
                      <h3 className="text-[18px] font-semibold tracking-tight truncate text-(--text-primary)">{m.modelName}</h3>
                      <p className="text-[13px] text-(--text-secondary)">{m.modelType}</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-3">
                    <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}>
                      <div className="flex items-center gap-1.5 text-(--text-secondary)">
                        <Activity className="w-3.5 h-3.5" />
                        <span className="text-[10px] uppercase tracking-wider font-semibold">Total</span>
                      </div>
                      <span className="text-[20px] font-semibold tracking-tight text-(--text-primary)">{m.projectCount}</span>
                    </div>
                    <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}>
                      <div className="flex items-center gap-1.5 text-emerald-500">
                        <Play className="w-3.5 h-3.5" />
                        <span className="text-[10px] uppercase tracking-wider font-semibold">Run</span>
                      </div>
                      <span className="text-[20px] font-semibold tracking-tight text-(--text-primary)">{m.running}</span>
                    </div>
                    <div className="rounded-xl p-3" style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}>
                      <div className="flex items-center gap-1.5 text-violet-500">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span className="text-[10px] uppercase tracking-wider font-semibold">Done</span>
                      </div>
                      <span className="text-[20px] font-semibold tracking-tight text-(--text-primary)">{m.completed}</span>
                    </div>
                  </div>

                  {m.failed > 0 && (
                    <div className="flex items-center gap-2 text-rose-500 text-[13px] font-medium">
                      <AlertCircle className="w-4 h-4" />
                      {m.failed} failed run{m.failed > 1 ? 's' : ''}
                    </div>
                  )}

                  {m.optimizers.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {m.optimizers.map((o) => (
                        <span key={o} className="text-[12px] font-medium px-2.5 py-1 rounded-full" style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}>
                          {o}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </motion.div>
        )}
      </div>
    </div>
  );
}
