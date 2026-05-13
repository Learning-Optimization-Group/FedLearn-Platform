import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { ResponsiveContainer, LineChart, Line, YAxis, XAxis, Tooltip, CartesianGrid } from 'recharts';
import { ChartLine, Target, TrendingDown, Gauge } from 'lucide-react';
import * as api from '../../services/apiServices';
import type { Project, ProjectResult } from '../../services/apiServices';

export function TrainingInsightsView() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [resultsMap, setResultsMap] = useState<Record<string, ProjectResult[]>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    (async () => {
      try {
        const projectRes = await api.fetchProjects();
        const loadedProjects = Array.isArray(projectRes.data) ? projectRes.data : [];
        setProjects(loadedProjects);

        const settled = await Promise.allSettled(
          loadedProjects.map((p) => api.fetchProjectResults(p.id).then((res) => ({ id: p.id, rows: res.data })))
        );
        const map: Record<string, ProjectResult[]> = {};
        settled.forEach((s) => {
          if (s.status === 'fulfilled') {
            map[s.value.id] = s.value.rows;
          }
        });
        setResultsMap(map);
      } catch {
        setError('Failed to load training insights.');
      } finally {
        setIsLoading(false);
      }
    })();
  }, []);

  const insight = useMemo(() => {
    const rows = Object.values(resultsMap).flat();
    const latest = rows[rows.length - 1];
    const bestAccuracy = rows.length ? Math.max(...rows.map((r) => r.accuracy)) : 0;
    const bestLoss = rows.length ? Math.min(...rows.map((r) => r.loss)) : 0;
    const totalRounds = rows.length;

    const timeline = rows
      .slice(-30)
      .map((r) => ({
        round: r.serverRound,
        accuracy: Number((r.accuracy * 100).toFixed(2)),
        loss: Number(r.loss.toFixed(4)),
      }));

    return {
      totalRounds,
      bestAccuracy,
      bestLoss,
      latestAccuracy: latest?.accuracy ?? 0,
      latestLoss: latest?.loss ?? 0,
      timeline,
      activeProjects: projects.filter((p) => p.status === 'RUNNING').length,
    };
  }, [projects, resultsMap]);

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden">
      <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px)' }}>
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="font-display text-4xl font-semibold tracking-tight text-(--text-primary)">Training Insights</h1>
          <p className="text-sm text-(--text-secondary) mt-1">Portfolio-level progress across recent federated rounds.</p>
        </motion.div>
      </div>

      <div className="flex-1 overflow-y-auto px-8 py-8 space-y-6">
        {error && (
          <div className="px-5 py-3 rounded-2xl text-sm font-medium" style={{ backgroundColor: 'color-mix(in srgb, #ef4444 12%, transparent)', color: '#ef4444', border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)' }}>
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center h-64 text-(--text-secondary)">Loading insights...</div>
        ) : (
          <>
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
              <div className="rounded-2xl p-4" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)' }}>
                <div className="text-xs text-(--text-secondary)">Active Projects</div>
                <div className="text-2xl font-semibold text-(--text-primary)">{insight.activeProjects}</div>
              </div>
              <div className="rounded-2xl p-4" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)' }}>
                <div className="text-xs text-(--text-secondary)">Total Rounds</div>
                <div className="text-2xl font-semibold text-(--text-primary)">{insight.totalRounds}</div>
              </div>
              <div className="rounded-2xl p-4" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)' }}>
                <div className="text-xs text-(--text-secondary)">Best Accuracy</div>
                <div className="text-2xl font-semibold text-(--text-primary)">{insight.bestAccuracy ? `${(insight.bestAccuracy * 100).toFixed(1)}%` : '—'}</div>
              </div>
              <div className="rounded-2xl p-4" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)' }}>
                <div className="text-xs text-(--text-secondary)">Latest Loss</div>
                <div className="text-2xl font-semibold text-(--text-primary)">{insight.latestLoss ? insight.latestLoss.toFixed(4) : '—'}</div>
              </div>
              <div className="rounded-2xl p-4" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)' }}>
                <div className="text-xs text-(--text-secondary)">Best Loss</div>
                <div className="text-2xl font-semibold text-(--text-primary)">{insight.bestLoss ? insight.bestLoss.toFixed(4) : '—'}</div>
              </div>
            </div>

            <div className="rounded-3xl p-6" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-(--text-primary) inline-flex items-center gap-2"><ChartLine className="w-5 h-5 text-(--accent-primary)" /> Convergence Window</h2>
                <div className="text-sm text-(--text-secondary)">Last {insight.timeline.length} points</div>
              </div>

              {insight.timeline.length > 1 ? (
                <div className="h-[360px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={insight.timeline} margin={{ top: 20, right: 24, left: 12, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="4 4" stroke="color-mix(in srgb, var(--border-color) 80%, transparent)" vertical={false} />
                      <XAxis dataKey="round" tickLine={false} axisLine={false} tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
                      <YAxis yAxisId="left" tickLine={false} axisLine={false} tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
                      <YAxis yAxisId="right" orientation="right" tickLine={false} axisLine={false} tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'var(--background-secondary)',
                          border: '1px solid var(--border-color)',
                          borderRadius: 12,
                          color: 'var(--text-primary)',
                        }}
                      />
                      <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2.4} dot={false} />
                      <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="var(--accent-primary)" strokeWidth={2.4} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-[220px] flex items-center justify-center text-(--text-secondary)">Not enough rounds yet to chart convergence.</div>
              )}
            </div>

            <div className="grid md:grid-cols-3 gap-4">
              <div className="rounded-2xl p-4" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)' }}>
                <div className="inline-flex items-center gap-2 text-(--text-secondary)"><Target className="w-4 h-4" /> Objective</div>
                <p className="text-sm text-(--text-secondary) mt-2">Increase aggregate accuracy while holding loss volatility low across rounds.</p>
              </div>
              <div className="rounded-2xl p-4" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)' }}>
                <div className="inline-flex items-center gap-2 text-(--text-secondary)"><TrendingDown className="w-4 h-4" /> Loss Health</div>
                <p className="text-sm text-(--text-secondary) mt-2">Best observed loss: {insight.bestLoss ? insight.bestLoss.toFixed(4) : 'n/a'}. Monitor sudden upward spikes.</p>
              </div>
              <div className="rounded-2xl p-4" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)' }}>
                <div className="inline-flex items-center gap-2 text-(--text-secondary)"><Gauge className="w-4 h-4" /> Readiness</div>
                <p className="text-sm text-(--text-secondary) mt-2">Latest accuracy: {insight.latestAccuracy ? `${(insight.latestAccuracy * 100).toFixed(1)}%` : 'n/a'}. Pair with domain-level validation before deployment.</p>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
