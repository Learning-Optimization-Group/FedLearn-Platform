// =============================================================================
// FedLearn Frontend — Redesigned ResultsModal
// =============================================================================
// Wired to real ProjectResult[] from apiServices.

import { useState } from 'react';
import { ResponsiveContainer, LineChart, Line, YAxis, XAxis, Tooltip, CartesianGrid } from 'recharts';
import { X, Trophy, Timer, TrendingDown, Table, LineChart as ChartIcon } from 'lucide-react';
import { cn } from '../../lib/utils';
import type { ProjectResult } from '../../services/apiServices';
import { Button, MetricTile } from '../ui';

interface ResultsModalProps {
  isOpen: boolean;
  onClose: () => void;
  projectName: string;
  results: ProjectResult[];
}

export function ResultsModalV2({ isOpen, onClose, projectName, results }: ResultsModalProps) {
  const [activeTab, setActiveTab] = useState<'chart' | 'table'>('chart');

  if (!isOpen) return null;

  const hasResults = results.length > 0;
  const chartData = results.map((r) => ({
    round: r.serverRound,
    loss: r.loss,
    accuracy: r.accuracy,
  }));

  const bestAccuracy = hasResults ? Math.max(...results.map((r) => r.accuracy)) : 0;
  const finalLoss = hasResults ? results[results.length - 1].loss : 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 lg:p-8 bg-canvas/70 backdrop-blur-sm font-sans">
      <div className="bg-surface-1 border border-hairline w-full max-w-6xl h-full max-h-[90vh] rounded-card flex flex-col overflow-hidden text-fg">

        {/* Header */}
        <div className="flex items-center justify-between p-8 pb-6 border-b border-hairline">
          <div>
            <h2 className="text-h2 text-fg">{projectName} — Results</h2>
            <p className="text-body text-fg-muted mt-1">
              {hasResults ? `${results.length} rounds completed.` : 'No training rounds recorded yet.'}
            </p>
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={onClose}
            aria-label="Close"
            className="w-8 px-0"
          >
            <X strokeWidth={1.5} className="w-4 h-4" />
          </Button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-8 pb-8 flex flex-col gap-6 scroll-smooth">

          {!hasResults && (
            <div className="flex flex-col items-center justify-center flex-1 text-center gap-3 py-16">
              <div className="w-14 h-14 rounded-card bg-surface-2 flex items-center justify-center">
                <ChartIcon strokeWidth={1.5} className="w-6 h-6 text-fg-muted" />
              </div>
              <h3 className="text-h4 text-fg">No results yet</h3>
              <p className="text-body text-fg-muted max-w-md">
                Results appear here once the project has completed at least one
                federated training round. Start the server and connect clients
                to produce data.
              </p>
              <Button variant="secondary" onClick={onClose} className="mt-4">
                Close
              </Button>
            </div>
          )}

          {hasResults && <>

          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-surface-2 rounded-card p-8 border border-hairline">
              <div className="flex items-center gap-2 text-success mb-3">
                <Trophy strokeWidth={1.5} className="w-5 h-5" />
                <span className="text-caption font-semibold tracking-wide uppercase text-fg-muted">Best Accuracy</span>
              </div>
              <MetricTile label="" value={`${(bestAccuracy * 100).toFixed(2)}%`} />
            </div>

            <div className="bg-surface-2 rounded-card p-8 border border-hairline">
              <div className="flex items-center gap-2 text-danger mb-3">
                <TrendingDown strokeWidth={1.5} className="w-5 h-5" />
                <span className="text-caption font-semibold tracking-wide uppercase text-fg-muted">Final Loss</span>
              </div>
              <MetricTile label="" value={finalLoss.toFixed(4)} />
            </div>

            <div className="bg-surface-2 rounded-card p-8 border border-hairline">
              <div className="flex items-center gap-2 text-accent mb-3">
                <Timer strokeWidth={1.5} className="w-5 h-5" />
                <span className="text-caption font-semibold tracking-wide uppercase text-fg-muted">Total Rounds</span>
              </div>
              <MetricTile label="" value={results.length} />
            </div>
          </div>

          {/* Tabs */}
          <div className="flex items-center gap-3 border-b border-hairline pb-2">
            <button
              onClick={() => setActiveTab('chart')}
              className={cn(
                "flex items-center gap-2 px-5 h-9 rounded-md text-body font-medium transition-colors duration-[160ms]",
                activeTab === 'chart' ? "bg-surface-2 text-accent border border-hairline" : "text-fg-muted hover:bg-surface-2 hover:text-fg border border-transparent"
              )}
            >
              <ChartIcon strokeWidth={1.5} className="w-4 h-4" />
              Performance Chart
            </button>
            <button
              onClick={() => setActiveTab('table')}
              className={cn(
                "flex items-center gap-2 px-5 h-9 rounded-md text-body font-medium transition-colors duration-[160ms]",
                activeTab === 'table' ? "bg-surface-2 text-accent border border-hairline" : "text-fg-muted hover:bg-surface-2 hover:text-fg border border-transparent"
              )}
            >
              <Table strokeWidth={1.5} className="w-4 h-4" />
              Raw Data
            </button>
          </div>

          {/* Tab Content */}
          <div className="flex-1 bg-code-well rounded-card p-8 min-h-[440px] border border-hairline">
            {activeTab === 'chart' ? (
              <div className="h-full w-full flex flex-col gap-6">
                <div className="flex items-center justify-end gap-6 text-label font-medium">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: 'var(--color-series-1)' }} />
                    <span className="text-fg-muted">Accuracy (Right Axis)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: 'var(--color-danger)' }} />
                    <span className="text-fg-muted">Loss (Left Axis)</span>
                  </div>
                </div>
                <div className="flex-1 min-h-[400px]">
                  <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                    <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="4 4" stroke="var(--color-hairline)" vertical={false} />
                      <XAxis dataKey="round" stroke="var(--color-hairline)" tick={{ fill: 'var(--color-fg-muted)', fontSize: 12 }} tickLine={false} axisLine={false} dy={10} />
                      <YAxis yAxisId="left" stroke="var(--color-hairline)" tick={{ fill: 'var(--color-fg-muted)', fontSize: 12 }} tickLine={false} axisLine={false} dx={-10} />
                      <YAxis yAxisId="right" orientation="right" stroke="var(--color-hairline)" tick={{ fill: 'var(--color-fg-muted)', fontSize: 12 }} tickLine={false} axisLine={false} dx={10} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                      <Tooltip
                        contentStyle={{ backgroundColor: 'var(--color-surface-2)', borderColor: 'var(--color-hairline)', borderRadius: '8px', color: 'var(--color-fg)' }}
                        itemStyle={{ fontWeight: 600, fontSize: '13px' }}
                        labelStyle={{ color: 'var(--color-fg-muted)', marginBottom: '8px', fontSize: '12px' }}
                        formatter={(value: number, name: string) => [name === 'accuracy' ? `${(value * 100).toFixed(2)}%` : value.toFixed(4), name.charAt(0).toUpperCase() + name.slice(1)]}
                        labelFormatter={(label) => `Round ${label}`}
                      />
                      <Line yAxisId="left" type="monotone" dataKey="loss" stroke="var(--color-danger)" strokeWidth={2.5} dot={false} activeDot={{ r: 5, strokeWidth: 0, fill: 'var(--color-danger)' }} />
                      <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="var(--color-series-1)" strokeWidth={2.5} dot={false} activeDot={{ r: 5, strokeWidth: 0, fill: 'var(--color-series-1)' }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <div className="h-[450px] overflow-auto rounded-card border border-hairline">
                <table className="w-full text-left border-collapse bg-surface-1">
                  <thead className="bg-surface-2 sticky top-0 z-10 border-b border-hairline">
                    <tr>
                      <th className="px-6 py-3 text-caption font-semibold uppercase tracking-wider text-fg-muted">Round</th>
                      <th className="px-6 py-3 text-caption font-semibold uppercase tracking-wider text-fg-muted">Loss</th>
                      <th className="px-6 py-3 text-caption font-semibold uppercase tracking-wider text-fg-muted">Accuracy</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-hairline text-label font-mono tabular-nums text-fg">
                    {results.slice().reverse().map((row) => (
                      <tr key={row.serverRound} className="hover:bg-surface-2 transition-colors">
                        <td className="px-6 py-3.5 text-fg-muted">#{row.serverRound}</td>
                        <td className="px-6 py-3.5 text-danger">{row.loss.toFixed(4)}</td>
                        <td className="px-6 py-3.5 text-success">{(row.accuracy * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
          </>}
        </div>
      </div>
    </div>
  );
}
