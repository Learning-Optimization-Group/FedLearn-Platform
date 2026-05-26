// =============================================================================
// FedLearn Frontend — Redesigned ResultsModal (Apple-inspired)
// =============================================================================
// Wired to real ProjectResult[] from apiServices.
//
// DESIGN NOTE: Like LogViewer this surface uses a slate/cyan analytics-console
// palette rather than the global Instrument tokens. The dark canvas + cyan
// metric accents are intentional — they read like a Bloomberg/Datadog panel,
// which is appropriate for dense numeric/chart content. Do not migrate to
// design tokens without an explicit redesign of the layout.

import { useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  YAxis,
  XAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { X, Trophy, Timer, TrendingDown, Table, LineChart as ChartIcon } from 'lucide-react';
import { cn } from '../../lib/utils';
import type { ProjectResult } from '../../services/apiServices';

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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 lg:p-8 bg-slate-950/60 backdrop-blur-sm font-sans">
      <div className="bg-slate-900 border border-slate-700 w-full max-w-6xl h-full max-h-[90vh] rounded-md shadow-2xl shadow-cyan-900/10 flex flex-col overflow-hidden text-slate-200">
        {/* Header */}
        <div className="flex items-center justify-between p-8 pb-6 border-b border-slate-800">
          <div>
            <h2 className="text-[24px] font-semibold tracking-tight text-slate-100">
              {projectName} — Results
            </h2>
            <p className="text-[14px] text-slate-400 mt-1 tracking-tight">
              {hasResults
                ? `${results.length} rounds completed.`
                : 'No training rounds recorded yet.'}
            </p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center text-slate-400 bg-slate-800 hover:bg-slate-700 rounded-sm transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-8 pb-8 flex flex-col gap-6 scroll-smooth">
          {!hasResults && (
            <div className="flex flex-col items-center justify-center flex-1 text-center gap-3 py-16">
              <div className="w-14 h-14 rounded-md bg-slate-800 flex items-center justify-center">
                <ChartIcon className="w-6 h-6 text-slate-400" />
              </div>
              <h3 className="text-[20px] font-semibold tracking-tight text-slate-200">
                No results yet
              </h3>
              <p className="text-[14px] text-slate-400 max-w-md tracking-tight">
                Results appear here once the project has completed at least one federated training
                round. Start the server and connect clients to produce data.
              </p>
              <button
                onClick={onClose}
                className="mt-4 bg-slate-800 hover:bg-slate-700 text-slate-200 px-6 py-2.5 rounded-md text-[14px] font-medium tracking-tight border border-slate-700 transition-colors"
              >
                Close
              </button>
            </div>
          )}

          {hasResults && (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-slate-800/40 rounded-md p-8 border border-slate-700/50">
                  <div className="flex items-center gap-2 text-green-500 mb-3">
                    <Trophy className="w-5 h-5" />
                    <span className="font-semibold tracking-wide uppercase text-[12px] text-slate-400">
                      Best Accuracy
                    </span>
                  </div>
                  <div className="text-[36px] font-semibold tracking-tighter text-slate-100">
                    {(bestAccuracy * 100).toFixed(2)}%
                  </div>
                </div>

                <div className="bg-slate-800/40 rounded-md p-8 border border-slate-700/50">
                  <div className="flex items-center gap-2 text-rose-500 mb-3">
                    <TrendingDown className="w-5 h-5" />
                    <span className="font-semibold tracking-wide uppercase text-[12px] text-slate-400">
                      Final Loss
                    </span>
                  </div>
                  <div className="text-[36px] font-semibold tracking-tighter text-slate-100">
                    {finalLoss.toFixed(4)}
                  </div>
                </div>

                <div className="bg-slate-800/40 rounded-md p-8 border border-slate-700/50">
                  <div className="flex items-center gap-2 text-cyan-500 mb-3">
                    <Timer className="w-5 h-5" />
                    <span className="font-semibold tracking-wide uppercase text-[12px] text-slate-400">
                      Total Rounds
                    </span>
                  </div>
                  <div className="text-[36px] font-semibold tracking-tighter text-slate-100">
                    {results.length}
                  </div>
                </div>
              </div>

              {/* Tabs */}
              <div className="flex items-center gap-3 border-b border-slate-800 pb-2">
                <button
                  onClick={() => setActiveTab('chart')}
                  className={cn(
                    'flex items-center gap-2 px-5 py-2.5 rounded-md text-[14px] font-medium tracking-tight transition-all duration-200',
                    activeTab === 'chart'
                      ? 'bg-slate-800 text-cyan-400 border border-slate-700'
                      : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-transparent'
                  )}
                >
                  <ChartIcon className="w-4 h-4" />
                  Performance Chart
                </button>
                <button
                  onClick={() => setActiveTab('table')}
                  className={cn(
                    'flex items-center gap-2 px-5 py-2.5 rounded-md text-[14px] font-medium tracking-tight transition-all duration-200',
                    activeTab === 'table'
                      ? 'bg-slate-800 text-cyan-400 border border-slate-700'
                      : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-transparent'
                  )}
                >
                  <Table className="w-4 h-4" />
                  Raw Data
                </button>
              </div>

              {/* Tab Content */}
              <div className="flex-1 bg-slate-950 rounded-md p-8 min-h-[440px] border border-slate-800 shadow-inner">
                {activeTab === 'chart' ? (
                  <div className="h-full w-full flex flex-col gap-6">
                    <div className="flex items-center justify-end gap-6 text-[13px] font-medium tracking-tight">
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-sm bg-green-500" />
                        <span className="text-slate-400">Accuracy (Right Axis)</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-sm bg-rose-500" />
                        <span className="text-slate-400">Loss (Left Axis)</span>
                      </div>
                    </div>
                    <div className="flex-1 min-h-[400px]">
                      <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                        <LineChart
                          data={chartData}
                          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                        >
                          <CartesianGrid strokeDasharray="4 4" stroke="#334155" vertical={false} />
                          <XAxis
                            dataKey="round"
                            stroke="#475569"
                            tick={{ fill: '#94a3b8', fontSize: 12 }}
                            tickLine={false}
                            axisLine={false}
                            dy={10}
                          />
                          <YAxis
                            yAxisId="left"
                            stroke="#475569"
                            tick={{ fill: '#94a3b8', fontSize: 12 }}
                            tickLine={false}
                            axisLine={false}
                            dx={-10}
                          />
                          <YAxis
                            yAxisId="right"
                            orientation="right"
                            stroke="#475569"
                            tick={{ fill: '#94a3b8', fontSize: 12 }}
                            tickLine={false}
                            axisLine={false}
                            dx={10}
                            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: '#0f172a',
                              borderColor: '#334155',
                              borderRadius: '6px',
                              color: '#f1f5f9',
                              boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)',
                            }}
                            itemStyle={{ fontWeight: 600, fontSize: '13px' }}
                            labelStyle={{ color: '#94a3b8', marginBottom: '8px', fontSize: '12px' }}
                            formatter={(value: number, name: string) => [
                              name === 'accuracy'
                                ? `${(value * 100).toFixed(2)}%`
                                : value.toFixed(4),
                              name.charAt(0).toUpperCase() + name.slice(1),
                            ]}
                            labelFormatter={(label) => `Round ${label}`}
                          />
                          <Line
                            yAxisId="left"
                            type="monotone"
                            dataKey="loss"
                            stroke="#f43f5e"
                            strokeWidth={2.5}
                            dot={false}
                            activeDot={{ r: 5, strokeWidth: 0, fill: '#f43f5e' }}
                          />
                          <Line
                            yAxisId="right"
                            type="monotone"
                            dataKey="accuracy"
                            stroke="#22c55e"
                            strokeWidth={2.5}
                            dot={false}
                            activeDot={{ r: 5, strokeWidth: 0, fill: '#22c55e' }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                ) : (
                  <div className="h-[450px] overflow-auto rounded-md border border-slate-800">
                    <table className="w-full text-left border-collapse bg-slate-900">
                      <thead className="bg-slate-800 sticky top-0 z-10 shadow-sm border-b border-slate-700">
                        <tr>
                          <th className="px-6 py-3 text-[12px] font-semibold uppercase tracking-wider text-slate-400">
                            Round
                          </th>
                          <th className="px-6 py-3 text-[12px] font-semibold uppercase tracking-wider text-slate-400">
                            Loss
                          </th>
                          <th className="px-6 py-3 text-[12px] font-semibold uppercase tracking-wider text-slate-400">
                            Accuracy
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800 text-[13px] font-mono tracking-tight text-slate-300">
                        {results
                          .slice()
                          .reverse()
                          .map((row) => (
                            <tr
                              key={row.serverRound}
                              className="hover:bg-slate-800/50 transition-colors"
                            >
                              <td className="px-6 py-3.5 text-slate-400">#{row.serverRound}</td>
                              <td className="px-6 py-3.5 text-rose-400">{row.loss.toFixed(4)}</td>
                              <td className="px-6 py-3.5 text-green-400">
                                {(row.accuracy * 100).toFixed(2)}%
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
