// =============================================================================
// FedLearn Frontend — Redesigned ResultsModal (Apple-inspired)
// =============================================================================
// Wired to real ProjectResult[] from apiServices.

import { useState } from 'react';
import { ResponsiveContainer, LineChart, Line, YAxis, XAxis, Tooltip, CartesianGrid } from 'recharts';
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

  if (!isOpen || results.length === 0) return null;

  const chartData = results.map((r) => ({
    round: r.serverRound,
    loss: r.loss,
    accuracy: r.accuracy,
  }));

  const bestAccuracy = Math.max(...results.map((r) => r.accuracy));
  const finalLoss = results[results.length - 1].loss;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 lg:p-8 bg-black/40 backdrop-blur-xl font-sans">
      <div className="bg-[rgba(28,28,30,0.85)] border border-[rgba(255,255,255,0.1)] w-full max-w-6xl h-full max-h-[90vh] rounded-[40px] shadow-[0_20px_50px_rgba(0,0,0,0.5)] flex flex-col overflow-hidden text-[#f5f5f7]">

        {/* Header */}
        <div className="flex items-center justify-between p-8 pb-6">
          <div>
            <h2 className="text-[28px] font-semibold tracking-tight">{projectName} — Results</h2>
            <p className="text-[15px] text-[#86868b] mt-1 tracking-tight">
              {results.length} rounds completed.
            </p>
          </div>
          <button onClick={onClose} className="w-8 h-8 flex items-center justify-center text-[#86868b] bg-[#3a3a3c] hover:bg-[rgba(255,255,255,0.3)] rounded-full transition-colors">
            <X className="w-[18px] h-[18px]" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-8 pb-8 flex flex-col gap-6 scroll-smooth">

          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-[rgba(0,0,0,0.3)] rounded-[32px] p-8 border border-[rgba(255,255,255,0.05)]">
              <div className="flex items-center gap-2 text-[#32d74b] mb-3">
                <Trophy className="w-5 h-5" />
                <span className="font-semibold tracking-wide uppercase text-[12px]">Best Accuracy</span>
              </div>
              <div className="text-[44px] font-semibold tracking-tighter text-[#f5f5f7]">
                {(bestAccuracy * 100).toFixed(2)}%
              </div>
            </div>

            <div className="bg-[rgba(0,0,0,0.3)] rounded-[32px] p-8 border border-[rgba(255,255,255,0.05)]">
              <div className="flex items-center gap-2 text-[#bf5af2] mb-3">
                <TrendingDown className="w-5 h-5" />
                <span className="font-semibold tracking-wide uppercase text-[12px]">Final Loss</span>
              </div>
              <div className="text-[44px] font-semibold tracking-tighter text-[#f5f5f7]">
                {finalLoss.toFixed(4)}
              </div>
            </div>

            <div className="bg-[rgba(0,0,0,0.3)] rounded-[32px] p-8 border border-[rgba(255,255,255,0.05)]">
              <div className="flex items-center gap-2 text-[#0a84ff] mb-3">
                <Timer className="w-5 h-5" />
                <span className="font-semibold tracking-wide uppercase text-[12px]">Total Rounds</span>
              </div>
              <div className="text-[44px] font-semibold tracking-tighter text-[#f5f5f7]">
                {results.length}
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => setActiveTab('chart')}
              className={cn(
                "flex items-center gap-2 px-5 py-2.5 rounded-full text-[14px] font-medium tracking-tight transition-all duration-200",
                activeTab === 'chart' ? "bg-[#f5f5f7] text-black shadow-sm" : "bg-[rgba(255,255,255,0.05)] text-[#f5f5f7] hover:bg-[rgba(255,255,255,0.1)]"
              )}
            >
              <ChartIcon className="w-4 h-4" />
              Performance Chart
            </button>
            <button
              onClick={() => setActiveTab('table')}
              className={cn(
                "flex items-center gap-2 px-5 py-2.5 rounded-full text-[14px] font-medium tracking-tight transition-all duration-200",
                activeTab === 'table' ? "bg-[#f5f5f7] text-black shadow-sm" : "bg-[rgba(255,255,255,0.05)] text-[#f5f5f7] hover:bg-[rgba(255,255,255,0.1)]"
              )}
            >
              <Table className="w-4 h-4" />
              Raw Data
            </button>
          </div>

          {/* Tab Content */}
          <div className="flex-1 bg-black rounded-[32px] p-8 min-h-[440px] border border-[rgba(255,255,255,0.05)]">
            {activeTab === 'chart' ? (
              <div className="h-full w-full flex flex-col gap-6">
                <div className="flex items-center justify-end gap-6 text-[13px] font-medium tracking-tight">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#32d74b]" />
                    <span className="text-[#86868b]">Accuracy (Right Axis)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#bf5af2]" />
                    <span className="text-[#86868b]">Loss (Left Axis)</span>
                  </div>
                </div>
                <div className="flex-1 min-h-[400px]">
                  <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                    <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis dataKey="round" stroke="rgba(255,255,255,0.2)" tick={{ fill: '#86868b', fontSize: 13 }} tickLine={false} axisLine={false} dy={10} />
                      <YAxis yAxisId="left" stroke="rgba(255,255,255,0.2)" tick={{ fill: '#86868b', fontSize: 13 }} tickLine={false} axisLine={false} dx={-10} />
                      <YAxis yAxisId="right" orientation="right" stroke="rgba(255,255,255,0.2)" tick={{ fill: '#86868b', fontSize: 13 }} tickLine={false} axisLine={false} dx={10} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                      <Tooltip
                        contentStyle={{ backgroundColor: 'rgba(28,28,30,0.85)', backdropFilter: 'blur(20px)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '16px', color: '#f5f5f7', boxShadow: '0 20px 40px rgba(0,0,0,0.4)' }}
                        itemStyle={{ fontWeight: 600, fontSize: '15px' }}
                        labelStyle={{ color: '#86868b', marginBottom: '8px', fontSize: '13px' }}
                        formatter={(value: number, name: string) => [name === 'accuracy' ? `${(value * 100).toFixed(2)}%` : value.toFixed(4), name.charAt(0).toUpperCase() + name.slice(1)]}
                        labelFormatter={(label) => `Round ${label}`}
                      />
                      <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#bf5af2" strokeWidth={3} dot={false} activeDot={{ r: 6, strokeWidth: 0, fill: '#bf5af2' }} />
                      <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#32d74b" strokeWidth={3} dot={false} activeDot={{ r: 6, strokeWidth: 0, fill: '#32d74b' }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <div className="h-[450px] overflow-auto rounded-[20px]">
                <table className="w-full text-left border-collapse">
                  <thead className="bg-[rgba(28,28,30,0.85)] sticky top-0 backdrop-blur-xl z-10">
                    <tr>
                      <th className="px-6 py-4 text-[12px] font-semibold uppercase tracking-wider text-[#86868b]">Round</th>
                      <th className="px-6 py-4 text-[12px] font-semibold uppercase tracking-wider text-[#86868b]">Loss</th>
                      <th className="px-6 py-4 text-[12px] font-semibold uppercase tracking-wider text-[#86868b]">Accuracy</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[rgba(255,255,255,0.05)] text-[14px] font-mono tracking-tight text-[#f5f5f7]">
                    {results.slice().reverse().map((row) => (
                      <tr key={row.serverRound} className="hover:bg-[rgba(255,255,255,0.02)] transition-colors">
                        <td className="px-6 py-4 text-[#86868b]">#{row.serverRound}</td>
                        <td className="px-6 py-4 text-[#bf5af2]">{row.loss.toFixed(4)}</td>
                        <td className="px-6 py-4 text-[#32d74b]">{(row.accuracy * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
