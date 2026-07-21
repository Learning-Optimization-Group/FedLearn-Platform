// =============================================================================
// FedLearn Frontend — Redesigned ResultsModal
// =============================================================================
// Wired to real ProjectResult[] from apiServices.

import { useState, useRef, useEffect } from 'react';
import { ResponsiveContainer, LineChart, Line, YAxis, XAxis, Tooltip, CartesianGrid } from 'recharts';
import { X, Table, LineChart as ChartIcon } from 'lucide-react';
import { cn } from '../../lib/utils';
import type { ProjectResult } from '../../services/apiServices';
import { SectionLabel, StatGroup } from '../ui';
import { useFocusTrap } from '../../hooks/useFocusTrap';

interface ResultsModalProps {
  isOpen: boolean;
  onClose: () => void;
  projectName: string;
  results: ProjectResult[];
}

// Shared recharts tooltip styling — recharts requires inline styles, so these
// route through the token CSS variables rather than raw values.
const TOOLTIP_CONTENT_STYLE = {
  backgroundColor: 'var(--color-surface-1)',
  borderColor: 'var(--color-hairline)',
  borderRadius: 'var(--radius-md)',
  color: 'var(--color-fg)',
};
const TOOLTIP_ITEM_STYLE = { fontWeight: 600, fontSize: '13px' };
const TOOLTIP_LABEL_STYLE = { color: 'var(--color-fg-muted)', marginBottom: '8px', fontSize: '12px' };
const AXIS_TICK = { fill: 'var(--color-fg-muted)', fontSize: 12 };

export function ResultsModalV2({ isOpen, onClose, projectName, results }: ResultsModalProps) {
  const [activeTab, setActiveTab] = useState<'chart' | 'table'>('chart');

  // FE-13: this is a standalone dialog (not the shared Modal primitive, since it needs a
  // full-bleed 6xl/90vh layout). Give it the same a11y contract — dialog role, focus trap +
  // restore via useFocusTrap, and Escape-to-close.
  const panelRef = useRef<HTMLDivElement>(null);
  useFocusTrap(isOpen, panelRef);

  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const hasResults = results.length > 0;
  const chartData = results.map((r) => ({
    round: r.serverRound,
    loss: r.loss,
    accuracy: r.accuracy,
  }));

  const bestAccuracy = hasResults ? Math.max(...results.map((r) => r.accuracy)) : 0;
  const finalLoss = hasResults ? results[results.length - 1].loss : 0;

  const tabClass = (active: boolean) =>
    cn(
      'flex items-center gap-2 px-5 h-9 rounded-md text-body font-medium transition-colors duration-[160ms]',
      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface-1',
      active
        ? 'bg-surface-2 text-accent border border-hairline'
        : 'text-fg-muted hover:bg-surface-2 hover:text-fg border border-transparent',
    );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 lg:p-8 bg-scrim font-sans"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={`${projectName} — Results`}
        tabIndex={-1}
        className="bg-surface-1 border border-hairline w-full max-w-6xl h-full max-h-[90vh] rounded-card flex flex-col overflow-hidden text-fg shadow-overlay"
      >

        {/* Header */}
        <div className="flex items-start justify-between gap-4 px-6 pt-5 pb-4 border-b border-hairline">
          <div>
            <h2 className="text-h4 text-fg">{projectName} — Results</h2>
            <p className="text-body text-fg-muted mt-1">
              {hasResults ? `${results.length} training rounds done.` : 'No training rounds recorded yet.'}
            </p>
          </div>
          <button
            type="button"
            aria-label="Close"
            onClick={onClose}
            className={cn(
              'flex h-8 w-8 shrink-0 items-center justify-center rounded-md -mr-2',
              'text-fg-muted hover:text-fg hover:bg-surface-2 transition-colors duration-[120ms]',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent',
            )}
          >
            <X strokeWidth={1.5} className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-6 flex flex-col gap-6 scroll-smooth">

          {!hasResults && (
            <div className="flex flex-col items-center justify-center flex-1 text-center gap-3 py-16">
              <div className="w-14 h-14 rounded-pill bg-surface-2 flex items-center justify-center">
                <ChartIcon strokeWidth={1.5} className="w-6 h-6 text-fg-muted" />
              </div>
              <h3 className="text-h4 text-fg">No results yet</h3>
              <p className="text-body text-fg-muted max-w-md">
                Results appear once the project completes its first federated training round.
              </p>
            </div>
          )}

          {hasResults && <>

          {/* Summary stats — one card, internal dividers */}
          <StatGroup
            stats={[
              { label: 'Best accuracy', value: `${(bestAccuracy * 100).toFixed(2)}%` },
              { label: 'Final loss', value: finalLoss.toFixed(4) },
              { label: 'Rounds done', value: results.length },
            ]}
          />

          {/* Tabs */}
          <div className="flex items-center gap-3 border-b border-hairline pb-2">
            <button
              type="button"
              aria-pressed={activeTab === 'chart'}
              onClick={() => setActiveTab('chart')}
              className={tabClass(activeTab === 'chart')}
            >
              <ChartIcon strokeWidth={1.5} className="w-4 h-4" />
              Performance Chart
            </button>
            <button
              type="button"
              aria-pressed={activeTab === 'table'}
              onClick={() => setActiveTab('table')}
              className={tabClass(activeTab === 'table')}
            >
              <Table strokeWidth={1.5} className="w-4 h-4" />
              Raw Data
            </button>
          </div>

          {/* Tab Content */}
          {activeTab === 'chart' ? (
            // Two stacked single-axis charts over the same round domain —
            // single series each, so no legend; tooltips carry the values.
            <div className="flex flex-col gap-6">
              <div>
                <SectionLabel className="mb-3">Accuracy</SectionLabel>
                <div className="h-52 w-full">
                  <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                    <LineChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="4 4" stroke="var(--color-hairline)" vertical={false} />
                      <XAxis dataKey="round" stroke="var(--color-hairline)" tick={AXIS_TICK} tickLine={false} axisLine={false} dy={8} />
                      <YAxis stroke="var(--color-hairline)" tick={AXIS_TICK} tickLine={false} axisLine={false} width={44} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                      <Tooltip
                        contentStyle={TOOLTIP_CONTENT_STYLE}
                        itemStyle={TOOLTIP_ITEM_STYLE}
                        labelStyle={TOOLTIP_LABEL_STYLE}
                        formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Accuracy']}
                        labelFormatter={(label) => `Round ${label}`}
                      />
                      <Line type="monotone" dataKey="accuracy" stroke="var(--color-series-1)" strokeWidth={2.5} dot={false} activeDot={{ r: 4, strokeWidth: 0, fill: 'var(--color-series-1)' }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <SectionLabel className="mb-3">Loss</SectionLabel>
                <div className="h-52 w-full">
                  <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                    <LineChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="4 4" stroke="var(--color-hairline)" vertical={false} />
                      <XAxis dataKey="round" stroke="var(--color-hairline)" tick={AXIS_TICK} tickLine={false} axisLine={false} dy={8} />
                      <YAxis stroke="var(--color-hairline)" tick={AXIS_TICK} tickLine={false} axisLine={false} width={44} />
                      <Tooltip
                        contentStyle={TOOLTIP_CONTENT_STYLE}
                        itemStyle={TOOLTIP_ITEM_STYLE}
                        labelStyle={TOOLTIP_LABEL_STYLE}
                        formatter={(value: number) => [value.toFixed(4), 'Loss']}
                        labelFormatter={(label) => `Round ${label}`}
                      />
                      <Line type="monotone" dataKey="loss" stroke="var(--color-series-1)" strokeWidth={2.5} dot={false} activeDot={{ r: 4, strokeWidth: 0, fill: 'var(--color-series-1)' }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-h-[480px] overflow-auto rounded-lg border border-hairline">
              <table className="w-full text-left border-collapse bg-surface-1">
                <thead className="bg-surface-2 sticky top-0 z-10">
                  <tr>
                    <th className="px-6 py-3"><SectionLabel>Round</SectionLabel></th>
                    <th className="px-6 py-3"><SectionLabel>Loss</SectionLabel></th>
                    <th className="px-6 py-3"><SectionLabel>Accuracy</SectionLabel></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-hairline text-label font-mono tabular-nums text-fg">
                  {results.slice().reverse().map((row) => (
                    <tr key={row.serverRound} className="hover:bg-surface-2 transition-colors">
                      <td className="px-6 py-3 text-fg-muted">#{row.serverRound}</td>
                      <td className="px-6 py-3">{row.loss.toFixed(4)}</td>
                      <td className="px-6 py-3">{(row.accuracy * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          </>}
        </div>
      </div>
    </div>
  );
}
