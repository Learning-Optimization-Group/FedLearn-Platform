// =============================================================================
// FedLearn Frontend — BenchmarkDashboard (role: PLATFORM_ADMIN)
// =============================================================================
// Benchmarking & observability surface. Two views:
//   • Overview  — platform health strip + aggregate stats + the runs table.
//   • Drilldown — one project's per-round time series (quality / loss / calibration
//     / efficiency) plus the latest round's per-class table and confusion matrix.
// Headline metric is task-type-aware: classification → accuracy↑, CAUSAL_LM →
// perplexity↓. All data comes from GET /api/admin/benchmarks/** (admin-gated).

import { useEffect, useState, useCallback } from 'react';
import {
    ResponsiveContainer, LineChart, Line, BarChart, Bar,
    XAxis, YAxis, Tooltip, CartesianGrid, Legend,
} from 'recharts';
import { AlertCircle, ArrowLeft, Activity, RefreshCw } from 'lucide-react';
import * as api from '../../services/apiServices';
import {
    errorMessage,
    type BenchmarkOverview,
    type BenchmarkRun,
    type ProjectBenchmark,
} from '../../services/apiServices';
import { Card, StatGroup, Button } from '../ui';
import { PageHeader } from './PageHeader';
import { createLogger } from '../../lib/logger';

const log = createLogger('BenchmarkDashboard');

// ─── Formatters ──────────────────────────────────────────────────────────
const DASH = '—';
const isGenerative = (taskType?: string | null) => (taskType ?? '').toUpperCase() === 'CAUSAL_LM';

function pct(v?: number | null, digits = 1): string {
    return v == null ? DASH : `${(v * 100).toFixed(digits)}%`;
}
function num(v?: number | null, digits = 3): string {
    return v == null ? DASH : v.toFixed(digits);
}
function ms(v?: number | null): string {
    if (v == null) return DASH;
    if (v < 1000) return `${Math.round(v)} ms`;
    return `${(v / 1000).toFixed(1)} s`;
}
function params(v?: number | null): string {
    if (v == null) return DASH;
    if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
    if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
    if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
    return `${v}`;
}
function mb(v?: number | null): string {
    return v == null ? DASH : `${v.toFixed(v < 1 ? 3 : 1)} MB`;
}
function when(iso?: string | null): string {
    if (!iso) return DASH;
    try {
        return new Date(iso).toLocaleString();
    } catch {
        return iso;
    }
}

// Headline for a run, keyed off task type.
function headline(run: BenchmarkRun): { label: string; value: string } {
    if (isGenerative(run.taskType)) {
        return { label: 'Perplexity ↓', value: num(run.finalPerplexity, 2) };
    }
    return { label: 'Accuracy ↑', value: pct(run.bestAccuracy ?? run.finalAccuracy) };
}

// ─── Chart styling ───────────────────────────────────────────────────────
// Categorical series colors come from the shared series tokens, assigned in
// fixed order (series-1 first) per chart — never status colors, never cycled.
const AXIS = { stroke: 'var(--color-fg-subtle)', fontSize: 11 };
const GRID = 'var(--color-hairline)';
const SERIES = [
    'var(--color-series-1)',
    'var(--color-series-2)',
    'var(--color-series-3)',
];

// Legend text wears text tokens, not the series color — the swatch carries identity.
const legendFormatter = (value: string) => (
    <span style={{ color: 'var(--color-fg-muted)' }}>{value}</span>
);

function ChartCard({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
    return (
        <Card padding="md" className="flex flex-col gap-2">
            <div className="flex flex-col">
                <span className="text-label font-semibold text-fg">{title}</span>
                {subtitle && <span className="text-caption text-fg-muted">{subtitle}</span>}
            </div>
            <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                    {children as React.ReactElement}
                </ResponsiveContainer>
            </div>
        </Card>
    );
}

const tooltipStyle = {
    contentStyle: {
        background: 'var(--color-surface-2)',
        border: '1px solid var(--color-hairline)',
        borderRadius: 8,
        fontSize: 12,
    },
    labelStyle: { color: 'var(--color-fg-muted)' },
};

// ─── Confusion-matrix heatmap ────────────────────────────────────────────
// Single-hue sequential tint: magnitude maps onto stepped opacities of the
// accent ink (never a multi-hue diag/off-diag split). Ink flips to accent-fg
// on the darkest steps to stay readable.
const TINT_STEPS = [0, 15, 30, 45, 62, 80];

function tintStep(cell: number, max: number): number {
    if (cell <= 0) return 0;
    const idx = Math.min(TINT_STEPS.length - 1, 1 + Math.floor((cell / max) * (TINT_STEPS.length - 2)));
    return TINT_STEPS[idx];
}

function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
    const max = Math.max(1, ...matrix.flat());
    return (
        <div className="overflow-x-auto">
            <table className="border-collapse text-caption">
                <thead>
                    <tr>
                        <th className="p-1.5 text-fg-subtle font-medium text-right">actual ╲ pred</th>
                        {labels.map((l) => (
                            <th key={l} className="p-1.5 text-fg-muted font-medium text-center max-w-[80px] truncate" title={l}>
                                {l}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {matrix.map((row, i) => (
                        <tr key={i}>
                            <td className="p-1.5 text-fg-muted font-medium text-right max-w-[90px] truncate" title={labels[i]}>
                                {labels[i] ?? i}
                            </td>
                            {row.map((cell, j) => {
                                const step = tintStep(cell, max);
                                return (
                                    <td
                                        key={j}
                                        className="p-1.5 text-center tabular-nums border border-hairline"
                                        style={{
                                            background: step === 0
                                                ? 'transparent'
                                                : `color-mix(in srgb, var(--color-accent) ${step}%, transparent)`,
                                            color: step >= 62 ? 'var(--color-accent-fg)' : 'var(--color-fg)',
                                        }}
                                        title={`actual ${labels[i] ?? i} → predicted ${labels[j] ?? j}: ${cell}`}
                                    >
                                        {cell}
                                    </td>
                                );
                            })}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}

const th = 'text-left text-caption font-semibold uppercase tracking-wider text-fg-muted px-3 py-2';
const td = 'px-3 py-2 text-body text-fg align-middle tabular-nums';

// ─── Drilldown ───────────────────────────────────────────────────────────
function ProjectDetail({ data, onBack }: { data: ProjectBenchmark; onBack: () => void }) {
    const generative = isGenerative(data.taskType);
    const summary = data.summary;
    const chart = data.rounds.map((r) => ({
        round: r.serverRound,
        accuracy: r.accuracy != null ? r.accuracy * 100 : null,
        balancedAccuracy: r.balancedAccuracy != null ? r.balancedAccuracy * 100 : null,
        f1Macro: r.f1Macro != null ? r.f1Macro * 100 : null,
        loss: r.loss,
        perplexity: r.perplexity,
        ece: r.ece,
        brier: r.brier,
        roundSec: r.roundDurationMs != null ? r.roundDurationMs / 1000 : null,
        clients: r.clientCount,
    }));

    return (
        <div className="flex flex-col gap-5">
            <div className="flex items-center gap-3">
                <Button variant="ghost" size="sm" onClick={onBack}>
                    <ArrowLeft className="h-4 w-4" strokeWidth={1.5} /> Back
                </Button>
                <h3 className="text-h4 font-semibold text-fg">
                    {summary?.projectName ?? 'Project'}
                </h3>
                <span className="chip">{summary?.modelType ?? 'model'}</span>
                {data.taskType && <span className="text-caption text-fg-muted">{data.taskType}</span>}
            </div>

            {/* Summary stats — one card, hairline-divided */}
            <StatGroup
                stats={[
                    {
                        label: generative ? 'Final perplexity' : 'Best accuracy',
                        value: generative ? num(summary?.finalPerplexity, 2) : pct(summary?.bestAccuracy),
                    },
                    { label: 'Macro-F1', value: pct(summary?.finalF1Macro) },
                    {
                        label: 'Rounds → target',
                        value: summary?.roundsToTarget != null ? `${summary.roundsToTarget}` : DASH,
                        sparkline:
                            summary?.targetAccuracy != null ? (
                                <span className="text-caption text-fg-subtle">target {pct(summary.targetAccuracy)}</span>
                            ) : undefined,
                    },
                    { label: 'Avg round', value: ms(summary?.avgRoundMs) },
                    { label: 'Model size', value: mb(summary?.modelSizeMb) },
                    { label: 'Params', value: params(summary?.paramCount) },
                ]}
            />

            {/* Charts */}
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                {!generative && (
                    <ChartCard title="Quality over rounds" subtitle="accuracy · balanced accuracy · macro-F1 (%)">
                        <LineChart data={chart} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
                            <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
                            <XAxis dataKey="round" {...AXIS} />
                            <YAxis domain={[0, 100]} {...AXIS} />
                            <Tooltip {...tooltipStyle} />
                            <Legend wrapperStyle={{ fontSize: 11 }} formatter={legendFormatter} />
                            <Line type="monotone" dataKey="accuracy" name="accuracy" stroke={SERIES[0]} dot={false} strokeWidth={2} connectNulls />
                            <Line type="monotone" dataKey="balancedAccuracy" name="balanced acc" stroke={SERIES[1]} dot={false} strokeWidth={1.5} connectNulls />
                            <Line type="monotone" dataKey="f1Macro" name="macro-F1" stroke={SERIES[2]} dot={false} strokeWidth={1.5} connectNulls />
                        </LineChart>
                    </ChartCard>
                )}

                <ChartCard title={generative ? 'Loss & perplexity' : 'Loss over rounds'} subtitle="convergence">
                    <LineChart data={chart} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
                        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
                        <XAxis dataKey="round" {...AXIS} />
                        <YAxis {...AXIS} />
                        <Tooltip {...tooltipStyle} />
                        {generative && <Legend wrapperStyle={{ fontSize: 11 }} formatter={legendFormatter} />}
                        <Line type="monotone" dataKey="loss" name="loss" stroke={SERIES[0]} dot={false} strokeWidth={2} connectNulls />
                        {generative && (
                            <Line type="monotone" dataKey="perplexity" name="perplexity" stroke={SERIES[1]} dot={false} strokeWidth={1.5} connectNulls />
                        )}
                    </LineChart>
                </ChartCard>

                {!generative && (
                    <ChartCard title="Calibration over rounds" subtitle="ECE · Brier (lower is better — fragile under FedAvg)">
                        <LineChart data={chart} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
                            <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
                            <XAxis dataKey="round" {...AXIS} />
                            <YAxis {...AXIS} />
                            <Tooltip {...tooltipStyle} />
                            <Legend wrapperStyle={{ fontSize: 11 }} formatter={legendFormatter} />
                            <Line type="monotone" dataKey="ece" name="ECE" stroke={SERIES[0]} dot={false} strokeWidth={1.5} connectNulls />
                            <Line type="monotone" dataKey="brier" name="Brier" stroke={SERIES[1]} dot={false} strokeWidth={1.5} connectNulls />
                        </LineChart>
                    </ChartCard>
                )}

                <ChartCard title="Efficiency over rounds" subtitle="round wall-clock (s)">
                    <BarChart data={chart} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
                        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
                        <XAxis dataKey="round" {...AXIS} />
                        <YAxis {...AXIS} />
                        <Tooltip {...tooltipStyle} />
                        <Bar dataKey="roundSec" name="round (s)" fill={SERIES[0]} radius={[3, 3, 0, 0]} />
                    </BarChart>
                </ChartCard>
            </div>

            {/* Per-class + confusion matrix (classification only) */}
            {!generative && data.latestPerClass && data.latestPerClass.length > 0 && (
                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                    <Card padding="md" className="flex flex-col gap-3">
                        <span className="text-label font-semibold text-fg">Per-class metrics (latest round)</span>
                        <div className="overflow-x-auto">
                            <table className="w-full border-collapse">
                                <thead>
                                    <tr>
                                        <th className={th}>Class</th>
                                        <th className={th}>Precision</th>
                                        <th className={th}>Recall</th>
                                        <th className={th}>F1</th>
                                        <th className={th}>Support</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {data.latestPerClass.map((c) => (
                                        <tr key={c.label} className="border-t border-hairline">
                                            <td className="px-3 py-2 text-body text-fg">{c.label}</td>
                                            <td className={td}>{pct(c.precision)}</td>
                                            <td className={td}>{pct(c.recall)}</td>
                                            <td className={td}>{pct(c.f1)}</td>
                                            <td className={td}>{c.support ?? DASH}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </Card>

                    {data.latestConfusionMatrix && data.classLabels && (
                        <Card padding="md" className="flex flex-col gap-3">
                            <span className="text-label font-semibold text-fg">Confusion matrix (latest round)</span>
                            <ConfusionMatrix matrix={data.latestConfusionMatrix} labels={data.classLabels} />
                        </Card>
                    )}
                </div>
            )}

            {data.rounds.length === 0 && (
                <Card padding="lg">
                    <p className="text-body text-fg-muted">No benchmark rounds recorded for this project yet.</p>
                </Card>
            )}
        </div>
    );
}

// ─── Overview ────────────────────────────────────────────────────────────
export function BenchmarkDashboard() {
    const [overview, setOverview] = useState<BenchmarkOverview | null>(null);
    const [detail, setDetail] = useState<ProjectBenchmark | null>(null);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(true);

    const loadOverview = useCallback(async () => {
        setLoading(true);
        try {
            const res = await api.fetchBenchmarkOverview();
            setOverview(res.data);
            setError('');
        } catch (e) {
            log.error('overview load failed', e);
            setError(errorMessage(e));
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadOverview();
    }, [loadOverview]);

    const openProject = useCallback(async (projectId: string) => {
        try {
            const res = await api.fetchProjectBenchmark(projectId);
            setDetail(res.data);
        } catch (e) {
            log.error('project benchmark load failed', e);
            setError(errorMessage(e));
        }
    }, []);

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
            <PageHeader
                title="Benchmarks & Observability"
                subtitle="Model quality, system efficiency, and federation health across every benchmarked run."
            >
                <Button variant="secondary" size="sm" onClick={loadOverview}>
                    <RefreshCw className="h-4 w-4" strokeWidth={1.5} /> Refresh
                </Button>
            </PageHeader>

            <div className="flex-1 overflow-y-auto">
                <div className="mx-auto w-full max-w-[1400px] px-6 py-6 md:px-10 flex flex-col gap-6">
                    {error && (
                        <div className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-4 py-3 text-body font-medium text-danger">
                            <AlertCircle className="h-4 w-4 flex-shrink-0" strokeWidth={1.5} />
                            {error}
                        </div>
                    )}

                    {detail ? (
                        <ProjectDetail data={detail} onBack={() => setDetail(null)} />
                    ) : (
                        <>
                            {/* Health strip */}
                            <StatGroup
                                stats={[
                                    {
                                        label: 'Best accuracy',
                                        value: pct(overview?.bestAccuracy),
                                        sparkline: overview?.bestAccuracyProject ? (
                                            <span className="text-caption text-fg-subtle truncate">{overview.bestAccuracyProject}</span>
                                        ) : undefined,
                                    },
                                    { label: 'Benchmarked projects', value: overview?.benchmarkedProjects ?? 0 },
                                    { label: 'Rounds recorded', value: overview?.totalRoundsRecorded ?? 0 },
                                    { label: 'Avg round time', value: ms(overview?.avgRoundDurationMs) },
                                ]}
                            />

                            {/* Secondary stats */}
                            <StatGroup
                                stats={[
                                    { label: 'Classification runs', value: overview?.classificationRuns ?? 0 },
                                    { label: 'Generative runs', value: overview?.generativeRuns ?? 0 },
                                    { label: 'Avg accuracy', value: pct(overview?.avgFinalAccuracy) },
                                    { label: 'Avg macro-F1', value: pct(overview?.avgFinalF1Macro) },
                                    { label: 'Avg model size', value: mb(overview?.avgModelSizeMb) },
                                ]}
                            />

                            {/* Runs table */}
                            <Card padding="none" className="overflow-hidden">
                                <div className="flex items-center gap-2 px-4 py-3 border-b border-hairline">
                                    <Activity className="h-4 w-4 text-accent" strokeWidth={1.5} />
                                    <span className="text-label font-semibold text-fg">Benchmarked runs</span>
                                </div>
                                <div className="overflow-x-auto">
                                    <table className="w-full border-collapse">
                                        <thead>
                                            <tr>
                                                <th className={th}>Project</th>
                                                <th className={th}>Model</th>
                                                <th className={th}>Task</th>
                                                <th className={th}>Rounds</th>
                                                <th className={th}>Headline</th>
                                                <th className={th}>Macro-F1</th>
                                                <th className={th}>TTA</th>
                                                <th className={th}>Avg round</th>
                                                <th className={th}>Params</th>
                                                <th className={th}>Updated</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {(overview?.runs ?? []).map((r) => {
                                                const h = headline(r);
                                                return (
                                                    <tr
                                                        key={r.projectId}
                                                        className="border-t border-hairline hover:bg-surface-2 cursor-pointer transition-colors"
                                                        onClick={() => openProject(r.projectId)}
                                                    >
                                                        <td className="px-3 py-2">
                                                            {/* Real button so the drilldown is keyboard-reachable. */}
                                                            <button
                                                                type="button"
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    openProject(r.projectId);
                                                                }}
                                                                className="text-body font-medium text-fg rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface-1"
                                                            >
                                                                {r.projectName ?? r.projectId.slice(0, 8)}
                                                            </button>
                                                        </td>
                                                        <td className="px-3 py-2 text-body text-fg-muted">{r.modelType ?? DASH}</td>
                                                        <td className="px-3 py-2 text-caption text-fg-subtle">{isGenerative(r.taskType) ? 'CAUSAL_LM' : 'classification'}</td>
                                                        <td className={td}>{r.roundsCompleted ?? DASH}</td>
                                                        <td className={td}><span className="text-fg">{h.value}</span> <span className="text-caption text-fg-subtle">{h.label}</span></td>
                                                        <td className={td}>{pct(r.finalF1Macro)}</td>
                                                        <td className={td}>{r.roundsToTarget != null ? `${r.roundsToTarget}r` : DASH}</td>
                                                        <td className={td}>{ms(r.avgRoundMs)}</td>
                                                        <td className={td}>{params(r.paramCount)}</td>
                                                        <td className="px-3 py-2 text-caption text-fg-subtle">{when(r.lastRecordedAt)}</td>
                                                    </tr>
                                                );
                                            })}
                                            {!loading && (overview?.runs ?? []).length === 0 && (
                                                <tr>
                                                    <td colSpan={10} className="px-4 py-8 text-center text-body text-fg-muted">
                                                        No benchmarked runs yet. Start a federated run — per-round metrics are captured automatically.
                                                    </td>
                                                </tr>
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </Card>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}

export default BenchmarkDashboard;
