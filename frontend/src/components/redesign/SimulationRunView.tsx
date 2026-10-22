import { useState } from 'react';
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Button, Card } from '../ui';
import { finalRound, finiteMetric, metric, statusLabel, sumRounds, evaluationScope } from './simulationPresentation';
import { simulationTerminal, type SimulationRecord, type SimulationRun } from '../../services/simulationService';

export function SimulationRunView({ run, record, onStop, onExport, onInspect, stopping, exporting, inspecting }: {
  run: SimulationRun; record: SimulationRecord | null; onStop: () => void; onExport: () => void;
  onInspect: () => void; stopping: boolean; exporting: boolean; inspecting: boolean;
}) {
  const [curve, setCurve] = useState<'auc' | 'loss' | 'ece'>('auc');
  const last = finalRound(run);
  const scope = evaluationScope(run.recipe);
  const curveValues = run.rounds.map(r => ({ round: r.round, auc: finiteMetric(r.auc), loss: finiteMetric(r.loss), ece: finiteMetric(r.ece) }));
  const peakRss = finiteMetric(record?.execution.peakRssBytes);
  const elapsed = run.startedAt ? (Date.parse(run.endedAt ?? new Date().toISOString()) - Date.parse(run.startedAt)) / 1000 : null;
  return <Card padding="lg" role="region" aria-label="Selected simulation" className="min-w-0 space-y-5">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0">
        <h2 className="text-h3 font-semibold">Selected simulation</h2>
        <p className="break-all text-caption text-fg-muted">Run {run.id}</p>
        <p role="status" className="mt-2 font-medium">{statusLabel(run)}</p>
        <p className="text-body">{run.completedRounds ?? '—'} / {run.requestedRounds ?? '—'} rounds completed</p>
      </div>
      {!simulationTerminal(run.status) && <Button variant="danger" onClick={onStop} disabled={stopping || run.status === 'STOPPING'}>
        {stopping ? 'Requesting stop…' : 'Stop simulation'}
      </Button>}
    </div>
    <p className="text-label text-fg-muted"><span>{run.status === 'COMPLETED' && run.qualification ? 'Execution qualified' : 'Not qualified as a complete run'}</span>.
      {' '}{scope.adjective} results from simulated patient groups; no clinical-use qualification.</p>
    {run.errorCode && <p role="alert" className="text-label text-danger">{run.errorCode}: {run.errorMessage ?? 'The run did not complete.'}</p>}
    <dl className="grid grid-cols-2 gap-x-4 gap-y-3 lg:grid-cols-3">
      {[['Final AUROC', metric(last?.auc)], ['Final accuracy', typeof last?.accuracy === 'number' ? `${metric(last.accuracy * 100, 1)}%` : '—'],
        ['Final loss', metric(last?.loss)], ['Sensitivity', metric(last?.sensitivity)], ['Specificity', metric(last?.specificity)],
        ['Calibration ECE', metric(last?.ece)], ['Accounted epsilon', metric(last?.privacy.epsilon_spent)],
        ['Process elapsed (API timestamps)', elapsed !== null && elapsed >= 0 ? `${metric(elapsed, 1)} s` : '—'],
        ['Measured CPU time', `${metric(sumRounds(run, r => r.cpuSeconds))} s`]].map(([label, value]) =>
        <div key={label}><dt className="text-caption text-fg-muted">{label}</dt><dd aria-label={label} className="text-body-lg font-semibold tabular-nums">{value}</dd></div>)}
    </dl>
    {run.rounds.length > 0 && <section aria-label="Observed round outcomes">
      <label className="text-label">Metric curve <select className="ml-2 rounded-md border border-line bg-surface-1 p-2" value={curve} onChange={e => setCurve(e.target.value as typeof curve)}>
        <option value="auc">{scope.adjective} AUROC</option><option value="loss">{scope.adjective} loss</option><option value="ece">Calibration ECE</option>
      </select></label>
      <p className="mt-2 text-caption text-fg-muted">Actual completed rounds; {curve === 'auc' ? 'AUROC' : curve === 'loss' ? 'loss' : 'ECE'} from the {scope.noun}. No projected points. Missing or nonfinite values appear as gaps.</p>
      <div className="mt-2 h-48 w-full" aria-hidden="true">
        <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
          <LineChart data={curveValues}><CartesianGrid stroke="var(--color-hairline)" /><XAxis dataKey="round" allowDecimals={false} /><YAxis domain={curve === 'loss' ? [0, 'auto'] : [0, 1]} /><Tooltip />
            <Line dataKey={curve} stroke="var(--color-series-1)" strokeWidth={2} isAnimationActive={false} connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="overflow-x-auto"><table className="w-full text-left text-caption tabular-nums">
        <caption className="sr-only">Observed completed-round metrics and modeled communication</caption>
        <thead><tr>{['Round', 'AUROC', 'Loss', 'ECE', 'Accepted clients', 'Aggregated clients', 'Modeled upload (MB)'].map(x => <th key={x} className="border-b border-hairline p-2">{x}</th>)}</tr></thead>
        <tbody>{run.rounds.map(r => <tr key={r.round}><td className="p-2">{r.round}</td><td>{metric(r.auc)}</td><td>{metric(r.loss)}</td><td>{metric(r.ece)}</td><td>{metric(r.participants.accepted, 0)}</td><td>{metric(r.participants.aggregated, 0)}</td><td>{finiteMetric(r.comm.up_bytes) === null ? '—' : metric((r.comm.up_bytes as number) / 1e6)}</td></tr>)}</tbody>
      </table></div>
    </section>}
    {!run.rounds.length && <p className="text-body text-fg-muted">No completed rounds have been reported.</p>}
    <details className="border-t border-hairline pt-3"><summary className="cursor-pointer text-label font-medium">Recipe and provenance</summary>
      <dl className="mt-3 grid gap-2 text-caption">
        <div><dt className="font-medium">Configuration identity</dt><dd className="break-all">{run.configHash}</dd></div>
        <div><dt className="font-medium">Feature report identity</dt><dd className="break-all">{run.recipe.featureReportSha256 ?? 'Unavailable'}</dd></div>
        {Object.entries(run.recipe).filter(([key]) => key !== 'featureReportSha256').map(([key, value]) => <div key={key}><dt className="inline font-medium">{key}: </dt><dd className="inline break-all">{value === null ? '—' : String(value)}</dd></div>)}
      </dl>
      <Button variant="secondary" size="sm" className="mt-3" onClick={onInspect} disabled={!run.recordAvailable || inspecting}>{inspecting ? 'Loading execution details…' : 'Load execution details'}</Button>
      {record && <dl className="mt-3 space-y-1 text-caption"><div><dt className="inline">Measured engine wall time: </dt><dd className="inline">{metric(record.execution.wallSeconds)} s</dd></div>
        <div><dt className="inline">Observed peak RSS: </dt><dd className="inline">{peakRss === null ? '—' : `${metric(peakRss / 1048576, 1)} MiB`} ({record.execution.peakRssScope ?? 'scope unavailable'})</dd></div>
        <div><dt className="inline">Source commit: </dt><dd className="inline break-all">{record.source.runtime?.git_commit ?? 'Unavailable'}</dd></div><p>{record.execution.scope}</p></dl>}
    </details>
    <div className="border-t border-hairline pt-3"><Button variant="secondary" onClick={onExport} disabled={!run.recordAvailable || exporting}>{exporting ? 'Downloading…' : 'Download research record'}</Button>
      <p className="mt-2 text-caption text-fg-muted">Diagnostic JSON with experiment seeds and outcomes. This is not a DP-safe release. Communication and deployment time are modeled; CPU and engine wall time are local observations.</p></div>
  </Card>;
}
