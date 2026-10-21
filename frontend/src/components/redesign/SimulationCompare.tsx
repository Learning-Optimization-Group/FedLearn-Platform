import { Card } from '../ui';
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { simulationTerminal, type SimulationRun } from '../../services/simulationService';
import { finalRound, finiteMetric, metric, statusLabel, sumRounds } from './simulationPresentation';

export function SimulationCompare({ runs }: { runs: SimulationRun[] }) {
  const first = runs[0]?.recipe;
  if (runs.length < 2 || runs.length > 4 || new Set(runs.map(r => r.id)).size !== runs.length || runs.some(r => !simulationTerminal(r.status)))
    return <p role="alert">Choose two to four distinct terminal runs.</p>;
  if (!first?.workloadId || !first.featureReportSha256 || !first.evaluationSplit || runs.some(r => r.recipe.workloadId !== first.workloadId || r.recipe.featureReportSha256 !== first.featureReportSha256 || r.recipe.evaluationSplit !== first.evaluationSplit))
    return <p role="alert">Comparisons require the same workload, feature pin and evaluation split.</p>;
  const measures: [string, (run: SimulationRun) => string][] = [
    ['State', r => `${statusLabel(r)}${r.status === 'COMPLETED' && r.qualification ? '' : ' (partial)'}`],
    ['Completed rounds', r => `${r.completedRounds ?? '—'} / ${r.requestedRounds ?? '—'}`],
    ['Final AUROC', r => metric(finalRound(r)?.auc)], ['Final loss', r => metric(finalRound(r)?.loss)],
    ['Final accuracy', r => metric(finalRound(r)?.accuracy)], ['Calibration ECE', r => metric(finalRound(r)?.ece)],
    ['Privacy', r => r.recipe.privacy ?? '—'], ['Epsilon target', r => r.recipe.privacy === 'central' ? metric(r.recipe.epsilon) : 'Not applied'],
    ['Modeled upload (MB)', r => { const n = sumRounds(r, x => x.comm.up_bytes); return n === null ? '—' : metric(n / 1e6); }],
    ['Modeled download (MB)', r => { const n = sumRounds(r, x => x.comm.down_bytes); return n === null ? '—' : metric(n / 1e6); }],
    ['Measured CPU time (s)', r => metric(sumRounds(r, x => x.cpuSeconds))],
    ['Modeled deployment time (s)', r => metric(sumRounds(r, x => x.simulatedRoundSeconds))],
  ];
  const keys = [...new Set(runs.flatMap(r => Object.keys(r.recipe)))].filter(key => new Set(runs.map(r => JSON.stringify(r.recipe[key as keyof typeof r.recipe]))).size > 1);
  const curve: ({ round: number } & Record<string, number | null>)[] = [...new Set(runs.flatMap(r => r.rounds.map(value => value.round)))].sort((a, b) => a - b).map(number => ({
    round: number, ...Object.fromEntries(runs.map(run => [run.id, finiteMetric(run.rounds.find(value => value.round === number)?.auc)])),
  }));
  return <Card padding="lg" className="space-y-4"><h2 className="text-h3 font-semibold">Exploratory comparison</h2>
    <p className="max-w-prose text-label text-fg-muted">Final observed rounds, including partial outcomes. These validation runs do not establish a statistical ranking. Different settings and missing measurements stay visible. Missing or nonfinite curve values appear as gaps.</p>
    <div className="overflow-x-auto"><table className="w-full text-left text-label tabular-nums"><caption className="sr-only">Matched-workload final endpoints and recipe differences</caption>
      <thead><tr><th className="p-2">Measure</th>{runs.map(r => <th className="p-2" key={r.id} title={r.id}>Run {r.id.slice(0, 8)}</th>)}</tr></thead>
      <tbody>{measures.map(([label, read]) => <tr className="border-t border-hairline" key={label}><th className="p-2 font-medium">{label}</th>{runs.map(r => <td className="p-2" key={r.id}>{read(r)}</td>)}</tr>)}</tbody>
    </table></div>
    <h3 className="text-label font-semibold">Validation AUROC across completed rounds</h3>
    <div className="h-52 w-full" aria-hidden="true"><ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
      <LineChart data={curve}><CartesianGrid stroke="var(--color-hairline)" /><XAxis dataKey="round" allowDecimals={false} /><YAxis domain={[0, 1]} /><Tooltip /><Legend />
        {runs.map((run, index) => <Line key={run.id} dataKey={run.id} name={`Run ${run.id.slice(0, 8)}`} stroke={`var(--color-series-${index + 1})`} strokeWidth={2} connectNulls={false} isAnimationActive={false} />)}
      </LineChart></ResponsiveContainer></div>
    <details><summary className="cursor-pointer text-label">Round-by-round values</summary><div className="mt-2 overflow-x-auto"><table className="w-full text-left text-caption tabular-nums">
      <caption>Observed AUROC by completed round</caption><thead><tr><th className="p-2">Round</th>{runs.map(run => <th key={run.id}>Run {run.id.slice(0, 8)}</th>)}</tr></thead>
      <tbody>{curve.map(row => <tr key={row.round} className="border-t border-hairline"><th className="p-2">{row.round}</th>{runs.map(run => <td key={run.id}>{metric(row[run.id])}</td>)}</tr>)}</tbody>
    </table></div></details>
    <h3 className="text-label font-semibold">Differing recipe fields</h3>
    {!keys.length ? <p className="text-label">The resolved recipe fields match.</p> : <div className="overflow-x-auto"><table className="w-full text-left text-caption"><caption className="sr-only">Resolved setting differences</caption><thead><tr><th className="p-2">Setting</th>{runs.map(r => <th key={r.id}>Run {r.id.slice(0, 8)}</th>)}</tr></thead>
      <tbody>{keys.map(key => <tr className="border-t border-hairline" key={key}><th className="p-2 font-medium">{key}</th>{runs.map(r => <td className="p-2" key={r.id}>{String(r.recipe[key as keyof typeof r.recipe] ?? '—')}</td>)}</tr>)}</tbody></table></div>}
  </Card>;
}
