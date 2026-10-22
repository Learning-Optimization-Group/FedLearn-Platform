import { useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { errorMessage, errorStatus } from '../../services/apiServices';
import * as api from '../../services/simulationService';
import type { SimulationChoices, SimulationPage, SimulationRecord, SimulationResolution, SimulationRun, SimulationWorkload } from '../../services/simulationService';
import { Button, Card } from '../ui';
import { PageHeader } from './PageHeader';
import { SimulationRunView } from './SimulationRunView';
import { statusLabel } from './simulationPresentation';
import { SimulationCompare } from './SimulationCompare';

const HISTORICAL = 'nih-qualified';
const WORKLOAD_NAMES: Record<string, string> = { [HISTORICAL]: 'Any finding · historical study', 'nih-effusion-r0': 'Effusion · capstone follow-up' };
/** Measured presets for the capstone Effusion R0 bundle; each fixes its own learning rate and clip. */
const EFFUSION_RECIPES: [string, string][] = [['effusion-clean', 'Measured clean · LR 0.1, clip 1.0'], ['effusion-dp', 'Measured DP · LR 0.03, clip 0.05']];
const INITIAL: SimulationChoices = { workloadId: HISTORICAL, recipe: 'standard', head: 'linear', numClients: 10,
  rounds: 3, localEpochs: 1, aggregation: 'mean', clipNorm: .05, privacy: 'none', epsilon: 4, behavior: 'clean', seed: 42 };
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const controlClass = 'mt-1 w-full rounded-md border border-line bg-surface-1 px-3 py-2 text-body focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent';

export function SimulationLab() {
  const { currentUser } = useAuth();
  if (!currentUser || !['PROJECT_OWNER', 'PLATFORM_ADMIN'].includes(currentUser.role))
    return <main className="p-6">A project-owner or administrator session is required.</main>;
  return <LabWorkspace key={JSON.stringify([currentUser.username, currentUser.email, currentUser.role])} />;
}

function LabWorkspace() {
  const [query, setQuery] = useSearchParams();
  const selectedId = query.get('run');
  const selectedRef = useRef(selectedId);
  selectedRef.current = selectedId;
  const [workloads, setWorkloads] = useState<SimulationWorkload[]>([]);
  const [history, setHistory] = useState<SimulationPage | null>(null);
  const [page, setPage] = useState(0);
  const [refresh, setRefresh] = useState(0);
  const [pollRetry, setPollRetry] = useState(0);
  const [loadError, setLoadError] = useState('');
  const [runError, setRunError] = useState('');
  const [actionError, setActionError] = useState('');
  const [choices, setChoices] = useState(INITIAL);
  const workload = workloads.find(w => w.id === choices.workloadId) ?? null;
  const historical = choices.workloadId === HISTORICAL;
  const [resolution, setResolution] = useState<SimulationResolution | null>(null);
  const [reviewing, setReviewing] = useState(false);
  const [starting, setStarting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [run, setRun] = useState<SimulationRun | null>(null);
  const [record, setRecord] = useState<SimulationRecord | null>(null);
  const [stoppingId, setStoppingId] = useState<string | null>(null);
  const [exportingId, setExportingId] = useState<string | null>(null);
  const [inspectingId, setInspectingId] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [comparison, setComparison] = useState<SimulationRun[]>([]);
  const [comparing, setComparing] = useState(false);
  const pending = useRef(new Set<AbortController>());
  function controller() { const value = new AbortController(); pending.current.add(value); return value; }
  useEffect(() => {
    const requests = pending.current;
    return () => { requests.forEach(request => request.abort()); requests.clear(); };
  }, []);

  useEffect(() => {
    const request = new AbortController();
    setLoadError('');
    Promise.all([api.getSimulationWorkloads(request.signal), api.listSimulations(page, request.signal)])
      .then(([data, recent]) => {
        if (!request.signal.aborted) { setWorkloads(data.workloads); setHistory(recent); }
      }).catch(error => { if (!request.signal.aborted) setLoadError(errorMessage(error, 'Could not load the workload and owned history.')); });
    return () => request.abort();
  }, [page, refresh]);

  useEffect(() => {
    const request = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    setRun(previous => previous?.id === selectedId ? previous : null);
    setRecord(previous => previous?.runId === selectedId ? previous : null);
    setRunError(''); setActionError('');
    if (!selectedId) return () => request.abort();
    if (!UUID.test(selectedId)) { setRunError('This simulation identifier is invalid. Choose a run from your history.'); return () => request.abort(); }
    async function poll() {
      try {
        const latest = await api.getSimulation(selectedId!, request.signal);
        if (request.signal.aborted) return;
        if (latest.id !== selectedId) throw new Error('Simulation identity mismatch');
        setRun(latest);
        setHistory(previous => previous ? { ...previous, content: previous.content.map(r => r.id === latest.id ? latest : r) } : previous);
        if (!api.simulationTerminal(latest.status)) timer = setTimeout(() => { void poll(); }, 1500);
      } catch (error) {
        if (!request.signal.aborted) { setRun(null); setRecord(null); setRunError(errorMessage(error, 'Could not refresh this run. Retry to obtain its current state.')); }
      }
    }
    void poll();
    return () => { request.abort(); clearTimeout(timer); };
  }, [selectedId, pollRetry]);

  function open(id: string) { setQuery(previous => { const next = new URLSearchParams(previous); next.set('run', id); return next; }); }
  function change(key: keyof SimulationChoices, value: string | number) { setChoices(previous => ({ ...previous, [key]: value })); setResolution(null); setActionError(''); }
  function chooseWorkload(id: string) {
    setChoices(previous => ({ ...previous, workloadId: id, recipe: id === HISTORICAL ? 'standard' : EFFUSION_RECIPES[0][0], clipNorm: INITIAL.clipNorm }));
    setResolution(null); setActionError('');
  }
  // A measured recipe fixes its clip, so the request omits clipNorm rather than send a value the contract would refuse.
  function payload(): Partial<SimulationChoices> {
    if (historical) return choices;
    const { clipNorm: _fixedByRecipe, ...rest } = choices;
    return rest;
  }
  function preset(extra: Partial<SimulationChoices>) {
    setChoices({ ...INITIAL, workloadId: choices.workloadId, recipe: historical ? 'standard' : choices.recipe, ...extra }); setResolution(null);
  }
  const valid = [choices.rounds, choices.localEpochs, choices.seed].every(Number.isInteger)
    && choices.rounds >= 1 && choices.rounds <= 20 && choices.localEpochs >= 1 && choices.localEpochs <= 7 && choices.seed >= 0 && choices.seed <= 1e9;
  const knownActive = history?.content.find(r => !api.simulationTerminal(r.status));
  const occupied = busy || !!knownActive || !!(run && !api.simulationTerminal(run.status));

  async function review() {
    if (!valid || !workload?.ready || reviewing || starting) return;
    const request = controller(); setReviewing(true); setResolution(null); setActionError('');
    try { const next = await api.resolveSimulation(payload(), request.signal); if (!request.signal.aborted) setResolution(next); }
    catch (error) { if (!request.signal.aborted) setActionError(errorMessage(error, 'Could not resolve this recipe.')); }
    finally { if (!request.signal.aborted) setReviewing(false); pending.current.delete(request); }
  }
  async function start() {
    if (!resolution || starting || occupied || !workload?.ready) return;
    const request = controller(); setStarting(true); setActionError('');
    try {
      const accepted = await api.startSimulation(payload(), request.signal);
      if (!request.signal.aborted) { open(accepted.id); setRefresh(v => v + 1); }
    } catch (error) {
      if (!request.signal.aborted) { if (errorStatus(error) === 409) setBusy(true); setActionError(errorMessage(error, 'The simulation could not be started. Check owned history before retrying.')); }
    } finally { if (!request.signal.aborted) setStarting(false); pending.current.delete(request); }
  }
  async function stop() {
    if (!run || stoppingId === run.id) return;
    const id = run.id, request = controller(); setStoppingId(id); setActionError('');
    try {
      const next = await api.stopSimulation(id, request.signal);
      if (!request.signal.aborted && selectedRef.current === id) { setRun(next); setPollRetry(n => n + 1); }
    } catch (error) { if (!request.signal.aborted && selectedRef.current === id) setActionError(errorMessage(error, 'Could not request a stop. The server may still be running.')); }
    finally { if (!request.signal.aborted) setStoppingId(null); pending.current.delete(request); }
  }
  async function inspect() {
    if (!run) return;
    const id = run.id, request = controller(); setInspectingId(id); setActionError('');
    try {
      const next = await api.getSimulationRecord(id, request.signal);
      if (!request.signal.aborted && selectedRef.current === id) {
        if (next.runId !== id || next.configHash !== run.configHash) throw new Error('Record identity mismatch');
        setRecord(next);
      }
    } catch (error) { if (!request.signal.aborted && selectedRef.current === id) setActionError(errorMessage(error, 'Could not load execution details.')); }
    finally { if (!request.signal.aborted) setInspectingId(null); pending.current.delete(request); }
  }
  async function download() {
    if (!run || exportingId === run.id) return;
    const id = run.id, request = controller(); setExportingId(id); setActionError('');
    try {
      const blob = await api.downloadSimulationRecord(id, request.signal);
      if (!request.signal.aborted && selectedRef.current === id) {
        const url = URL.createObjectURL(blob), link = document.createElement('a');
        link.href = url; link.download = `simulation-${id}.json`; document.body.append(link); link.click(); link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    } catch (error) { if (!request.signal.aborted && selectedRef.current === id) setActionError(errorMessage(error, 'Could not download the owned research record.')); }
    finally { if (!request.signal.aborted) setExportingId(null); pending.current.delete(request); }
  }
  async function compare() {
    const request = controller(); setComparing(true); setActionError(''); setComparison([]);
    try { const values = await Promise.all(compareIds.map(id => api.getSimulation(id, request.signal))); if (!request.signal.aborted) setComparison(values); }
    catch (error) { if (!request.signal.aborted) setActionError(errorMessage(error, 'Could not open all selected owned runs.')); }
    finally { if (!request.signal.aborted) setComparing(false); pending.current.delete(request); }
  }
  const selectors: [keyof SimulationChoices, string, [string | number, string][]][] = [
    ['head', 'Trainable head', [['linear', 'Linear · 1,026 parameters'], ['mlp8', 'MLP 8 · 4,122 parameters'], ['mlp128', 'MLP 128 · 65,922 parameters'], ['mlp512', 'MLP 512 · 263,682 parameters']]],
    ['numClients', 'Simulated clients', [10, 80, 1000].map(n => [n, String(n)])],
    ['aggregation', 'Aggregation', [['mean', 'Uniform mean'], ['trimmed_mean', 'Coordinate trimming · 20%']]],
    ...(historical ? [['clipNorm', 'Client clipping norm', [[.05, '0.05'], [.4, '0.4']]] as [keyof SimulationChoices, string, [string | number, string][]]]
      : [['recipe', 'Measured recipe', EFFUSION_RECIPES] as [keyof SimulationChoices, string, [string | number, string][]]]),
    ['privacy', 'Privacy', [['none', 'No DP'], ['central', 'Central DP · client update']]],
    ['epsilon', 'Epsilon target', [1, 4, 8].map(n => [n, String(n)])],
    ['behavior', 'Client behavior', [['clean', 'Clean'], ['sign_flip', '20% sign flip · scale −4'], ['label_flip', '20% binary label flip']]],
  ];
  return <main className="flex min-w-0 flex-1 flex-col overflow-y-auto">
    <PageHeader title="Simulation Lab" subtitle="Exploratory validation runs" />
    <div className="space-y-6 p-4 md:p-8">
      <section className="max-w-5xl space-y-2" aria-label="Prepared workload">
        <h2 className="text-h3 font-semibold">{workload?.label ?? 'NIH chest X-ray workload'}</h2>
        <p className="text-body text-fg-muted">{workload?.representation ?? 'Loading prepared workload…'}</p>
        <p className="max-w-prose text-label text-fg-muted">{historical
          ? 'Clients are simulated patient groups. These runs use validation data and the existing any-finding task; they do not test the proposed disease-specific model.'
          : 'Clients are simulated patient groups. These runs use the Effusion follow-up\u2019s development selection half; the locked holdout and the official test are not available in the Lab.'}</p>
        {workload && !workload.ready && <p role="status" className="text-warning">{workload.reason ?? 'Qualified workload unavailable.'}</p>}
        {workload?.ready && <p className="text-caption text-fg-muted">Training: {workload.splitSummary?.train?.images?.toLocaleString() ?? '—'} images; {historical
          ? <>validation: {workload.splitSummary?.validation?.images?.toLocaleString() ?? '—'} images.</>
          : <>{workload.splitSummary?.selection?.images?.toLocaleString() ?? '—'} development-selection images.</>} Limit: {workload.limits.deadlineSeconds} seconds, {workload.limits.cpuThreads} CPU threads.</p>}
        <a className="inline-block text-label font-medium text-accent underline" href="/capstone/study-v1/index.html" target="_blank" rel="noreferrer">View completed reference study</a>
      </section>
      {loadError && <p role="alert" className="text-danger">{loadError}</p>}
      {actionError && <p role="alert" className="text-danger">{actionError}</p>}
      <div className="grid items-start gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
        <Card padding="lg" className="space-y-4"><h2 className="text-h3 font-semibold">Configure a run</h2>
          <label className="block text-label">Workload<select className={controlClass} value={choices.workloadId} disabled={reviewing || starting} onChange={e => chooseWorkload(e.target.value)}>
            {workloads.length ? workloads.map(w => <option key={w.id} value={w.id}>{WORKLOAD_NAMES[w.id] ?? w.id}{w.ready ? '' : ' (unavailable)'}</option>)
              : <option value={choices.workloadId}>Loading workloads…</option>}</select></label>
          <div className="flex flex-wrap gap-2"><Button variant="secondary" size="sm" disabled={reviewing || starting} onClick={() => preset({})}>Three-round demo</Button>
            <Button variant="secondary" size="sm" disabled={reviewing || starting} onClick={() => preset({ numClients: 80, rounds: 20, localEpochs: 7 })}>Reference E7/R20 schedule</Button></div>
          <form onSubmit={e => { e.preventDefault(); void review(); }} className="space-y-3">
            <fieldset disabled={reviewing || starting} className="space-y-3">
              {selectors.map(([key, label, options]) => <label key={key} className="block text-label">{label}<select className={controlClass} value={choices[key]} disabled={key === 'epsilon' && choices.privacy === 'none'} onChange={e => change(key, typeof INITIAL[key] === 'number' ? Number(e.target.value) : e.target.value)}>{options.map(([value, name]) => <option key={value} value={value}>{name}</option>)}</select></label>)}
              {([['rounds', 'Rounds', 1, 20], ['localEpochs', 'Local epochs', 1, 7], ['seed', 'Experiment seed', 0, 1e9]] as const).map(([key, label, min, max]) => <label className="block text-label" key={key}>{label}<input className={controlClass} type="number" step={1} min={min} max={max} required value={Number.isFinite(choices[key]) ? choices[key] : ''} onChange={e => change(key, e.target.valueAsNumber)} /></label>)}
            </fieldset>
            <Button type="submit" variant="secondary" disabled={!valid || !workload?.ready || reviewing || starting}>{reviewing ? 'Resolving…' : 'Review recipe'}</Button>
          </form>
          {resolution && <section aria-label="Resolved recipe" className="space-y-2 border-t border-hairline pt-3 text-caption"><h3 className="font-semibold">Resolved recipe</h3>
            <p>{resolution.recipe.numClients} clients, {resolution.recipe.localEpochs} local epochs, {resolution.recipe.rounds} rounds. Learning rate {resolution.recipe.learningRate}, batch size {resolution.recipe.batchSize}. Evaluation: {resolution.recipe.evaluationSplit ?? 'validation'}.</p>
            <p className="break-all">Configuration: {resolution.configHash}</p></section>}
          <Button onClick={() => { void start(); }} disabled={!resolution || occupied || !workload?.ready || starting || reviewing}>{starting ? 'Starting…' : 'Start simulation'}</Button>
          {occupied && <p role="status" className="text-caption text-fg-muted">A worker is active or awaiting recovery. Open owned history or refresh its state.{knownActive && <button className="ml-1 underline" onClick={() => open(knownActive.id)}>Open active run</button>}</p>}
          <p className="text-caption text-fg-muted">Exploratory settings; a valid run can time out. Central DP applies to client-update releases. Diagnostic records are not DP-safe.</p>
        </Card>
        <div className="min-w-0">
          {runError && <div role="alert" className="mb-3 text-danger">{runError}<Button variant="secondary" size="sm" className="ml-2" onClick={() => setPollRetry(n => n + 1)}>Retry selected run</Button></div>}
          {run ? <SimulationRunView key={run.id} run={run} record={record} onStop={() => { void stop(); }} onExport={() => { void download(); }} onInspect={() => { void inspect(); }} stopping={stoppingId === run.id} exporting={exportingId === run.id} inspecting={inspectingId === run.id} />
            : <Card padding="lg"><h2 className="text-h3 font-semibold">{selectedId ? 'Opening selected run' : 'Ready to explore'}</h2><p className="mt-2 text-body text-fg-muted">{selectedId ? 'Waiting for its server-owned state.' : 'Review a recipe to start, or reopen a simulation from your history.'}</p></Card>}
        </div>
      </div>
      <Card padding="lg" className="space-y-4"><div className="flex flex-wrap items-center justify-between gap-3"><h2 className="text-h3 font-semibold">Your simulations</h2><Button variant="secondary" size="sm" onClick={() => { setBusy(false); setRefresh(n => n + 1); setPollRetry(n => n + 1); }}>Refresh history</Button></div>
        <p className="text-caption text-fg-muted">Reopening does not restart training. Select two to four terminal runs for comparison.</p>
        {!history?.content.length ? <p className="text-body text-fg-muted">No owned simulations on this page.</p> : <div className="overflow-x-auto"><table className="w-full text-left text-label"><caption className="sr-only">Owned simulation history</caption><thead><tr>{['Compare', 'Run', 'State', 'Recipe', 'Created'].map(x => <th className="p-2" key={x}>{x}</th>)}</tr></thead><tbody>
          {history.content.map(item => <tr key={item.id} className="border-t border-hairline"><td className="p-2"><input type="checkbox" aria-label={`Compare ${item.id}`} checked={compareIds.includes(item.id)} disabled={comparing || !api.simulationTerminal(item.status) || (compareIds.length >= 4 && !compareIds.includes(item.id))} onChange={e => { setCompareIds(previous => e.target.checked ? [...previous, item.id] : previous.filter(id => id !== item.id)); setComparison([]); }} /></td>
            <td className="p-2"><button className="text-accent underline" aria-label={`Open ${item.id}`} onClick={() => open(item.id)}>{item.id.slice(0, 8)}</button></td><td className="p-2">{statusLabel(item)}</td><td className="p-2">{item.recipe.head} / {item.recipe.aggregation} / {item.recipe.privacy}</td><td className="p-2">{new Date(item.createdAt).toLocaleString()}</td></tr>)}
        </tbody></table></div>}
        <div className="flex flex-wrap items-center gap-3"><Button variant="secondary" size="sm" disabled={page === 0} onClick={() => setPage(n => n - 1)}>Previous page</Button><span className="text-caption">Page {page + 1}</span><Button variant="secondary" size="sm" disabled={!history || (page + 1) * history.size >= history.totalElements} onClick={() => setPage(n => n + 1)}>Next page</Button>
          <Button variant="secondary" disabled={compareIds.length < 2 || compareIds.length > 4 || comparing} onClick={() => { void compare(); }}>{comparing ? 'Loading comparison…' : `Compare selected (${compareIds.length})`}</Button></div>
      </Card>
      {comparison.length > 0 && <SimulationCompare runs={comparison} />}
    </div>
  </main>;
}
