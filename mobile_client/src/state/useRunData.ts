import { useCallback, useEffect, useState } from 'react';

import { listDatasets, type DatasetSnapshot } from '../lib/datasetService';
import { datasetFit, importForRun, loadRunDataPlan, type RunDataPlan } from '../lib/runDataPlan';
import { fetchRunManifest, type JoinedRun } from '../lib/runJoin';

/** The joined run's data plan and the datasets on this device that could train it (Stage 3 slice B2). */
export interface RunData {
  /** null while loading, for a run this device will not train, or when not joined. */
  plan: RunDataPlan | null;
  datasets: DatasetSnapshot[];
  selectedId: string | null;
  select: (snapshotId: string) => void;
  importing: boolean;
  importError: string | null;
  importFile: () => Promise<void>;
  /** The chosen dataset, when it fits the run; what Start trains. */
  selected: DatasetSnapshot | undefined;
  /** False only for a run on the device's own data with no fitting dataset chosen. */
  readyToStart: boolean;
}

export function useRunData(joined: JoinedRun | null): RunData {
  const [plan, setPlan] = useState<RunDataPlan | null>(null);
  const [datasets, setDatasets] = useState<DatasetSnapshot[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

  useEffect(() => {
    setPlan(null);
    setDatasets([]);
    setSelectedId(null);
    setImportError(null);
    if (!joined) return;
    let alive = true;
    (async () => {
      try {
        const p = await loadRunDataPlan(joined, { fetchManifest: fetchRunManifest });
        if (!alive) return;
        setPlan(p);
        if (p?.source === 'LOCAL_SNAPSHOT') {
          const list = await listDatasets();
          if (alive) setDatasets(list);
        }
      } catch {
        // No plan: Start still runs the training loop, which reports why this device cannot train the run.
        if (alive) setPlan(null);
      }
    })();
    return () => {
      alive = false;
    };
  }, [joined]);

  const importFile = useCallback(async () => {
    if (plan?.source !== 'LOCAL_SNAPSHOT') return;
    setImporting(true);
    setImportError(null);
    try {
      const { snapshot, error } = await importForRun(plan);
      if (error) setImportError(error);
      if (snapshot) {
        setDatasets(await listDatasets());
        if (datasetFit(snapshot, plan).length === 0) setSelectedId(snapshot.snapshotId);
      }
    } finally {
      setImporting(false);
    }
  }, [plan]);

  const selected = plan?.source === 'LOCAL_SNAPSHOT'
    ? datasets.find((d) => d.snapshotId === selectedId && datasetFit(d, plan).length === 0)
    : undefined;
  return {
    plan, datasets, selectedId, select: setSelectedId, importing, importError, importFile, selected,
    readyToStart: plan?.source !== 'LOCAL_SNAPSHOT' || selected !== undefined,
  };
}
