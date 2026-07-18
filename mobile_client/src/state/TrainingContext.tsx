// App-wide training run state: TrainingScreen's former component state + effects, lifted
// into a provider mounted under the authenticated navigator (AppNavigator) so Home,
// Projects, and the stage-2 pushes all read/drive ONE run.
//
// Ownership rules preserved from the screen:
//   · exactly one STOMP subscription per joined run (the provider owns the handle — a
//     pushed screen reading `logs` must never re-subscribe);
//   · the MO-3 server-status heartbeat runs while joined, independent of the round loop;
//   · the cooperative stop flag (stopRef) is polled by the training loop; Stop also aborts
//     the native path (nativeCore.stop) and the Android foreground service;
//   · known refusals (FedAvg-unsupported / model-delivery-unavailable) log as INFO, all
//     other loop failures set `error` + a WARN line — identical to the old catch branch.
//
// New in the lift (additive, not behavioral): every completed round is appended to the
// persisted contribution ledger (contributionLedger.ts).
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
} from 'react';

import { joinRun } from '../lib/runJoin';
import nativeCore from '../lib/nativeCore';
import { connectStomp, type StompHandle } from '../lib/stompClient';
import { foregroundService } from '../lib/foregroundService';
import { runTrainingLoop, MobileFedAvgUnsupportedError } from '../lib/training';
import { startServerStatusHeartbeat } from '../lib/statusHeartbeat';
import { ModelDeliveryUnavailableError } from '../lib/modelProvisioning';
import { readError } from '../lib/errors';
import { contributionLedger } from '../lib/contributionLedger';
import {
  initialTrainingState,
  trainingReducer,
  type TrainingState,
} from './trainingReducer';

export interface TrainingContextValue {
  state: TrainingState;
  /** Join the project's active run (registers the native FL client). */
  join: (projectId: string, projectName?: string) => Promise<void>;
  /** Start the on-device training loop for the joined run. */
  startTraining: () => Promise<void>;
  /** Abort training + native path + foreground service and reset to notJoined. */
  stopTraining: () => Promise<void>;
}

const TrainingContext = createContext<TrainingContextValue | null>(null);

export function TrainingProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(trainingReducer, initialTrainingState);
  const stompRef = useRef<StompHandle | null>(null);
  const stopRef = useRef(false); // cooperative stop flag polled by the training loop
  const joined = state.joined;
  const projectName = state.projectName;

  // Once joined, stream the server's round logs for this project (parity with desktop's Activity
  // Log). Same /ws-logs STOMP endpoint the web dashboard uses, authenticated with the mobile
  // Bearer token. The provider is the ONE owner of this subscription.
  useEffect(() => {
    if (!joined) return;
    let alive = true;
    dispatch({ type: 'LOGS_CLEAR' });
    (async () => {
      try {
        const handle = await connectStomp(
          (msg) => alive && dispatch({ type: 'LOG_APPEND', body: msg, level: 'WARN' }),
        );
        if (!alive) {
          handle.deactivate();
          return;
        }
        stompRef.current = handle;
        handle.subscribe(`/topic/logs/${joined.projectId}`, (body) =>
          dispatch({ type: 'LOG_APPEND', body }),
        );
      } catch (e) {
        if (alive) dispatch({ type: 'LOG_APPEND', body: readError(e), level: 'WARN' });
      }
    })();
    return () => {
      alive = false;
      stompRef.current?.deactivate();
      stompRef.current = null;
    };
  }, [joined]);

  // MO-3: once joined, a server-status heartbeat independent of the training round cadence keeps
  // the live round number / deadline honest even while a DeComFL round occupies the loop.
  // Best-effort: a failed poll is swallowed (the heartbeat keeps retrying).
  useEffect(() => {
    if (!joined) {
      dispatch({ type: 'SERVER_STATUS', status: null });
      return;
    }
    const hb = startServerStatusHeartbeat({
      runId: joined.runId,
      onStatus: (status) => dispatch({ type: 'SERVER_STATUS', status }),
      onError: () => {
        /* best-effort telemetry — the card holds its last value until a poll succeeds */
      },
    });
    return () => {
      hb.stop();
    };
  }, [joined]);

  const join = useCallback(async (projectId: string, name?: string) => {
    dispatch({ type: 'JOIN_START' });
    try {
      const result = await joinRun({ projectId });
      dispatch({ type: 'JOIN_SUCCESS', joined: result, projectName: name });
    } catch (e) {
      // MO-16: readError, not String(e) — an axios join failure otherwise renders as
      // the meaningless "[object Object]".
      dispatch({ type: 'JOIN_FAILURE', error: readError(e) });
    }
  }, []);

  // Start the on-device training loop: stage the model + local data, then run rounds. All compute
  // is on-device; only seeds + gradient scalars are uploaded (raw data never leaves).
  const startTraining = useCallback(async () => {
    if (!joined) return;
    const ledgerProjectName = projectName ?? joined.projectId;
    dispatch({ type: 'TRAINING_START' });
    stopRef.current = false;
    foregroundService.start();
    try {
      await runTrainingLoop(joined, {
        onLog: (line) => dispatch({ type: 'LOG_APPEND', body: line }),
        onRound: (r) => {
          dispatch({ type: 'ROUND_RESULT', round: r });
          // Persist the completed round to the device-local contribution ledger. Best-effort:
          // a storage failure must never interrupt the run.
          void contributionLedger
            .record({
              projectId: joined.projectId,
              projectName: ledgerProjectName,
              round: r.round,
              wallClockMs: r.computeMs,
              bytesUp: r.uplinkBytes,
              bytesDown: r.downlinkBytes,
              at: new Date().toISOString(),
            })
            .catch(() => {});
        },
        shouldStop: () => stopRef.current,
      });
    } catch (e) {
      if (e instanceof ModelDeliveryUnavailableError || e instanceof MobileFedAvgUnsupportedError) {
        // Known "can't train here (yet)" refusals — informational, not a failure.
        dispatch({ type: 'LOG_APPEND', body: e.message, level: 'INFO' });
      } else {
        dispatch({ type: 'TRAINING_ERROR', error: readError(e) });
      }
    } finally {
      foregroundService.stop();
      dispatch({ type: 'TRAINING_END' });
    }
  }, [joined, projectName]);

  // Stop = abort the native gRPC/training path (sets the abort flag + joins threads), stop the
  // Android foreground service, tear down the STOMP stream, and reset to notJoined.
  const stopTraining = useCallback(async () => {
    dispatch({ type: 'STOP_START' });
    stopRef.current = true; // break the training loop before the native abort
    try {
      await nativeCore.stop();
    } catch {
      /* already stopped / not registered */
    }
    foregroundService.stop();
    stompRef.current?.deactivate();
    stompRef.current = null;
    dispatch({ type: 'STOP_COMPLETE' });
  }, []);

  const value = useMemo<TrainingContextValue>(
    () => ({ state, join, startTraining, stopTraining }),
    [state, join, startTraining, stopTraining],
  );

  return <TrainingContext.Provider value={value}>{children}</TrainingContext.Provider>;
}

/** Read the shared training run. Throws outside <TrainingProvider> (a wiring bug, not a state). */
export function useTraining(): TrainingContextValue {
  const ctx = useContext(TrainingContext);
  if (!ctx) throw new Error('useTraining must be used within a TrainingProvider');
  return ctx;
}
