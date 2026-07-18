// Pure reducer for the app-wide training run state (stage 1 of the IA redesign).
//
// This is TrainingScreen's former component state, lifted verbatim into a reducer so Home,
// Projects, and the stage-2 pushes (ProjectDetail, ActivityLog) can all read one run. The
// state machine is explicit — notJoined | joined | training | error — but every transition
// preserves the old screen's behavior exactly:
//   · a JOIN failure → machine 'error' (the old phase === 'error'): joined stays null.
//   · a TRAINING failure keeps machine 'joined' with `error` set — the old screen kept
//     phase 'ready' and showed the ErrorBanner next to the joined card; a training error
//     is NOT the machine's error state.
//   · STOP resets everything except `error` (the old onStop never cleared it).
//   · known refusals (FedAvg-unsupported, model-delivery-unavailable) arrive as INFO log
//     lines, not errors — dispatched by the provider, same as the old catch branch.
//
// Kept pure (no imports beyond types) so it is unit-testable without the native module.
import type { RoundResult, ServerStatus } from '../lib/nativeCore';
import type { JoinedRun } from '../lib/runJoin';

export const MAX_LOG_LINES = 500; // ring the activity log so a long run can't grow memory unbounded

// Activity-log line: server output verbatim, or a client-side line tagged with a severity level
// rendered as a token-colored text prefix (WARN → warning, INFO → muted) — no glyphs/emoji.
export type LogLevel = 'WARN' | 'INFO';
export interface LogLine {
  level?: LogLevel;
  text: string;
}

export type TrainingMachineState = 'notJoined' | 'joined' | 'training' | 'error';

/** Session summary: a pure fold over the rounds completed since the last Start. */
export interface SessionSummary {
  rounds: number;
  scalarsTransmitted: number;
  bytesUp: number;
  bytesDown: number;
  computeMs: number;
}

export interface TrainingState {
  machine: TrainingMachineState;
  /** In-flight joinRun (only meaningful while machine is notJoined | error). */
  joining: boolean;
  /** In-flight stop (native abort + teardown). */
  stopping: boolean;
  error: string | null;
  joined: JoinedRun | null;
  /** Human name of the joined project (handed over from the picker; falls back to the id). */
  projectName: string | null;
  logs: LogLine[];
  latestRound: RoundResult | null;
  serverStatus: ServerStatus | null;
  session: SessionSummary;
}

export const emptySession: SessionSummary = {
  rounds: 0,
  scalarsTransmitted: 0,
  bytesUp: 0,
  bytesDown: 0,
  computeMs: 0,
};

export const initialTrainingState: TrainingState = {
  machine: 'notJoined',
  joining: false,
  stopping: false,
  error: null,
  joined: null,
  projectName: null,
  logs: [],
  latestRound: null,
  serverStatus: null,
  session: emptySession,
};

export type TrainingAction =
  | { type: 'JOIN_START' }
  | { type: 'JOIN_SUCCESS'; joined: JoinedRun; projectName?: string }
  | { type: 'JOIN_FAILURE'; error: string }
  | { type: 'TRAINING_START' }
  | { type: 'ROUND_RESULT'; round: RoundResult }
  | { type: 'TRAINING_ERROR'; error: string }
  | { type: 'TRAINING_END' }
  | { type: 'STOP_START' }
  | { type: 'STOP_COMPLETE' }
  | { type: 'LOG_APPEND'; body: string; level?: LogLevel }
  | { type: 'LOGS_CLEAR' }
  | { type: 'SERVER_STATUS'; status: ServerStatus | null };

/** Split a (possibly multi-line) body into log lines and append onto the capped ring. */
export function appendLogLines(prev: LogLine[], body: string, level?: LogLevel): LogLine[] {
  const incoming = String(body)
    .split('\n')
    .filter((l) => l.length > 0)
    .map((text): LogLine => ({ level, text }));
  if (incoming.length === 0) return prev;
  const next = prev.concat(incoming);
  return next.length > MAX_LOG_LINES ? next.slice(next.length - MAX_LOG_LINES) : next;
}

export function trainingReducer(state: TrainingState, action: TrainingAction): TrainingState {
  switch (action.type) {
    case 'JOIN_START':
      // Old onJoin: setError(null); setPhase('joining').
      return { ...state, joining: true, error: null };
    case 'JOIN_SUCCESS':
      return {
        ...state,
        machine: 'joined',
        joining: false,
        joined: action.joined,
        projectName: action.projectName ?? action.joined.projectId,
      };
    case 'JOIN_FAILURE':
      return { ...state, machine: 'error', joining: false, error: action.error };
    case 'TRAINING_START':
      // Old onStartTraining: setError(null); setLatestRound(null); setTraining(true).
      // The session fold restarts with the run ("THIS SESSION").
      return {
        ...state,
        machine: 'training',
        error: null,
        latestRound: null,
        session: emptySession,
      };
    case 'ROUND_RESULT':
      return {
        ...state,
        latestRound: action.round,
        session: {
          rounds: state.session.rounds + 1,
          scalarsTransmitted: state.session.scalarsTransmitted + action.round.scalarsTransmitted,
          bytesUp: state.session.bytesUp + action.round.uplinkBytes,
          bytesDown: state.session.bytesDown + action.round.downlinkBytes,
          computeMs: state.session.computeMs + action.round.computeMs,
        },
      };
    case 'TRAINING_ERROR':
      // Old catch branch: setError(msg) + a WARN log line; the run stays joined.
      return { ...state, error: action.error, logs: appendLogLines(state.logs, action.error, 'WARN') };
    case 'TRAINING_END':
      // finally-block: back to the joined view — unless a stop already tore the run down.
      return { ...state, machine: state.joined ? 'joined' : 'notJoined' };
    case 'STOP_START':
      return { ...state, stopping: true };
    case 'STOP_COMPLETE':
      // Old onStop reset everything to idle — except `error`, which it never touched.
      return {
        ...initialTrainingState,
        error: state.error,
      };
    case 'LOG_APPEND':
      return { ...state, logs: appendLogLines(state.logs, action.body, action.level) };
    case 'LOGS_CLEAR':
      return { ...state, logs: [] };
    case 'SERVER_STATUS':
      return { ...state, serverStatus: action.status };
    default:
      return state;
  }
}
