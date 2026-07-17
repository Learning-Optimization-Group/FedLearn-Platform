import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Platform, Pressable, ScrollView, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRoute, useNavigation } from '@react-navigation/native';
import { Play, Square } from 'lucide-react-native';

import { joinRun, type JoinedRun } from '../lib/runJoin';
import nativeCore, { isNativeCoreAvailable, type RoundResult, type ServerStatus } from '../lib/nativeCore';
import { connectStomp, type StompHandle } from '../lib/stompClient';
import { foregroundService } from '../lib/foregroundService';
import { runTrainingLoop, MobileFedAvgUnsupportedError } from '../lib/training';
import { startServerStatusHeartbeat, formatRoundDeadline } from '../lib/statusHeartbeat';
import { ModelDeliveryUnavailableError } from '../lib/modelProvisioning';
import { readError } from '../lib/errors';
import { StatusBadge, type StatusVariant } from '../components/StatusBadge';
import { DeviceBanner } from '../components/DeviceBanner';
import { ErrorBanner } from '../components/ErrorBanner';
import { useThemeTokens } from '../theme/useThemeTokens';

const MAX_LOG_LINES = 500; // ring the activity log so a long run can't grow memory unbounded

type Phase = 'idle' | 'joining' | 'ready' | 'error';

// Activity-log line: server output verbatim, or a client-side line tagged with a severity level
// rendered as a token-colored text prefix (WARN → warning, INFO → muted) — no glyphs/emoji.
type LogLevel = 'WARN' | 'INFO';
type LogLine = { level?: LogLevel; text: string };

export function TrainingScreen() {
  const { colors } = useThemeTokens();
  const insets = useSafeAreaInsets(); // top safe area on both roots; bottom belongs to the tab bar
  // Whether the native C++ FL core is compiled into this build. Absent on the iOS scaffold (MO-14),
  // where on-device training can't run — we render an explicit unavailable state instead (MO-5).
  const nativeAvailable = isNativeCoreAvailable();
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [joined, setJoined] = useState<JoinedRun | null>(null);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [stopping, setStopping] = useState(false);
  const [training, setTraining] = useState(false);
  const [latestRound, setLatestRound] = useState<RoundResult | null>(null);
  // MO-3: the server's LIVE round + deadline, polled on a heartbeat independent of the local round
  // cadence. `now` is a 1s ticker so the deadline renders as a live countdown between heartbeats.
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null);
  const [now, setNow] = useState<number>(() => Date.now());
  const logScrollRef = useRef<ScrollView | null>(null);
  const stompRef = useRef<StompHandle | null>(null);
  const stopRef = useRef(false); // cooperative stop flag polled by the training loop

  // Route params from the project picker (Task 4 → navigation.navigate('Training', { projectId })).
  // No typed ParamList exists yet for the navigators (see RootNavigator.tsx) — same rationale as
  // ProjectPickerScreen.tsx's identical useNavigation<any>().
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const route = useRoute<any>();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const navigation = useNavigation<any>();
  const projectId: string = route.params?.projectId ?? '';

  const badge: { label: string; variant: StatusVariant } = {
    idle: { label: 'Idle', variant: 'idle' as StatusVariant },
    joining: { label: 'Joining…', variant: 'running' as StatusVariant },
    ready: { label: 'Ready', variant: 'success' as StatusVariant },
    error: { label: 'Error', variant: 'danger' as StatusVariant },
  }[phase];

  const onJoin = useCallback(async () => {
    setError(null);
    setPhase('joining');
    try {
      const result = await joinRun({ projectId });
      setJoined(result);
      setPhase('ready');
    } catch (e) {
      // MO-16: readError, not String(e) — an axios join failure (e.g. a 409 "Run is not currently
      // running", or "No active run yet") otherwise renders as the meaningless "[object Object]".
      setError(readError(e));
      setPhase('error');
    }
  }, [projectId]);

  // Once joined, stream the server's round logs for this project (parity with desktop's Activity Log).
  // Same /ws-logs STOMP endpoint the web dashboard uses, authenticated with the mobile Bearer token.
  useEffect(() => {
    if (!joined) return;
    let alive = true;
    setLogs([]);
    (async () => {
      try {
        const handle = await connectStomp((msg) => alive && appendLog(setLogs, msg, 'WARN'));
        if (!alive) {
          handle.deactivate();
          return;
        }
        stompRef.current = handle;
        handle.subscribe(`/topic/logs/${joined.projectId}`, (body) =>
          appendLog(setLogs, body),
        );
      } catch (e) {
        if (alive) appendLog(setLogs, readError(e), 'WARN');
      }
    })();
    return () => {
      alive = false;
      stompRef.current?.deactivate();
      stompRef.current = null;
    };
  }, [joined]);

  // MO-3: once joined, run a server-status heartbeat + a 1s countdown ticker, both independent of the
  // training round loop. This keeps the live round number and round-deadline countdown honest even
  // while a DeComFL round is occupying the loop (or between rounds). Best-effort: a failed poll is
  // swallowed here (the heartbeat itself keeps retrying) so a blip never freezes the view for good.
  useEffect(() => {
    if (!joined) {
      setServerStatus(null);
      return;
    }
    const hb = startServerStatusHeartbeat({
      runId: joined.runId,
      onStatus: setServerStatus,
      onError: () => {
        /* best-effort telemetry — the card just holds its last value until a poll succeeds */
      },
    });
    setNow(Date.now());
    const ticker = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      hb.stop();
      clearInterval(ticker);
    };
  }, [joined]);

  // Stop = abort the native gRPC/training path (sets the abort flag + joins threads), stop the Android
  // foreground service, and reset the UI to idle.
  const onStop = useCallback(async () => {
    setStopping(true);
    stopRef.current = true; // break the training loop before the native abort
    try {
      await nativeCore.stop();
    } catch {
      /* already stopped / not registered */
    }
    foregroundService.stop();
    stompRef.current?.deactivate();
    stompRef.current = null;
    setJoined(null);
    setLogs([]);
    setTraining(false);
    setLatestRound(null);
    setServerStatus(null);
    setPhase('idle');
    setStopping(false);
  }, []);

  // Start the on-device training loop: stage the model + local data, then run rounds. All compute is
  // on-device; only seeds + gradient scalars are uploaded (raw data never leaves).
  const onStartTraining = useCallback(async () => {
    if (!joined) return;
    setError(null);
    setLatestRound(null);
    stopRef.current = false;
    setTraining(true);
    foregroundService.start();
    try {
      await runTrainingLoop(joined, {
        onLog: (line) => appendLog(setLogs, line),
        onRound: (r) => setLatestRound(r),
        shouldStop: () => stopRef.current,
      });
    } catch (e) {
      if (e instanceof ModelDeliveryUnavailableError || e instanceof MobileFedAvgUnsupportedError) {
        // Known "can't train here (yet)" refusals — informational, not a failure. MO-4 / model-delivery.
        appendLog(setLogs, e.message, 'INFO');
      } else {
        const msg = readError(e);
        setError(msg);
        appendLog(setLogs, msg, 'WARN');
      }
    } finally {
      foregroundService.stop();
      setTraining(false);
    }
  }, [joined]);

  // Native core absent (iOS scaffold): disable the training entry point entirely and explain why,
  // rather than letting Join/Start call into a native module that isn't there. All hooks above still
  // run unconditionally, so this early return respects the Rules of Hooks.
  if (!nativeAvailable) {
    const heading =
      Platform.OS === 'ios' ? 'iOS training preview unavailable' : 'On-device training unavailable';
    return (
      <ScrollView className="flex-1 bg-canvas" style={{ paddingTop: insets.top }}>
        <View className="px-4 pt-4 pb-2 flex-row items-center justify-between">
          <Text className="text-h2 font-sans text-fg">Training</Text>
          <StatusBadge label="Unavailable" variant="idle" />
        </View>
        <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
          <Text className="text-label font-sans font-semibold text-fg mb-1">{heading}</Text>
          <Text className="text-caption font-sans text-fg-muted">
            The native federated-learning core isn’t built into this app on
            {Platform.OS === 'ios' ? ' iOS' : ' this platform'}, so on-device training is disabled here.
            You can still browse projects and models. On-device training runs on Android; the native iOS
            port is in progress (MO-14).
          </Text>
        </View>
        <View className="h-8" />
      </ScrollView>
    );
  }

  return (
    <ScrollView className="flex-1 bg-canvas" style={{ paddingTop: insets.top }}>
      <View className="px-4 pt-4 pb-2 flex-row items-center justify-between">
        <Text className="text-h2 font-sans text-fg">Training</Text>
        <StatusBadge label={badge.label} variant={badge.variant} />
      </View>

      <DeviceBanner metrics={null} />

      {!joined ? (
        <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
          <Text className="text-label font-sans font-semibold text-fg mb-1">Join a training run</Text>
          <Text className="text-caption font-sans text-fg-muted mb-3">Your data stays on this device — only learning updates are shared.</Text>
          {projectId ? (
            <>
              <Text className="text-label font-sans text-fg-muted">Selected project</Text>
              <Text className="mt-1 text-body font-mono text-fg">{projectId}</Text>
              <Pressable
                accessibilityRole="link"
                accessibilityLabel="Change project"
                className="mt-3 py-2 self-start active:opacity-80"
                hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                onPress={() => navigation.navigate('Projects')}>
                <Text className="text-body font-sans text-accent">Change project</Text>
              </Pressable>
            </>
          ) : (
            <Pressable
              accessibilityRole="link"
              accessibilityLabel="Choose a project to train"
              className="py-2 self-start active:opacity-80"
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
              onPress={() => navigation.navigate('Projects')}>
              <Text className="text-body font-sans text-accent">Choose a project to train</Text>
            </Pressable>
          )}
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Join run"
            accessibilityState={{ disabled: phase === 'joining' }}
            className={`mt-4 flex-row items-center justify-center bg-accent rounded-md py-3 active:opacity-80 ${
              phase === 'joining' ? 'opacity-50' : ''
            }`}
            disabled={phase === 'joining'}
            onPress={onJoin}>
            {phase === 'joining' ? (
              <ActivityIndicator color={colors['accent-fg']} />
            ) : (
              <Text className="text-accent-fg text-label font-sans">Join run</Text>
            )}
          </Pressable>
        </View>
      ) : (
        <>
        <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
          <Text className="text-label font-sans text-fg-muted">Registered with run</Text>
          <Text className="mt-1 text-body font-mono text-fg">{joined.runId}</Text>
          <View className="mt-3 flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">Partition</Text>
            <Text className="text-caption font-mono text-fg">{joined.partitionId}</Text>
          </View>
          <View className="mt-1 flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">Assigned round</Text>
            <Text className="text-caption font-mono text-fg">{joined.assignedRound}</Text>
          </View>
          <View className="mt-1 flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">Model</Text>
            <Text className="text-caption font-sans text-fg">{joined.manifest.recipeKey} · {joined.manifest.strategy}</Text>
          </View>
        </View>

        {/* MO-3: live server progress — the run's current round + deadline countdown + how many clients
            have reported this round, refreshed on a heartbeat independent of this device's round pace. */}
        {serverStatus ? (
          <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
            <View className="flex-row items-center justify-between">
              <Text className="text-label font-sans text-fg-muted">Server progress</Text>
              <Text className="text-caption font-mono text-fg-subtle">{serverStatus.serverState}</Text>
            </View>
            <View className="mt-2 flex-row justify-between">
              <Text className="text-caption font-sans text-fg-muted">Current round</Text>
              <Text className="text-caption font-mono text-fg">{serverStatus.currentRound}</Text>
            </View>
            <View className="mt-1 flex-row justify-between">
              <Text className="text-caption font-sans text-fg-muted">Round deadline</Text>
              <Text className="text-caption font-mono text-fg">
                {formatRoundDeadline(serverStatus.roundDeadlineUnixMs, now)}
              </Text>
            </View>
            <View className="mt-1 flex-row justify-between">
              <Text className="text-caption font-sans text-fg-muted">Clients reported</Text>
              <Text className="text-caption font-mono text-fg">
                {serverStatus.receivedUpdatesThisRound}/{serverStatus.requiredClientsForRound}
              </Text>
            </View>
            <View className="mt-1 flex-row justify-between">
              <Text className="text-caption font-sans text-fg-muted">Active clients</Text>
              <Text className="text-caption font-mono text-fg">{serverStatus.activeClients}</Text>
            </View>
          </View>
        ) : null}

        {/* One action slot: Start when idle; while a run is training it becomes the (danger) Stop —
            aborting the native gRPC/training path and the foreground service. */}
        {training ? (
          <>
            <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
              <View className="flex-row items-center justify-between">
                <Text className="text-body font-sans text-fg">Training on device…</Text>
                <ActivityIndicator color={colors.accent} />
              </View>
              {latestRound ? (
                <>
                  <View className="mt-2 flex-row justify-between">
                    <Text className="text-caption font-sans text-fg-muted">Round</Text>
                    <Text className="text-caption font-mono text-fg">{latestRound.round}</Text>
                  </View>
                  <View className="mt-1 flex-row justify-between">
                    <Text className="text-caption font-sans text-fg-muted">Loss</Text>
                    <Text className="text-caption font-mono text-fg">{latestRound.loss.toFixed(4)}</Text>
                  </View>
                  <View className="mt-1 flex-row justify-between">
                    <Text className="text-caption font-sans text-fg-muted">Scalars uploaded</Text>
                    <Text className="text-caption font-mono text-fg">{latestRound.scalarsTransmitted}</Text>
                  </View>
                </>
              ) : (
                <Text className="mt-2 text-caption font-sans text-fg-subtle">Staging model + on-device data…</Text>
              )}
            </View>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Stop training"
              accessibilityState={{ disabled: stopping }}
              className={`mx-4 mt-3 flex-row items-center justify-center bg-danger rounded-md py-3 active:opacity-80 ${
                stopping ? 'opacity-50' : ''
              }`}
              disabled={stopping}
              onPress={() => {
                void onStop();
              }}>
              {stopping ? (
                <ActivityIndicator color={colors['accent-fg']} />
              ) : (
                <>
                  <Square color={colors['accent-fg']} size={18} strokeWidth={1.5} />
                  <Text className="text-accent-fg text-label font-sans ml-2">Stop training</Text>
                </>
              )}
            </Pressable>
          </>
        ) : (
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Start training"
            className="mx-4 mt-3 flex-row items-center justify-center bg-accent rounded-md py-3 active:opacity-80"
            onPress={() => {
              void onStartTraining();
            }}>
            <Play color={colors['accent-fg']} size={18} strokeWidth={1.5} />
            <Text className="text-accent-fg text-label font-sans ml-2">Start training</Text>
          </Pressable>
        )}

        {/* Live activity log (server round output over STOMP /topic/logs). */}
        <View className="mx-4 mt-3 rounded-card bg-code-well border border-hairline overflow-hidden">
          <View className="px-3 py-2 border-b border-hairline">
            <Text className="text-label font-sans font-semibold text-fg">Activity log</Text>
          </View>
          <ScrollView
            ref={logScrollRef}
            className="px-3 py-2"
            style={{ maxHeight: 260 }}
            onContentSizeChange={() => logScrollRef.current?.scrollToEnd({ animated: true })}>
            {logs.length === 0 ? (
              <Text className="text-caption font-mono text-fg-subtle">
                Waiting for server output…
              </Text>
            ) : (
              logs.map((line, i) => (
                <Text key={i} className="text-caption font-mono text-fg" selectable>
                  {line.level ? (
                    <Text
                      className={`text-caption font-mono ${
                        line.level === 'WARN' ? 'text-warning' : 'text-fg-muted'
                      }`}>
                      {`${line.level} `}
                    </Text>
                  ) : null}
                  {line.text}
                </Text>
              ))
            )}
          </ScrollView>
        </View>
        </>
      )}

      {error ? <ErrorBanner message={error} className="mx-4 my-3" /> : null}
      <View className="h-8" />
    </ScrollView>
  );
}

// Append server log line(s) to the ring-buffered activity log (capped so a long run stays bounded).
// A `level` tags client-side lines with a severity the renderer colors by token.
function appendLog(
  setLogs: React.Dispatch<React.SetStateAction<LogLine[]>>,
  body: string,
  level?: LogLevel,
): void {
  const incoming = String(body)
    .split('\n')
    .filter((l) => l.length > 0)
    .map((text): LogLine => ({ level, text }));
  if (incoming.length === 0) return;
  setLogs((prev) => {
    const next = prev.concat(incoming);
    return next.length > MAX_LOG_LINES ? next.slice(next.length - MAX_LOG_LINES) : next;
  });
}
