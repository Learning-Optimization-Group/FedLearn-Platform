import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Platform, Pressable, ScrollView, Text, View } from 'react-native';
import { useRoute, useNavigation } from '@react-navigation/native';
import { Play, Square } from 'lucide-react-native';

import { joinRun, type JoinedRun } from '../lib/runJoin';
import nativeCore, { isNativeCoreAvailable, type RoundResult } from '../lib/nativeCore';
import { connectStomp, type StompHandle } from '../lib/stompClient';
import { foregroundService } from '../lib/foregroundService';
import { runTrainingLoop } from '../lib/training';
import { ModelDeliveryUnavailableError } from '../lib/modelProvisioning';
import { StatusBadge, type StatusVariant } from '../components/StatusBadge';
import { DeviceBanner } from '../components/DeviceBanner';
import { useThemeTokens } from '../theme/useThemeTokens';

const MAX_LOG_LINES = 500; // ring the activity log so a long run can't grow memory unbounded

type Phase = 'idle' | 'joining' | 'ready' | 'error';

export function TrainingScreen() {
  const { colors } = useThemeTokens();
  // Whether the native C++ FL core is compiled into this build. Absent on the iOS scaffold (MO-14),
  // where on-device training can't run — we render an explicit unavailable state instead (MO-5).
  const nativeAvailable = isNativeCoreAvailable();
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [joined, setJoined] = useState<JoinedRun | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [stopping, setStopping] = useState(false);
  const [training, setTraining] = useState(false);
  const [latestRound, setLatestRound] = useState<RoundResult | null>(null);
  const logScrollRef = useRef<ScrollView | null>(null);
  const stompRef = useRef<StompHandle | null>(null);
  const stopRef = useRef(false); // cooperative stop flag polled by the training loop

  // Route params from the project picker (Task 4 → navigation.navigate('Training', { projectId })).
  const route = useRoute<any>();
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
      setError(String(e));
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
        const handle = await connectStomp((msg) => alive && appendLog(setLogs, `⚠ ${msg}`));
        if (!alive) {
          handle.deactivate();
          return;
        }
        stompRef.current = handle;
        handle.subscribe(`/topic/logs/${joined.projectId}`, (body) =>
          appendLog(setLogs, body),
        );
      } catch (e) {
        if (alive) appendLog(setLogs, `⚠ ${String(e)}`);
      }
    })();
    return () => {
      alive = false;
      stompRef.current?.deactivate();
      stompRef.current = null;
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
      if (e instanceof ModelDeliveryUnavailableError) {
        appendLog(setLogs, `ℹ ${e.message}`);
      } else {
        setError(String(e));
        appendLog(setLogs, `⚠ ${String(e)}`);
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
      <ScrollView className="flex-1 bg-canvas">
        <View className="px-4 pt-4 pb-2 flex-row items-center justify-between">
          <Text className="text-h2 font-sans text-fg">Training</Text>
          <StatusBadge label="Unavailable" variant="idle" />
        </View>
        <View className="mx-4 mt-2 p-4 rounded-card bg-surface-1 border border-hairline">
          <Text className="text-body font-sans text-fg mb-1">{heading}</Text>
          <Text className="text-caption text-fg-muted">
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
    <ScrollView className="flex-1 bg-canvas">
      <View className="px-4 pt-4 pb-2 flex-row items-center justify-between">
        <Text className="text-h2 font-sans text-fg">Training</Text>
        <StatusBadge label={badge.label} variant={badge.variant} />
      </View>

      <DeviceBanner metrics={null} />

      {!joined ? (
        <View className="mx-4 mt-2 p-4 rounded-card bg-surface-1 border border-hairline">
          <Text className="text-body font-sans text-fg mb-1">Join a training run</Text>
          <Text className="text-caption text-fg-muted mb-3">Your data stays on this device — only learning updates are shared.</Text>
          {projectId ? (
            <>
              <Text className="text-label font-sans text-fg-muted">Selected project</Text>
              <Text className="mt-1 text-body font-mono text-fg">{projectId}</Text>
              <Pressable className="mt-3" onPress={() => navigation.navigate('Projects')}>
                <Text className="text-body font-sans text-accent">Change project</Text>
              </Pressable>
            </>
          ) : (
            <Pressable className="py-2" onPress={() => navigation.navigate('Projects')}>
              <Text className="text-body font-sans text-accent">Choose a project to train</Text>
            </Pressable>
          )}
          <Pressable
            className="flex-row items-center justify-center bg-accent rounded-md py-3"
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
        <View className="mx-4 mt-2 p-4 rounded-card bg-surface-1 border border-hairline">
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
          <Text className="mt-3 text-caption font-sans text-fg-subtle">
            Training runs entirely on this device — only learning updates (perturbation seeds + gradient
            scalars) are shared, never your data.
          </Text>
        </View>

        {/* Start on-device training, or show live round progress while it runs. */}
        {training ? (
          <View className="mx-4 mt-2 p-4 rounded-card bg-surface-1 border border-hairline">
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
        ) : (
          <Pressable
            className="mx-4 mt-2 flex-row items-center justify-center bg-accent rounded-card py-3"
            onPress={() => {
              void onStartTraining();
            }}>
            <Play color={colors['accent-fg']} size={18} strokeWidth={1.5} />
            <Text className="text-accent-fg text-label font-sans ml-2">Start training</Text>
          </Pressable>
        )}

        {/* Stop control — aborts the native gRPC/training path and the foreground service. */}
        <Pressable
          className="mx-4 mt-2 flex-row items-center justify-center bg-danger rounded-card py-3"
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

        {/* Live activity log (server round output over STOMP /topic/logs). */}
        <View className="mx-4 mt-3 rounded-card bg-code-well border border-hairline overflow-hidden">
          <View className="px-3 py-2 border-b border-hairline">
            <Text className="text-caption font-sans text-fg-muted">Activity log</Text>
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
                  {line}
                </Text>
              ))
            )}
          </ScrollView>
        </View>
        </>
      )}

      {error ? (
        <View className="mx-4 my-3 p-3 rounded-md bg-danger">
          <Text className="text-accent-fg text-body">{error}</Text>
        </View>
      ) : null}
      <View className="h-8" />
    </ScrollView>
  );
}

// Append server log line(s) to the ring-buffered activity log (capped so a long run stays bounded).
function appendLog(
  setLogs: React.Dispatch<React.SetStateAction<string[]>>,
  body: string,
): void {
  const incoming = String(body)
    .split('\n')
    .filter((l) => l.length > 0);
  if (incoming.length === 0) return;
  setLogs((prev) => {
    const next = prev.concat(incoming);
    return next.length > MAX_LOG_LINES ? next.slice(next.length - MAX_LOG_LINES) : next;
  });
}
