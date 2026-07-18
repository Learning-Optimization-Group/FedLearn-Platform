// HOME — the default landing tab: "what is my phone doing, and what did it contribute".
//
// Replaces the old Training tab. All run state lives in TrainingContext (the provider owns
// the STOMP stream, heartbeat, and stop semantics); this screen is a view over it plus two
// local presentation concerns: a 1s ticker for the deadline countdown and the polled device
// metrics that finally feed DeviceBanner real data (it received a hard-coded null before).
//
// Top-to-bottom: header + StatusBadge · DeviceBanner · NOW card (state machine) ·
// THIS SESSION fold · recent contribution-ledger rows · live activity log · ErrorBanner.
// One primary action per state: Join run → Start training → Stop training (danger).
import React, { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Platform, Pressable, ScrollView, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect, useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import { Play, Square } from 'lucide-react-native';

import { isNativeCoreAvailable, type DeviceMetrics } from '../lib/nativeCore';
import { startDeviceMetricsPoll } from '../lib/deviceMetricsPoll';
import { formatRoundDeadline } from '../lib/statusHeartbeat';
import { contributionLedger, type ContributionEntry } from '../lib/contributionLedger';
import { formatBytes, formatDurationMs } from '../lib/format';
import { useTraining } from '../state/TrainingContext';
import { StatusBadge, type StatusVariant } from '../components/StatusBadge';
import { DeviceBanner } from '../components/DeviceBanner';
import { ErrorBanner } from '../components/ErrorBanner';
import { useThemeTokens } from '../theme/useThemeTokens';
import type { MainTabParamList, MainTabScreenProps } from '../navigation/types';

const RECENT_LEDGER_ROWS = 5;

export function HomeScreen() {
  const { colors } = useThemeTokens();
  const insets = useSafeAreaInsets(); // top safe area on the root; bottom belongs to the tab bar
  // Whether the native C++ FL core is compiled into this build. Absent on the iOS scaffold (MO-14),
  // where on-device training can't run — we render an explicit unavailable state instead (MO-5).
  const nativeAvailable = isNativeCoreAvailable();
  const { state, join, startTraining, stopTraining } = useTraining();
  const { machine, joining, stopping, error, joined, logs, latestRound, serverStatus, session } =
    state;

  const route = useRoute<RouteProp<MainTabParamList, 'Home'>>();
  const navigation = useNavigation<MainTabScreenProps<'Home'>['navigation']>();
  const projectId: string = route.params?.projectId ?? '';
  const projectName: string | undefined = route.params?.projectName;

  const logScrollRef = React.useRef<ScrollView | null>(null);

  // 1s ticker so the round-deadline countdown renders live between heartbeats (view concern —
  // the heartbeat itself lives in the provider).
  const [now, setNow] = useState<number>(() => Date.now());
  useEffect(() => {
    if (!joined) return;
    setNow(Date.now());
    const ticker = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(ticker);
  }, [joined]);

  // Poll real device metrics (thermal / battery) for the DeviceBanner. The TurboModule exposes
  // getDeviceMetrics(); gate on native availability so the iOS scaffold doesn't reject every tick.
  const [metrics, setMetrics] = useState<DeviceMetrics | null>(null);
  useEffect(() => {
    if (!nativeAvailable) return;
    const poll = startDeviceMetricsPoll({ onMetrics: setMetrics });
    return () => poll.stop();
  }, [nativeAvailable]);

  // Recent persisted contributions — refreshed on focus and after each completed round.
  const [recent, setRecent] = useState<ContributionEntry[]>([]);
  const loadRecent = useCallback(async () => {
    setRecent(await contributionLedger.list(RECENT_LEDGER_ROWS));
  }, []);
  useFocusEffect(
    useCallback(() => {
      void loadRecent();
    }, [loadRecent]),
  );
  useEffect(() => {
    void loadRecent();
  }, [session.rounds, loadRecent]);

  const badge: { label: string; variant: StatusVariant } = joining
    ? { label: 'Joining…', variant: 'running' }
    : {
        notJoined: { label: 'Idle', variant: 'idle' as StatusVariant },
        joined: { label: 'Ready', variant: 'success' as StatusVariant },
        training: { label: 'Training', variant: 'running' as StatusVariant },
        error: { label: 'Error', variant: 'danger' as StatusVariant },
      }[machine];

  // Native core absent (iOS scaffold): disable the training entry point entirely and explain why,
  // rather than letting Join/Start call into a native module that isn't there. All hooks above
  // still run unconditionally, so this early return respects the Rules of Hooks.
  if (!nativeAvailable) {
    const heading =
      Platform.OS === 'ios' ? 'iOS training preview unavailable' : 'On-device training unavailable';
    return (
      <ScrollView className="flex-1 bg-canvas" style={{ paddingTop: insets.top }}>
        <View className="px-4 pt-4 pb-2 flex-row items-center justify-between">
          <Text className="text-h2 font-sans text-fg">Home</Text>
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
        <Text className="text-h2 font-sans text-fg">Home</Text>
        <StatusBadge label={badge.label} variant={badge.variant} />
      </View>

      <DeviceBanner metrics={metrics} />

      {/* NOW card — driven by the shared state machine. */}
      {!joined ? (
        projectId ? (
          <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
            <Text className="text-label font-sans font-semibold text-fg mb-1">Join a training run</Text>
            <Text className="text-caption font-sans text-fg-muted mb-3">Your data stays on this device — only learning updates are shared.</Text>
            <Text className="text-label font-sans text-fg-muted">Selected project</Text>
            <Text className="mt-1 text-body font-sans text-fg">{projectName ?? projectId}</Text>
            {projectName ? (
              <Text className="mt-0.5 text-caption font-mono text-fg-subtle">{projectId}</Text>
            ) : null}
            <Pressable
              accessibilityRole="link"
              accessibilityLabel="Change project"
              className="mt-3 py-2 self-start active:opacity-80"
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
              onPress={() => navigation.navigate('Projects')}>
              <Text className="text-body font-sans text-accent">Change project</Text>
            </Pressable>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Join run"
              accessibilityState={{ disabled: joining }}
              className={`mt-4 flex-row items-center justify-center bg-accent rounded-md py-3 active:opacity-80 ${
                joining ? 'opacity-50' : ''
              }`}
              disabled={joining}
              onPress={() => {
                void join(projectId, projectName);
              }}>
              {joining ? (
                <ActivityIndicator color={colors['accent-fg']} />
              ) : (
                <Text className="text-accent-fg text-label font-sans">Join run</Text>
              )}
            </Pressable>
          </View>
        ) : (
          <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
            <Text className="text-label font-sans font-semibold text-fg mb-1">This phone is idle</Text>
            <Text className="text-caption font-sans text-fg-muted mb-3">
              Pick a project to train on this device. Your data stays here — only learning updates are
              shared.
            </Text>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Choose a project"
              className="flex-row items-center justify-center bg-accent rounded-md py-3 active:opacity-80"
              onPress={() => navigation.navigate('Projects')}>
              <Text className="text-accent-fg text-label font-sans">Choose a project</Text>
            </Pressable>
          </View>
        )
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

          {/* MO-3: live server progress — the run's current round + deadline countdown + how many
              clients have reported this round, refreshed on the provider's heartbeat. */}
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

          {/* One action slot: Start when idle; while training it becomes the (danger) Stop. */}
          {machine === 'training' ? (
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
                    {/* "What left my phone": RoundResult carried these on every round; they were
                        never rendered before. */}
                    <View className="mt-1 flex-row justify-between">
                      <Text className="text-caption font-sans text-fg-muted">Bytes up</Text>
                      <Text className="text-caption font-mono text-fg">{formatBytes(latestRound.uplinkBytes)}</Text>
                    </View>
                    <View className="mt-1 flex-row justify-between">
                      <Text className="text-caption font-sans text-fg-muted">Bytes down</Text>
                      <Text className="text-caption font-mono text-fg">{formatBytes(latestRound.downlinkBytes)}</Text>
                    </View>
                    <View className="mt-1 flex-row justify-between">
                      <Text className="text-caption font-sans text-fg-muted">Compute time</Text>
                      <Text className="text-caption font-mono text-fg">{formatDurationMs(latestRound.computeMs)}</Text>
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
                  void stopTraining();
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
                void startTraining();
              }}>
              <Play color={colors['accent-fg']} size={18} strokeWidth={1.5} />
              <Text className="text-accent-fg text-label font-sans ml-2">Start training</Text>
            </Pressable>
          )}
        </>
      )}

      {/* THIS SESSION — fold over the rounds completed since Start. */}
      {session.rounds > 0 ? (
        <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
          <Text className="text-label font-sans font-semibold text-fg mb-2">This session</Text>
          <View className="flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">Rounds completed</Text>
            <Text className="text-caption font-mono text-fg">{session.rounds}</Text>
          </View>
          <View className="mt-1 flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">Scalars uploaded</Text>
            <Text className="text-caption font-mono text-fg">{session.scalarsTransmitted}</Text>
          </View>
          <View className="mt-1 flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">Bytes up</Text>
            <Text className="text-caption font-mono text-fg">{formatBytes(session.bytesUp)}</Text>
          </View>
          <View className="mt-1 flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">Bytes down</Text>
            <Text className="text-caption font-mono text-fg">{formatBytes(session.bytesDown)}</Text>
          </View>
          <View className="mt-1 flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">Compute time</Text>
            <Text className="text-caption font-mono text-fg">{formatDurationMs(session.computeMs)}</Text>
          </View>
        </View>
      ) : null}

      {/* Recent contributions — persisted device-local ledger (client-side facts: "submitted",
          never "accepted"; server-side acceptance needs a backend API that doesn't exist yet). */}
      {recent.length > 0 ? (
        <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
          <Text className="text-label font-sans font-semibold text-fg mb-2">Recent contributions</Text>
          {recent.map((e, i) => (
            <View key={`${e.projectId}-${e.at}-${i}`} className={i > 0 ? 'mt-2 pt-2 border-t border-hairline' : ''}>
              <View className="flex-row justify-between">
                <Text className="text-caption font-sans text-fg" numberOfLines={1}>
                  {e.projectName}
                </Text>
                <Text className="text-caption font-mono text-fg">round {e.round}</Text>
              </View>
              <View className="mt-0.5 flex-row justify-between">
                <Text className="text-caption font-sans text-fg-muted">
                  {new Date(e.at).toLocaleString()}
                </Text>
                <Text className="text-caption font-mono text-fg-muted">
                  {formatBytes(e.bytesUp)} up · {formatDurationMs(e.wallClockMs)}
                </Text>
              </View>
            </View>
          ))}
        </View>
      ) : null}

      {/* Live activity log (server round output over STOMP /topic/logs — the provider owns the
          subscription; this is a read-only view of the shared ring). */}
      {joined ? (
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
      ) : null}

      {error ? <ErrorBanner message={error} className="mx-4 my-3" /> : null}
      <View className="h-8" />
    </ScrollView>
  );
}
