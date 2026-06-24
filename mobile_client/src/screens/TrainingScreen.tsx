import React, { useCallback, useRef, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, Text, View } from 'react-native';
import { Play, Square } from 'lucide-react-native';
import { useRoute, useNavigation } from '@react-navigation/native';

import nativeCore, { type DeviceMetrics, type RoundConfig, type RoundResult } from '../lib/nativeCore';
import { joinRun, type JoinedRun } from '../lib/runJoin';
import { foregroundService } from '../lib/foregroundService';
import { StatusBadge, type StatusVariant } from '../components/StatusBadge';
import { MetricTile } from '../components/MetricTile';
import { DeviceBanner } from '../components/DeviceBanner';
import { useThemeTokens } from '../theme/useThemeTokens';

type Phase = 'idle' | 'joining' | 'ready' | 'training' | 'complete' | 'error';

// Server config is AUTHORITATIVE per round (the C++ FederatedLoop fetches K/P/eta/mu); these are
// telemetry defaults + the torch-version the round is gated against (set from the run manifest).
const defaultConfig = (torchVersion: string): RoundConfig => ({
  strategy: 'DeComFL',
  learningRate: 0.001,
  mu: 0.001,
  numPerturbations: 10,
  numLocalSteps: 5,
  gradEstimateMethod: 'forward',
  seed: 42,
  torchVersion,
});

export function TrainingScreen() {
  const { colors } = useThemeTokens();
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [joined, setJoined] = useState<JoinedRun | null>(null);
  const [last, setLast] = useState<RoundResult | null>(null);
  const [metrics, setMetrics] = useState<DeviceMetrics | null>(null);
  const stopRef = useRef(false);

  // Route params from the project picker (Task 4 → navigation.navigate('Training', { projectId })).
  const route = useRoute<any>();
  const navigation = useNavigation<any>();
  const projectId: string = route.params?.projectId ?? '';
  // slice-1b: real values come from the enroll/manifest flow.
  const datasetVersionId = '';
  const modelPath = '';

  const badge: { label: string; variant: StatusVariant } = {
    idle: { label: 'Idle', variant: 'idle' as StatusVariant },
    joining: { label: 'Joining…', variant: 'running' as StatusVariant },
    ready: { label: 'Ready', variant: 'success' as StatusVariant },
    training: { label: 'Training', variant: 'running' as StatusVariant },
    complete: { label: 'Complete', variant: 'success' as StatusVariant },
    error: { label: 'Error', variant: 'danger' as StatusVariant },
  }[phase];

  const onJoin = useCallback(async () => {
    setError(null);
    setPhase('joining');
    try {
      const result = await joinRun({
        projectId,
        strategy: 'DeComFL',
        numRounds: 20,
        minClients: 2,
        datasetVersionId,
        modelPath,
      });
      setJoined(result);
      setPhase('ready');
    } catch (e) {
      setError(String(e));
      setPhase('error');
    }
  }, [projectId, datasetVersionId, modelPath]);

  const onStart = useCallback(async () => {
    if (!joined) return;
    stopRef.current = false;
    setPhase('training');
    foregroundService.start(); // Android: survive Doze for the run's lifetime (task 16)
    const config = defaultConfig(joined.manifest.torchVersion);
    try {
      for (;;) {
        if (stopRef.current) break;

        const dm = await nativeCore.getDeviceMetrics();
        setMetrics(dm);
        if (dm.thermalState === 'SERIOUS' || dm.thermalState === 'CRITICAL') {
          // Pause to cool down; the banner explains. Re-sample on the next tick.
          await new Promise((r) => setTimeout(r, 5000));
          continue;
        }

        const res = await nativeCore.runDeComFLRound(joined.runId, config);
        setLast(res);

        const status = await nativeCore.getServerStatus(joined.runId);
        if (status.serverState === 'TRAINING_COMPLETE') {
          setPhase('complete');
          break;
        }
      }
    } catch (e) {
      const msg = String(e);
      if (msg.includes('STOP:')) {
        // A clean server-initiated stop (deadline / quorum-lost / stopped), not an error.
        setPhase('complete');
      } else {
        setError(msg);
        setPhase('error');
      }
    } finally {
      await nativeCore.stop();
      foregroundService.stop();
    }
  }, [joined]);

  const onStop = useCallback(async () => {
    stopRef.current = true;
    await nativeCore.stop();
  }, []);

  return (
    <ScrollView className="flex-1 bg-canvas">
      <View className="px-4 pt-4 pb-2 flex-row items-center justify-between">
        <Text className="text-h2 font-sans text-fg">Training</Text>
        <StatusBadge label={badge.label} variant={badge.variant} />
      </View>

      <DeviceBanner metrics={metrics} />

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
        <View className="mx-4 mt-2">
          <View className="p-4 rounded-card bg-surface-1 border border-hairline">
            <Text className="text-body text-fg-muted">Run</Text>
            <Text
              className="text-label font-mono text-fg"
              style={{ fontVariant: ['tabular-nums'] }}
              numberOfLines={1}>
              {joined.runId}
            </Text>
            <Text className="text-caption text-fg-muted mt-1">
              DeComFL · model dim {joined.modelInfo.paramCount} · trainable{' '}
              {joined.modelInfo.trainableParamCount} · tier {joined.modelInfo.tier}
            </Text>
          </View>

          <View className="flex-row mt-2">
            <MetricTile label="Round" value={last?.round ?? '—'} />
            <MetricTile label="Loss" value={last ? last.loss.toFixed(4) : '—'} />
          </View>
          <View className="flex-row">
            <MetricTile
              label="Scalars sent"
              value={last?.scalarsTransmitted ?? '—'}
              hint="K×P (DeComFL wedge)"
            />
            <MetricTile
              label="Uplink"
              value={last ? `${last.uplinkBytes} B` : '—'}
              hint="vs full model in FedAvg"
            />
          </View>
          <View className="flex-row">
            <MetricTile label="Compute" value={last ? `${last.computeMs} ms` : '—'} />
            <MetricTile label="Reverted" value={last ? (last.reverted ? 'yes' : 'no') : '—'} />
          </View>

          {phase === 'training' ? (
            <Pressable
              className="flex-row items-center justify-center bg-danger rounded-md py-3 mt-2"
              onPress={onStop}>
              <Square color={colors['accent-fg']} size={18} strokeWidth={1.5} />
              <Text className="text-accent-fg text-label font-sans ml-2">Stop</Text>
            </Pressable>
          ) : (
            <Pressable
              className="flex-row items-center justify-center bg-accent rounded-md py-3 mt-2"
              onPress={onStart}>
              <Play color={colors['accent-fg']} size={18} strokeWidth={1.5} />
              <Text className="text-accent-fg text-label font-sans ml-2">
                {phase === 'complete' ? 'Run complete' : 'Start training'}
              </Text>
            </Pressable>
          )}
        </View>
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
