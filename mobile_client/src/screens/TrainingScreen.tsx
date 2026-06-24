import React, { useCallback, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, Text, View } from 'react-native';
import { useRoute, useNavigation } from '@react-navigation/native';

import { joinRun, type JoinedRun } from '../lib/runJoin';
import { StatusBadge, type StatusVariant } from '../components/StatusBadge';
import { DeviceBanner } from '../components/DeviceBanner';
import { useThemeTokens } from '../theme/useThemeTokens';

type Phase = 'idle' | 'joining' | 'ready' | 'error';

export function TrainingScreen() {
  const { colors } = useThemeTokens();
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [joined, setJoined] = useState<JoinedRun | null>(null);

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
            Model download and on-device training arrive in the next update.
          </Text>
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
