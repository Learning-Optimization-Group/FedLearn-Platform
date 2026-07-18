// PROJECT DETAIL — stack push over the tabs (native-stack header supplies title + back).
//
// Top-to-bottom: project identity (name, model type, visibility, status, eligibility) ·
// the PRIVACY LABEL card (static truthful copy from privacyLabel.ts, plus the live server
// endpoint for this run once joined) · run identity when joined (run id, partition) · ONE
// action decided by decideJoinAction (Join / Leave / an explanatory note) · per-project
// contribution history from the device-local ledger.
//
// The join executes HERE — the privacy label is the single interstitial. A PUBLIC project the
// device isn't a member of gets the REST membership join first, then the run join through
// TrainingContext (the single owner of the run lifecycle). Leave maps to stopTraining: it
// aborts the native path and resets the shared run to notJoined.
import React, { useCallback, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, Text, View } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';

import { collectDeviceCapabilities } from '../lib/deviceClass';
import { eligibilitySummary } from '../lib/evaluateEligibility';
import { annotateEligibility, joinProject, listProjects } from '../lib/projectsApi';
import { contributionLedger, type ContributionEntry } from '../lib/contributionLedger';
import { decideJoinAction, type JoinDecision } from '../lib/joinDecision';
import { PRIVACY_SECTIONS } from '../lib/privacyLabel';
import { isNativeCoreAvailable } from '../lib/nativeCore';
import { readError } from '../lib/errors';
import { formatBytes, formatDurationMs } from '../lib/format';
import { useTraining } from '../state/TrainingContext';
import { ErrorBanner } from '../components/ErrorBanner';
import { useThemeTokens } from '../theme/useThemeTokens';
import type { AnnotatedProject } from '../lib/projectSegments';
import type { AppStackScreenProps } from '../navigation/types';

const HISTORY_ROWS = 10;

export function ProjectDetailScreen({ route }: AppStackScreenProps<'ProjectDetail'>) {
  const { projectId } = route.params;
  const { colors } = useThemeTokens();
  const nativeAvailable = isNativeCoreAvailable();
  const { state, join, stopTraining } = useTraining();

  const [annotated, setAnnotated] = useState<AnnotatedProject | null>(null);
  const [history, setHistory] = useState<ContributionEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [joinBusy, setJoinBusy] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setBusy(true);
    try {
      const [projects, caps, entries] = await Promise.all([
        listProjects(),
        collectDeviceCapabilities(),
        contributionLedger.entriesForProject(projectId, HISTORY_ROWS),
      ]);
      const found = annotateEligibility(projects, caps).find(
        (r) => r.project.projectId === projectId,
      );
      setAnnotated(found ?? null);
      setHistory(entries);
    } finally {
      setBusy(false);
    }
  }, [projectId]);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load]),
  );

  const joinedHere = state.joined?.projectId === projectId ? state.joined : null;

  const decision: JoinDecision | null = annotated
    ? decideJoinAction({
        project: annotated.project,
        eligible: annotated.result.eligible,
        nativeAvailable,
        activeRunProjectId: state.joined?.projectId ?? null,
        activeRunProjectName: state.projectName,
      })
    : null;

  const onJoin = useCallback(async () => {
    if (!annotated || decision?.kind !== 'join') return;
    setLocalError(null);
    setJoinBusy(true);
    try {
      // Membership first (PUBLIC auto-join), then the run join through the shared provider.
      if (decision.needsMembership) await joinProject(projectId);
      await join(projectId, annotated.project.name);
      await load(); // refresh the membership flag + history
    } catch (e) {
      setLocalError(readError(e));
    } finally {
      setJoinBusy(false);
    }
  }, [annotated, decision, join, load, projectId]);

  // join() reports failures through the shared state, not a throw — surface it here while
  // there is no joined run (once joined, run-time errors belong to Home's banner).
  const errorMsg = localError ?? (!state.joined && state.error ? state.error : null);
  const joining = joinBusy || state.joining;
  const s = annotated ? eligibilitySummary(annotated.result) : null;

  return (
    <ScrollView className="flex-1 bg-canvas">
      {/* Identity */}
      {annotated ? (
        <View className="mx-4 mt-4 p-4 rounded-card bg-surface-1 border border-hairline">
          <Text className="text-h3 font-sans text-fg">{annotated.project.name}</Text>
          <View className="mt-3 flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">Model type</Text>
            <Text className="text-caption font-mono text-fg">{annotated.project.modelType}</Text>
          </View>
          <View className="mt-1 flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">Visibility</Text>
            <Text className="text-caption font-mono text-fg">
              {annotated.project.visibility ?? 'UNKNOWN'}
            </Text>
          </View>
          <View className="mt-1 flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">Status</Text>
            <Text className="text-caption font-mono text-fg">{annotated.project.status}</Text>
          </View>
          <View className="mt-1 flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">This device</Text>
            <Text
              className={`text-caption font-sans ${
                annotated.result.eligible ? 'text-fg' : 'text-danger'
              }`}>
              {s ? s.marker.replace(' — ', '') : ''}
            </Text>
          </View>
          {s && s.lines.length > 0 ? (
            <Text
              className={`mt-2 text-caption font-sans ${
                annotated.result.eligible ? 'text-warning' : 'text-danger'
              }`}>
              {s.lines.join(' · ')}
            </Text>
          ) : null}
        </View>
      ) : busy ? (
        <View className="items-center mt-8">
          <ActivityIndicator color={colors.accent} />
        </View>
      ) : (
        <ErrorBanner
          message="This project is not visible to this account anymore."
          className="mx-4 mt-4"
        />
      )}

      {/* Privacy label — always rendered: it is the interstitial the join sits behind. */}
      <View className="mx-4 mt-3 rounded-card bg-surface-1 border border-hairline overflow-hidden">
        <View className="px-4 py-3 border-b border-hairline">
          <Text className="text-label font-sans font-semibold text-fg">Privacy label</Text>
          <Text className="text-caption font-sans text-fg-muted mt-0.5">
            What training on this project can and cannot see.
          </Text>
        </View>
        {PRIVACY_SECTIONS.map((section) => (
          <View key={section.key} className="px-4 py-3 border-t border-hairline">
            <Text className="text-label font-sans font-semibold text-fg">{section.heading}</Text>
            {section.points.map((point, i) => (
              <Text key={i} className="text-caption font-sans text-fg-muted mt-1">
                {point}
              </Text>
            ))}
            {section.key === 'leaves' && joinedHere ? (
              <View className="mt-2 flex-row justify-between">
                <Text className="text-caption font-sans text-fg-muted">Training server</Text>
                <Text className="text-caption font-mono text-fg">{joinedHere.grpcEndpoint}</Text>
              </View>
            ) : null}
          </View>
        ))}
      </View>

      {/* Run identity — only while this project is the joined run. */}
      {joinedHere ? (
        <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
          <Text className="text-label font-sans font-semibold text-fg mb-2">This run</Text>
          <View className="flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">Run id</Text>
            <Text className="text-caption font-mono text-fg" numberOfLines={1}>
              {joinedHere.runId}
            </Text>
          </View>
          <View className="mt-1 flex-row justify-between">
            <Text className="text-caption font-sans text-fg-muted">Partition</Text>
            <Text className="text-caption font-mono text-fg">{joinedHere.partitionId}</Text>
          </View>
          <Text className="text-caption font-sans text-fg-subtle mt-2">
            Training controls and live progress are on Home.
          </Text>
        </View>
      ) : null}

      {/* ONE action slot, decided by the pure decision table. */}
      {decision ? (
        decision.kind === 'join' ? (
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Join training run"
            accessibilityState={{ disabled: joining }}
            className={`mx-4 mt-3 flex-row items-center justify-center bg-accent rounded-md py-3 active:opacity-80 ${
              joining ? 'opacity-50' : ''
            }`}
            disabled={joining}
            onPress={() => {
              void onJoin();
            }}>
            {joining ? (
              <ActivityIndicator color={colors['accent-fg']} />
            ) : (
              <Text className="text-accent-fg text-label font-sans">Join training run</Text>
            )}
          </Pressable>
        ) : decision.kind === 'leave' ? (
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Leave run"
            accessibilityState={{ disabled: state.stopping }}
            className={`mx-4 mt-3 flex-row items-center justify-center bg-surface-1 border border-hairline rounded-md py-3 active:opacity-80 ${
              state.stopping ? 'opacity-50' : ''
            }`}
            disabled={state.stopping}
            onPress={() => {
              void stopTraining();
            }}>
            {state.stopping ? (
              <ActivityIndicator color={colors.danger} />
            ) : (
              <Text className="text-danger text-label font-sans">Leave run</Text>
            )}
          </Pressable>
        ) : (
          <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
            <Text className="text-caption font-sans text-fg-muted">
              {decision.kind === 'busyElsewhere'
                ? `This phone is already joined to ${decision.otherProjectName}. Leave that run before joining this project.`
                : decision.kind === 'unavailable'
                  ? 'On-device training is not available in this build, so this project cannot be joined here.'
                  : decision.kind === 'ineligible'
                    ? 'This device does not meet the project requirements listed above.'
                    : 'This project needs owner approval. Request access from the web dashboard, then come back here.'}
            </Text>
          </View>
        )
      ) : null}

      {/* Per-project contribution history (device-local ledger; "submitted", never "accepted"). */}
      <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
        <Text className="text-label font-sans font-semibold text-fg mb-2">
          Contributions from this device
        </Text>
        {history.length === 0 ? (
          <Text className="text-caption font-sans text-fg-subtle">
            No contributions from this device yet.
          </Text>
        ) : (
          history.map((e, i) => (
            <View
              key={`${e.at}-${i}`}
              className={i > 0 ? 'mt-2 pt-2 border-t border-hairline' : ''}>
              <View className="flex-row justify-between">
                <Text className="text-caption font-sans text-fg-muted">
                  {new Date(e.at).toLocaleString()}
                </Text>
                <Text className="text-caption font-mono text-fg">round {e.round}</Text>
              </View>
              <View className="mt-0.5 flex-row justify-between">
                <Text className="text-caption font-sans text-fg-subtle">Submitted</Text>
                <Text className="text-caption font-mono text-fg-muted">
                  {formatBytes(e.bytesUp)} up · {formatDurationMs(e.wallClockMs)}
                </Text>
              </View>
            </View>
          ))
        )}
      </View>

      {errorMsg ? <ErrorBanner message={errorMsg} className="mx-4 my-3" /> : null}
      <View className="h-8" />
    </ScrollView>
  );
}
