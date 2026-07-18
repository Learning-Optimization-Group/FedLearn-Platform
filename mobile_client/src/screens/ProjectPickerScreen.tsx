// PROJECTS — the tab root, in two segments:
//   · Joined   — projects this account belongs to, each row showing what this phone has
//                actually contributed (device-local ledger fold); tap → ProjectDetail.
//   · Discover — everything else from the projects fetch, with device-eligibility markers.
//                PUBLIC rows get a one-tap Join affordance that ROUTES THROUGH ProjectDetail:
//                the join executes there, behind the privacy label (the single interstitial) —
//                this screen never joins directly anymore.
import React, { useCallback, useState } from 'react';
import { View, Text, Pressable, FlatList, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { Compass, FolderOpen } from 'lucide-react-native';

import { collectDeviceCapabilities } from '../lib/deviceClass';
import { eligibilitySummary } from '../lib/evaluateEligibility';
import { annotateEligibility, listProjects } from '../lib/projectsApi';
import { contributionLedger } from '../lib/contributionLedger';
import {
  canOneTapJoin,
  foldLedgerByProject,
  formatLastContribution,
  splitProjects,
  type AnnotatedProject,
  type ProjectLedgerMap,
} from '../lib/projectSegments';
import { useThemeTokens } from '../theme/useThemeTokens';
import type { MainTabScreenProps } from '../navigation/types';

type Segment = 'joined' | 'discover';

const SEGMENTS: ReadonlyArray<{ key: Segment; label: string }> = [
  { key: 'joined', label: 'Joined' },
  { key: 'discover', label: 'Discover' },
];

export function ProjectPickerScreen() {
  const navigation = useNavigation<MainTabScreenProps<'Projects'>['navigation']>();
  const { colors } = useThemeTokens();
  const [segment, setSegment] = useState<Segment>('joined');
  const [rows, setRows] = useState<AnnotatedProject[]>([]);
  const [ledger, setLedger] = useState<ProjectLedgerMap>({});
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setBusy(true);
    try {
      const [projects, caps, entries] = await Promise.all([
        listProjects(),
        collectDeviceCapabilities(),
        contributionLedger.list(),
      ]);
      setRows(annotateEligibility(projects, caps));
      setLedger(foldLedgerByProject(entries));
    } finally {
      setBusy(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load]),
  );

  // Every path into a project goes through ProjectDetail — identity, privacy label, and the
  // join action all live there.
  const open = useCallback(
    (projectId: string) => {
      navigation.navigate('ProjectDetail', { projectId });
    },
    [navigation],
  );

  const { joined, discover } = splitProjects(rows);
  const data = segment === 'joined' ? joined : discover;

  return (
    // Top/side safe areas only — the bottom inset belongs to the tab bar.
    <SafeAreaView edges={['top', 'left', 'right']} className="flex-1 bg-canvas">
      <View className="px-4 pt-4 pb-2">
        <Text className="text-h2 font-sans text-fg">Projects</Text>
      </View>

      {/* Segmented control (same accessible tablist pattern as the Playground mode switch). */}
      <View className="mx-4 mb-2 flex-row rounded-pill bg-surface-2 p-1" accessibilityRole="tablist">
        {SEGMENTS.map((s) => (
          <Pressable
            key={s.key}
            accessibilityRole="tab"
            accessibilityLabel={s.label}
            accessibilityState={{ selected: segment === s.key }}
            className={`flex-1 items-center py-2 rounded-pill active:opacity-80 ${
              segment === s.key ? 'bg-accent' : ''
            }`}
            onPress={() => setSegment(s.key)}>
            <Text
              className={`text-label font-sans ${
                segment === s.key ? 'text-accent-fg' : 'text-fg-muted'
              }`}>
              {s.label}
            </Text>
          </Pressable>
        ))}
      </View>

      {busy && rows.length === 0 ? (
        <View className="flex-1 items-center justify-center">
          <ActivityIndicator color={colors.accent} />
        </View>
      ) : (
        <FlatList
          contentContainerClassName="px-4 pb-8"
          data={data}
          keyExtractor={(r) => r.project.projectId}
          ListEmptyComponent={
            segment === 'joined' ? (
              <View className="items-center mt-24">
                <View className="w-16 h-16 rounded-pill bg-surface-2 items-center justify-center">
                  <FolderOpen color={colors['fg-muted']} size={28} strokeWidth={1.5} />
                </View>
                <Text className="text-body-lg font-sans font-semibold text-fg mt-3">
                  No joined projects yet
                </Text>
                <Text className="text-body font-sans text-fg-muted mt-1">
                  Find one under Discover to start contributing.
                </Text>
              </View>
            ) : (
              <View className="items-center mt-24">
                <View className="w-16 h-16 rounded-pill bg-surface-2 items-center justify-center">
                  <Compass color={colors['fg-muted']} size={28} strokeWidth={1.5} />
                </View>
                <Text className="text-body-lg font-sans font-semibold text-fg mt-3">
                  Nothing to discover right now
                </Text>
                <Text className="text-body font-sans text-fg-muted mt-1">
                  Projects you can join will appear here.
                </Text>
              </View>
            )
          }
          renderItem={({ item }) => {
            if (segment === 'joined') {
              return (
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel={item.project.name}
                  className="mb-2 p-4 rounded-card bg-surface-1 border border-hairline active:opacity-80"
                  onPress={() => open(item.project.projectId)}>
                  <Text className="text-body-lg font-sans text-fg">{item.project.name}</Text>
                  <Text className="text-caption font-sans text-fg-muted mt-0.5">
                    {item.project.modelType} · {item.project.status}
                  </Text>
                  <Text className="text-caption font-sans text-fg-subtle mt-1">
                    {formatLastContribution(ledger[item.project.projectId])}
                  </Text>
                </Pressable>
              );
            }
            const s = eligibilitySummary(item.result);
            return (
              <Pressable
                accessibilityRole="button"
                accessibilityLabel={`${item.project.name}${s.marker}`}
                className="mb-2 p-4 rounded-card bg-surface-1 border border-hairline active:opacity-80"
                onPress={() => open(item.project.projectId)}>
                <Text className="text-body-lg font-sans text-fg">
                  {item.project.name}
                  <Text
                    className={`text-caption font-sans ${
                      item.result.eligible ? 'text-fg-muted' : 'text-danger'
                    }`}>
                    {s.marker}
                  </Text>
                </Text>
                <Text className="text-caption font-sans text-fg-muted">
                  {item.project.modelType} · {item.project.status}
                </Text>
                {s.lines.length > 0 && (
                  <Text
                    className={`mt-1 text-caption font-sans ${
                      item.result.eligible ? 'text-warning' : 'text-danger'
                    }`}>
                    {s.lines.join(' · ')}
                  </Text>
                )}
                {canOneTapJoin(item.project) ? (
                  // One-tap Join still routes through ProjectDetail — the privacy label is the
                  // single interstitial and the join executes there, never from this list.
                  <Pressable
                    accessibilityRole="button"
                    accessibilityLabel={`Join ${item.project.name}`}
                    className="mt-3 self-start px-4 py-2 rounded-md border border-accent active:opacity-80"
                    onPress={() => open(item.project.projectId)}>
                    <Text className="text-label font-sans text-accent">Join</Text>
                  </Pressable>
                ) : null}
              </Pressable>
            );
          }}
        />
      )}
    </SafeAreaView>
  );
}
