import React, { useCallback, useState } from 'react';
import { View, Text, Pressable, FlatList, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { FolderOpen } from 'lucide-react-native';
import { collectDeviceCapabilities } from '../lib/deviceClass';
import { eligibilitySummary } from '../lib/evaluateEligibility';
import { annotateEligibility, joinProject, listProjects } from '../lib/projectsApi';
import type { ClientProject } from '../lib/projectsApi';
import type { EligibilityResult } from '../lib/deviceCapabilities.types';
import { useThemeTokens } from '../theme/useThemeTokens';

type AnnotatedRow = { project: ClientProject; result: EligibilityResult };

export function ProjectPickerScreen() {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const navigation = useNavigation<any>();
  const { colors } = useThemeTokens();
  const [rows, setRows] = useState<AnnotatedRow[]>([]);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setBusy(true);
    try {
      const [projects, caps] = await Promise.all([listProjects(), collectDeviceCapabilities()]);
      setRows(annotateEligibility(projects, caps));
    } finally {
      setBusy(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load]),
  );

  const onSelect = useCallback(
    async (p: ClientProject) => {
      if (p.visibility === 'PUBLIC' && p.joined === false) {
        try {
          await joinProject(p.projectId);
        } catch {
          // surfaced on next load
        }
      }
      navigation.navigate('Training', { projectId: p.projectId });
    },
    [navigation],
  );

  return (
    // Top/side safe areas only — the bottom inset belongs to the tab bar.
    <SafeAreaView edges={['top', 'left', 'right']} className="flex-1 bg-canvas">
      <View className="px-4 pt-4 pb-2">
        <Text className="text-h2 font-sans text-fg">Choose a project</Text>
      </View>
      {busy && rows.length === 0 ? (
        <View className="flex-1 items-center justify-center">
          <ActivityIndicator color={colors.accent} />
        </View>
      ) : (
        <FlatList
          contentContainerClassName="px-4 pb-8"
          data={rows}
          keyExtractor={(r) => r.project.projectId}
          ListEmptyComponent={
            <View className="items-center mt-24">
              <View className="w-16 h-16 rounded-pill bg-surface-2 items-center justify-center">
                <FolderOpen color={colors['fg-muted']} size={28} strokeWidth={1.5} />
              </View>
              <Text className="text-body-lg font-sans font-semibold text-fg mt-3">
                No projects available
              </Text>
              <Text className="text-body font-sans text-fg-muted mt-1">
                Projects you can join will appear here.
              </Text>
            </View>
          }
          renderItem={({ item }) => {
            const s = eligibilitySummary(item.result);
            return (
              <Pressable
                accessibilityRole="button"
                accessibilityLabel={`${item.project.name}${s.marker}`}
                className="mb-2 p-4 rounded-card bg-surface-1 border border-hairline active:opacity-80"
                onPress={() => { void onSelect(item.project); }}>
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
                  {item.project.joined === false ? ' · tap to join' : ''}
                </Text>
                {s.lines.length > 0 && (
                  <Text
                    className={`mt-1 text-caption font-sans ${
                      item.result.eligible ? 'text-warning' : 'text-danger'
                    }`}>
                    {s.lines.join(' · ')}
                  </Text>
                )}
              </Pressable>
            );
          }}
        />
      )}
    </SafeAreaView>
  );
}
