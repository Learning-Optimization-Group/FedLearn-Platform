import React, { useCallback, useState } from 'react';
import { View, Text, Pressable, FlatList, ActivityIndicator } from 'react-native';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { collectDeviceCapabilities } from '../lib/deviceClass';
import { eligibilitySummary } from '../lib/evaluateEligibility';
import { annotateEligibility, joinProject, listProjects } from '../lib/projectsApi';
import type { ClientProject } from '../lib/projectsApi';
import type { EligibilityResult } from '../lib/deviceCapabilities.types';

type AnnotatedRow = { project: ClientProject; result: EligibilityResult };

export function ProjectPickerScreen() {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const navigation = useNavigation<any>();
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
    <View className="flex-1 bg-canvas">
      <View className="px-4 pt-4 pb-2">
        <Text className="text-h2 font-sans text-fg">Choose a project</Text>
      </View>
      {busy && rows.length === 0 ? (
        <View className="flex-1 items-center justify-center">
          <ActivityIndicator />
        </View>
      ) : (
        <FlatList
          contentContainerClassName="px-4 pb-8"
          data={rows}
          keyExtractor={(r) => r.project.projectId}
          ListEmptyComponent={
            <View className="items-center mt-24">
              <Text className="text-body font-sans text-fg-muted">No projects available.</Text>
            </View>
          }
          renderItem={({ item }) => {
            const s = eligibilitySummary(item.result);
            return (
              <Pressable
                className="mb-2 p-4 rounded-card bg-surface-1 border border-hairline"
                onPress={() => { void onSelect(item.project); }}>
                <Text className="text-body-lg font-sans text-fg">
                  {s.marker} {item.project.name}
                </Text>
                <Text className="text-caption font-sans text-fg-muted">
                  {item.project.modelType} · {item.project.status}
                  {item.project.joined === false ? ' · tap to join' : ''}
                </Text>
                {s.marker !== '✅' && s.lines.length > 0 && (
                  <Text
                    className={`mt-1 text-caption font-sans ${
                      s.marker === '⚠️' ? 'text-danger' : 'text-fg-subtle'
                    }`}>
                    {s.lines.join(' · ')}
                  </Text>
                )}
              </Pressable>
            );
          }}
        />
      )}
    </View>
  );
}
