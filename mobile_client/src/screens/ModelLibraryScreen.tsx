// MODELS — the single model hub (tab root). Lists the on-device saved-model registry and, per
// row, pushes the two model surfaces over the tabs:
//   · Test       → ModelTestingScreen (on-device native inference) for THAT snapshot;
//   · Playground → PlaygroundScreen (server-side classify/chat — its models are the server's
//                  trained projects, so the push carries no per-snapshot param).
// Storage use: the registry stores metadata only (no byte sizes), so the header reports the
// honest figure we have — the snapshot count — rather than a fabricated size.
import React, { useCallback, useState } from 'react';
import { FlatList, Pressable, RefreshControl, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { Boxes } from 'lucide-react-native';

import { listModels, type SavedModel } from '../lib/modelStore';
import { useThemeTokens } from '../theme/useThemeTokens';
import type { MainTabScreenProps } from '../navigation/types';

export function ModelLibraryScreen() {
  const navigation = useNavigation<MainTabScreenProps<'Models'>['navigation']>();
  const { colors } = useThemeTokens();
  const [models, setModels] = useState<SavedModel[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    setRefreshing(true);
    try {
      setModels(await listModels());
    } finally {
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load]),
  );

  return (
    // Top/side safe areas only — the bottom inset belongs to the tab bar.
    <SafeAreaView edges={['top', 'left', 'right']} className="flex-1 bg-canvas">
      <View className="px-4 pt-4 pb-2">
        <Text className="text-h2 font-sans text-fg">Models</Text>
        <Text className="text-caption font-sans text-fg-muted mt-1">
          {models.length > 0
            ? `${models.length} on-device snapshot${models.length === 1 ? '' : 's'} · encrypted at rest`
            : 'On-device snapshots, encrypted at rest'}
        </Text>
      </View>
      <FlatList
        data={models}
        keyExtractor={(m) => m.path}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={load}
            tintColor={colors['fg-muted']}
          />
        }
        contentContainerClassName="px-4 pb-8"
        ListEmptyComponent={
          <View className="items-center mt-24">
            <View className="w-16 h-16 rounded-pill bg-surface-2 items-center justify-center">
              <Boxes color={colors['fg-muted']} size={28} strokeWidth={1.5} />
            </View>
            <Text className="text-body-lg font-sans font-semibold text-fg mt-3">No models yet</Text>
            <Text className="text-body font-sans text-fg-muted mt-1">
              Finish a training run to save one.
            </Text>
          </View>
        }
        renderItem={({ item }) => (
          <View className="p-4 mb-2 rounded-card bg-surface-1 border border-hairline">
            <Text className="text-fg text-body font-sans">{item.name}</Text>
            {/* IDs/metrics: mono + tabular figures. */}
            <Text
              className="text-caption text-fg-muted font-mono mt-1"
              style={{ fontVariant: ['tabular-nums'] }}>
              tier {item.tier} · round {item.round} · saved {new Date(item.savedAt).toLocaleString()}
            </Text>
            <Text
              className="text-caption text-fg-subtle font-mono mt-1"
              style={{ fontVariant: ['tabular-nums'] }}
              numberOfLines={1}>
              sha256 {item.sha256}
            </Text>
            <View className="flex-row mt-3">
              <Pressable
                accessibilityRole="button"
                accessibilityLabel={`Test ${item.name}`}
                className="flex-1 items-center justify-center bg-surface-1 border border-hairline rounded-md py-2 mr-1 active:opacity-80"
                onPress={() => navigation.navigate('ModelTesting', { modelPath: item.path })}>
                <Text className="text-fg text-label font-sans">Test</Text>
              </Pressable>
              <Pressable
                accessibilityRole="button"
                accessibilityLabel={`Open the playground from ${item.name}`}
                className="flex-1 items-center justify-center bg-surface-1 border border-hairline rounded-md py-2 ml-1 active:opacity-80"
                onPress={() => navigation.navigate('Playground')}>
                <Text className="text-fg text-label font-sans">Playground</Text>
              </Pressable>
            </View>
          </View>
        )}
      />
    </SafeAreaView>
  );
}
