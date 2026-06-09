import React, { useCallback, useState } from 'react';
import { FlatList, RefreshControl, Text, View } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { Boxes } from 'lucide-react-native';

import { listModels, type SavedModel } from '../lib/modelStore';
import { useThemeTokens } from '../theme/useThemeTokens';

export function ModelLibraryScreen() {
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
    <View className="flex-1 bg-canvas">
      <View className="px-4 pt-4 pb-2">
        <Text className="text-h2 font-sans text-fg">Model Library</Text>
        <Text className="text-caption text-fg-muted mt-1">On-device snapshots, encrypted at rest</Text>
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
            <Boxes color={colors['fg-muted']} size={40} strokeWidth={1.5} />
            <Text className="text-fg-muted text-body mt-3">
              No models yet — finish a training run to save one.
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
          </View>
        )}
      />
    </View>
  );
}
