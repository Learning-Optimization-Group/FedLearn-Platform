import React, { useCallback, useState } from 'react';
import { FlatList, RefreshControl, Text, View } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { Boxes } from 'lucide-react-native';

import { listModels, type SavedModel } from '../lib/modelStore';

export function ModelLibraryScreen() {
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
    <View className="flex-1 bg-background">
      <View className="px-4 pt-4 pb-2">
        <Text className="text-2xl font-extrabold text-foreground">Model Library</Text>
        <Text className="text-xs text-muted mt-1">On-device snapshots, encrypted at rest</Text>
      </View>
      <FlatList
        data={models}
        keyExtractor={(m) => m.path}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={load} />}
        contentContainerClassName="px-4 pb-8"
        ListEmptyComponent={
          <View className="items-center mt-24">
            <Boxes color="oklch(0.55 0.02 250)" size={40} />
            <Text className="text-muted mt-3">No models yet — finish a training run to save one.</Text>
          </View>
        }
        renderItem={({ item }) => (
          <View className="p-4 mb-2 rounded-2xl bg-surface border border-border">
            <Text className="text-foreground font-semibold">{item.name}</Text>
            <Text className="text-xs text-muted mt-1">
              tier {item.tier} · round {item.round} · saved {new Date(item.savedAt).toLocaleString()}
            </Text>
            <Text className="text-[10px] text-muted mt-1" numberOfLines={1}>
              sha256 {item.sha256}
            </Text>
          </View>
        )}
      />
    </View>
  );
}
