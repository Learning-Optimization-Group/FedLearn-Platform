import React from 'react';
import { Text, View } from 'react-native';

export function MetricTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string | number;
  hint?: string;
}) {
  return (
    <View className="flex-1 m-1 p-3 rounded-2xl bg-surface border border-border">
      <Text className="text-xs text-muted">{label}</Text>
      <Text className="text-xl font-bold text-foreground mt-1">{value}</Text>
      {hint ? <Text className="text-[10px] text-muted mt-0.5">{hint}</Text> : null}
    </View>
  );
}
