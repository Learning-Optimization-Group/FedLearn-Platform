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
    <View className="flex-1 m-1 p-3 rounded-card bg-surface-1 border border-hairline">
      <Text className="text-caption text-fg-muted">{label}</Text>
      {/* Metric value: mono + tabular figures (NativeWind has no tabular utility). */}
      <Text className="text-h4 font-mono text-fg mt-1" style={{ fontVariant: ['tabular-nums'] }}>
        {value}
      </Text>
      {hint ? <Text className="text-caption text-fg-subtle mt-0.5">{hint}</Text> : null}
    </View>
  );
}
