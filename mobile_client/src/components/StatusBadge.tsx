import React from 'react';
import { Text, View } from 'react-native';

export type StatusVariant = 'idle' | 'running' | 'success' | 'warning' | 'danger';

const bg: Record<StatusVariant, string> = {
  idle: 'bg-surface-muted',
  running: 'bg-primary',
  success: 'bg-success',
  warning: 'bg-warning',
  danger: 'bg-danger',
};
const fg: Record<StatusVariant, string> = {
  idle: 'text-foreground',
  running: 'text-primary-foreground',
  success: 'text-primary-foreground',
  warning: 'text-foreground',
  danger: 'text-primary-foreground',
};

export function StatusBadge({ label, variant = 'idle' }: { label: string; variant?: StatusVariant }) {
  return (
    <View className={`px-3 py-1 rounded-full ${bg[variant]}`}>
      <Text className={`text-xs font-semibold ${fg[variant]}`}>{label}</Text>
    </View>
  );
}
