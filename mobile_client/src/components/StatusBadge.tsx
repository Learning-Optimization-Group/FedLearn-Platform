import React from 'react';
import { Text, View } from 'react-native';

export type StatusVariant = 'idle' | 'running' | 'success' | 'warning' | 'danger';

// Single StatusPill semantics (matches web/desktop): running→accent (NOT a separate blue/green),
// completed/success→success, error/danger→danger, pending/warning→warning, idle/stopped→fg-muted.
const bg: Record<StatusVariant, string> = {
  idle: 'bg-fg-muted',
  running: 'bg-accent',
  success: 'bg-success',
  warning: 'bg-warning',
  danger: 'bg-danger',
};
// All variants are saturated fills, so the on-accent foreground reads against every one
// (white-on-color in light, dark-on-color in dark).
const fg: Record<StatusVariant, string> = {
  idle: 'text-accent-fg',
  running: 'text-accent-fg',
  success: 'text-accent-fg',
  warning: 'text-accent-fg',
  danger: 'text-accent-fg',
};

export function StatusBadge({ label, variant = 'idle' }: { label: string; variant?: StatusVariant }) {
  return (
    <View className={`px-3 py-1 rounded-pill ${bg[variant]}`}>
      <Text className={`text-caption font-sans ${fg[variant]}`}>{label}</Text>
    </View>
  );
}
