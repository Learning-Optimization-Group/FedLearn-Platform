import React from 'react';
import { Text, View } from 'react-native';

import { useThemeTokens } from '../theme/useThemeTokens';
import { withAlpha } from '../theme/withAlpha';

export type StatusVariant = 'idle' | 'running' | 'success' | 'warning' | 'danger';

// Single StatusPill semantics (matches web/desktop): running→accent (NOT a separate blue/green),
// completed/success→success, error/danger→danger, pending/warning→warning, idle/stopped→fg-muted.
//
// Tinted-background pattern: the semantic color as TEXT on a low-opacity fill of the same
// color, so every variant passes AA in both schemes (no more white-on-warning).
const text: Record<StatusVariant, string> = {
  idle: 'text-fg-muted',
  running: 'text-accent',
  success: 'text-success',
  warning: 'text-warning',
  danger: 'text-danger',
};

export function StatusBadge({ label, variant = 'idle' }: { label: string; variant?: StatusVariant }) {
  const { colors } = useThemeTokens();
  const tone: Record<StatusVariant, string> = {
    idle: colors['fg-muted'],
    running: colors.accent,
    success: colors.success,
    warning: colors.warning,
    danger: colors.danger,
  };
  return (
    <View
      className="px-3 py-1 rounded-pill"
      style={{ backgroundColor: withAlpha(tone[variant], 0.12) }}>
      <Text className={`text-caption font-sans ${text[variant]}`}>{label}</Text>
    </View>
  );
}
