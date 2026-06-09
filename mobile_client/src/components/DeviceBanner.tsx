import React from 'react';
import { Text, View } from 'react-native';
import { AlertTriangle, BatteryLow } from 'lucide-react-native';

import type { DeviceMetrics } from '../lib/nativeCore';
import { useThemeTokens } from '../theme/useThemeTokens';

// Surfaces a throttling/draining phone so the round loop can pause (15-LLD §10 bound 3).
export function DeviceBanner({ metrics }: { metrics: DeviceMetrics | null }) {
  const { colors } = useThemeTokens();
  if (!metrics) return null;
  const hot = metrics.thermalState === 'SERIOUS' || metrics.thermalState === 'CRITICAL';
  const lowBattery = metrics.batteryLevel >= 0 && metrics.batteryLevel < 0.15 && !metrics.batteryCharging;
  if (!hot && !lowBattery) return null;

  const message = hot
    ? `Device thermal state: ${metrics.thermalState} — training paused to cool down`
    : 'Battery low — connect a charger to keep training';
  const Icon = hot ? AlertTriangle : BatteryLow;

  return (
    <View className="mx-4 my-2 p-3 rounded-md bg-warning flex-row items-center">
      <Icon color={colors['accent-fg']} size={18} strokeWidth={1.5} />
      <Text className="text-accent-fg text-label font-sans ml-2 flex-1">{message}</Text>
    </View>
  );
}
