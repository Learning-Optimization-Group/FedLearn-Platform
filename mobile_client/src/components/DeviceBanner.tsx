import React from 'react';
import { Text, View } from 'react-native';
import type { DeviceMetrics } from '../lib/nativeCore';

// Surfaces a throttling/draining phone so the round loop can pause (15-LLD §10 bound 3).
export function DeviceBanner({ metrics }: { metrics: DeviceMetrics | null }) {
  if (!metrics) return null;
  const hot = metrics.thermalState === 'SERIOUS' || metrics.thermalState === 'CRITICAL';
  const lowBattery = metrics.batteryLevel >= 0 && metrics.batteryLevel < 0.15 && !metrics.batteryCharging;
  if (!hot && !lowBattery) return null;

  const message = hot
    ? `Device thermal state: ${metrics.thermalState} — training paused to cool down`
    : 'Battery low — connect a charger to keep training';

  return (
    <View className="mx-4 my-2 p-3 rounded-xl bg-warning">
      <Text className="text-foreground text-sm font-medium">{message}</Text>
    </View>
  );
}
