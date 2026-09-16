import { NativeModules, Platform } from 'react-native';

// Android only: a foreground service (FlForegroundService) keeps the JS round loop alive under
// Doze and samples device state for telemetry (task 16/17). iOS runs foreground-only (Apple does
// not allow arbitrary background compute), so these are no-ops there and the UI says so.
const FlService = NativeModules.FlService as { start?: () => void; stop?: () => void } | undefined;

export const foregroundService = {
  start(): void {
    if (Platform.OS === 'android') FlService?.start?.();
  },
  stop(): void {
    if (Platform.OS === 'android') FlService?.stop?.();
  },
};
