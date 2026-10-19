import { NativeModules, Platform } from 'react-native';

// Android uses a foreground service for process priority and native waits for round pacing.
// React Native pauses JavaScript timers when its activity backgrounds. iOS runs foreground-only (Apple does
// not allow arbitrary background compute), so these are no-ops there and the UI says so.
const FlService = NativeModules.FlService as {
  start?: () => void;
  stop?: () => void;
  delay?: (ms: number) => Promise<void>;
} | undefined;

function assertSchedulingAvailable(): void {
  if (Platform.OS === 'android' && !FlService?.delay) {
    throw new Error('Update the Android app to support background training waits.');
  }
}

export const foregroundService = {
  assertSchedulingAvailable,
  start(): void {
    if (Platform.OS === 'android') FlService?.start?.();
  },
  stop(): void {
    if (Platform.OS === 'android') FlService?.stop?.();
  },
  delay(ms: number): Promise<void> {
    if (!Number.isSafeInteger(ms) || ms < 0 || ms > 60000) {
      return Promise.reject(new Error('Invalid training delay.'));
    }
    if (Platform.OS === 'android') {
      try {
        assertSchedulingAvailable();
        return FlService!.delay!(ms);
      } catch (error) {
        return Promise.reject(error);
      }
    }
    return new Promise(resolve => setTimeout(resolve, ms));
  },
};
