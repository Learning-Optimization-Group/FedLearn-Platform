// Shared, PURE type definitions for the device-eligibility self-gate.
// NO Electron / Node / React-Native imports — compiles in main, renderer, and (copied) mobile.

/** Mirrors the backend DeviceRequirements record. Backend serializes with
 *  @JsonInclude(NON_NULL), so any field may be absent — all optional here. */
export interface DeviceRequirements {
  minRamGb?: number;
  minStorageGb?: number;
  minOsAndroid?: number;
  minOsIos?: string;
  mobileSafe?: boolean;
  maxTrainableParams?: number;
  minNpuTops?: number;
  estimatedRoundTimeSeconds?: number;
  minBatteryPct?: number;
  requiresWifi?: boolean;
  acceleratorBackends?: string[];
}

export type DeviceOs = 'android' | 'ios' | 'macos' | 'windows' | 'linux';

/** What a client self-reports about itself. Unreadable facts are left undefined. */
export interface DeviceCapabilities {
  ramGb: number;
  freeStorageGb?: number;
  osName: DeviceOs;
  /** android: API level as a string; ios: "16.0"; desktop: kernel string (informational). */
  osVersion?: string;
  npuTops?: number;
  batteryPct?: number; // 0-100
  onWifi?: boolean;
}

export interface EligibilityResult {
  eligible: boolean;
  hardFailures: string[];
  softWarnings: string[];
}
