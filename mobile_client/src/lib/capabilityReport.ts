import { Platform } from 'react-native';
import DeviceInfo from 'react-native-device-info';

import nativeCore, { type DeviceMetrics, type RuntimeCompatibility } from './nativeCore';

// Stage 3 D1: what this device reports about itself when it enrolls in a run, and in its shared diagnostics.
// Informational only — static facts never approve training; the contract gate and the qualification probe decide.
// Every fact is best-effort (left out when it cannot be read, never guessed) and kept within the bounds the backend
// accepts (dto/CapabilityReport.java), so a strange device value can never get an enrollment refused.

export interface CapabilityReport {
  platform: 'android' | 'ios';
  osVersion?: string;
  apiLevel?: number;
  abis?: string[];
  totalRamBytes?: number;
  freeStorageBytes?: number;
  deviceModel?: string;
  appVersion?: string;
  appBuild?: string;
  bridgeAbiVersion?: number;
  protocolVersion?: number;
  thermalState?: string;
  batteryPct?: number;
}

/** Where each fact comes from; injectable so the collection is testable without a device. */
export interface CapabilitySources {
  platform: 'android' | 'ios';
  apiLevel: () => Promise<number>;
  systemVersion: () => string;
  abis: () => Promise<string[]>;
  totalMemory: () => Promise<number>;
  freeDiskStorage: () => Promise<number>;
  model: () => string;
  version: () => string;
  buildNumber: () => string;
  runtimeCompatibility: () => Promise<RuntimeCompatibility>;
  deviceMetrics: () => Promise<DeviceMetrics>;
}

const deviceSources: CapabilitySources = {
  platform: Platform.OS === 'ios' ? 'ios' : 'android',
  apiLevel: () => DeviceInfo.getApiLevel(),
  systemVersion: () => DeviceInfo.getSystemVersion(),
  abis: () => DeviceInfo.supportedAbis(),
  totalMemory: () => DeviceInfo.getTotalMemory(),
  freeDiskStorage: () => DeviceInfo.getFreeDiskStorage(),
  model: () => DeviceInfo.getModel(),
  version: () => DeviceInfo.getVersion(),
  buildNumber: () => DeviceInfo.getBuildNumber(),
  runtimeCompatibility: () => nativeCore.getRuntimeCompatibility(),
  deviceMetrics: () => nativeCore.getDeviceMetrics(),
};

async function attempt<T>(read: () => Promise<T> | T): Promise<T | undefined> {
  try {
    return await read();
  } catch {
    return undefined;
  }
}

const text = (value: string | undefined, max: number) =>
  typeof value === 'string' && value.length > 0 ? value.slice(0, max) : undefined;
const count = (value: number | undefined) =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0 ? Math.floor(value) : undefined;

export async function collectCapabilityReport(src: CapabilitySources = deviceSources): Promise<CapabilityReport> {
  const [apiLevel, osVersion, abis, totalRam, freeStorage, model, version, build, compat, metrics] = await Promise.all([
    src.platform === 'android' ? attempt(src.apiLevel) : Promise.resolve(undefined),
    attempt(src.systemVersion), attempt(src.abis), attempt(src.totalMemory), attempt(src.freeDiskStorage),
    attempt(src.model), attempt(src.version), attempt(src.buildNumber), attempt(src.runtimeCompatibility),
    attempt(src.deviceMetrics),
  ]);
  const battery = metrics && metrics.batteryLevel >= 0 ? Math.round(Math.min(metrics.batteryLevel, 1) * 100) : undefined;
  const report: CapabilityReport = {
    platform: src.platform,
    osVersion: text(osVersion, 32),
    apiLevel: count(apiLevel),
    abis: abis?.filter((a) => typeof a === 'string').slice(0, 8).map((a) => a.slice(0, 32)),
    totalRamBytes: count(totalRam),
    freeStorageBytes: count(freeStorage),
    deviceModel: text(model, 64),
    appVersion: text(version, 32),
    appBuild: text(build, 32),
    bridgeAbiVersion: count(compat?.bridgeAbiVersion),
    protocolVersion: count(compat?.protocolVersion),
    // UNKNOWN means the platform has not sampled yet: left out, like any fact that could not be read.
    thermalState: metrics?.thermalState === 'UNKNOWN' ? undefined : text(metrics?.thermalState, 32),
    batteryPct: battery,
  };
  return Object.fromEntries(Object.entries(report).filter(([, v]) => v !== undefined)) as CapabilityReport;
}

/** The report as diagnostics text: one fact per line. */
export function formatCapabilityReport(report: CapabilityReport): string {
  return Object.entries(report)
    .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(', ') : String(value)}`)
    .join('\n');
}

/** The shared diagnostics: the journal, then this device's report (or a note that it could not be read). */
export function diagnosticsWithReport(journal: string, report: CapabilityReport | undefined): string {
  return `${journal}\n\nThis device\n${report ? formatCapabilityReport(report) : '(the device report could not be read)'}`;
}
