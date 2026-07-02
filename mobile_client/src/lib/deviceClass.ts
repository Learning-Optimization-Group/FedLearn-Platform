// Device-class detection -> maximum supported on-device model tier (A6 §M-H2).
//
// The 100M tier is NEVER offered on a phone: its ~2 GB transient zeroth-order working set OOMs
// mid-tier Android, and ~100 forward passes/round means minutes/round (15-LLD §10). 100M is a
// benchmark artifact, not a deployable mobile config — demote it to the server/desktop tier.
import { Platform } from 'react-native';
import DeviceInfo from 'react-native-device-info';
import NativeFedLearnCore from '@spec/NativeFedLearnCore';
import type { DeviceCapabilities } from './deviceCapabilities.types';

export type ModelTier = '1M' | '10M' | '100M';

export const NEVER_OFFER_ON_MOBILE: ModelTier = '100M';

export async function maxSupportedTier(): Promise<ModelTier> {
  let totalBytes = 0;
  try {
    totalBytes = await DeviceInfo.getTotalMemory();
  } catch {
    totalBytes = 0; // unknown -> conservative
  }
  const gb = totalBytes / 1024 ** 3;
  // 10M needs ~200-300 MB working set; only offer it on devices with comfortable headroom.
  if (gb >= 6) return '10M';
  return '1M';
}

export function isTierAllowed(tier: ModelTier, maxTier: ModelTier): boolean {
  if (tier === '100M') return false; // hard rule: never on mobile
  const order: ModelTier[] = ['1M', '10M', '100M'];
  return order.indexOf(tier) <= order.indexOf(maxTier);
}

/**
 * Collects this phone's capabilities for the eligibility self-gate. Each source
 * is independently fault-tolerant: a failed native call yields `undefined` for
 * that field (the eligibility rule treats undefined as "unknown" — soft, never a
 * hard failure). onWifi/npuTops have no source on mobile today → undefined.
 */
export async function collectDeviceCapabilities(): Promise<DeviceCapabilities> {
  const [totalBytes, freeBytes, metrics, androidApiLevel] = await Promise.all([
    DeviceInfo.getTotalMemory().catch(() => 0),
    DeviceInfo.getFreeDiskStorage().catch(() => undefined as number | undefined),
    NativeFedLearnCore.getDeviceMetrics().catch(() => null),
    // Android eligibility is keyed on the API LEVEL (SDK_INT), not the release string. getApiLevel()
    // is async, so fetch it here; iOS uses the (sync) release string below.
    Platform.OS === 'android'
      ? DeviceInfo.getApiLevel().catch(() => undefined as number | undefined)
      : Promise.resolve<number | undefined>(undefined),
  ]);
  const ramGb = (totalBytes ?? 0) / 1024 ** 3;
  const freeStorageGb =
    freeBytes != null && freeBytes >= 0 ? freeBytes / 1024 ** 3 : undefined;
  // Android: report the API level (e.g. "34") so evaluateEligibility's `parseInt(osVersion) < minOsAndroid`
  // compares like-for-like (getSystemVersion() returns "14", which fails against a minOsAndroid of 26/29/34
  // and would gate out every real device). iOS: keep the release string ("17.2").
  const osVersion =
    Platform.OS === 'android'
      ? androidApiLevel != null
        ? String(androidApiLevel)
        : DeviceInfo.getSystemVersion()
      : DeviceInfo.getSystemVersion();
  // The native layer returns -1.0 for an unreadable battery (DeviceState sentinel). Guard it so it
  // does not become -100% and trip the soft "battery below N%" warning; leave it undefined instead.
  const batteryPct =
    metrics != null && metrics.batteryLevel >= 0 ? Math.round(metrics.batteryLevel * 100) : undefined;
  return {
    ramGb,
    freeStorageGb,
    osName: Platform.OS as 'android' | 'ios',
    osVersion,
    batteryPct,
    npuTops: undefined,
    onWifi: undefined,
  };
}
