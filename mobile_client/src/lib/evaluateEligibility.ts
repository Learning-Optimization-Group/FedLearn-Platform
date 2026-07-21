// Canonical rule copied verbatim from fedlearn-desktop/src/shared/ — keep in sync.
import type {
  DeviceCapabilities,
  DeviceRequirements,
  EligibilityResult,
} from './deviceCapabilities.types';

const PHONE_OS = new Set(['android', 'ios']);

/** Canonical advisory self-gate (spec §3.5). hardFailures gate eligibility; softWarnings inform. */
export function evaluateEligibility(
  caps: DeviceCapabilities,
  req: DeviceRequirements | null | undefined,
): EligibilityResult {
  const hardFailures: string[] = [];
  const softWarnings: string[] = [];
  if (!req) return { eligible: true, hardFailures, softWarnings };

  const isPhone = PHONE_OS.has(caps.osName);

  if (req.minRamGb != null && caps.ramGb < req.minRamGb) {
    hardFailures.push(`Needs ${req.minRamGb} GB RAM, device has ${caps.ramGb.toFixed(1)} GB`);
  }
  if (req.minStorageGb != null && caps.freeStorageGb != null && caps.freeStorageGb < req.minStorageGb) {
    hardFailures.push(`Needs ${req.minStorageGb} GB free storage, device has ${caps.freeStorageGb.toFixed(1)} GB`);
  }
  if (isPhone && req.mobileSafe === false) {
    hardFailures.push('This model is not supported on phones');
  }
  if (isPhone && caps.osName === 'android' && req.minOsAndroid != null && caps.osVersion != null) {
    const api = parseInt(caps.osVersion, 10);
    if (!Number.isNaN(api) && api < req.minOsAndroid) {
      hardFailures.push(`Needs Android API ${req.minOsAndroid}+`);
    }
  }
  if (isPhone && caps.osName === 'ios' && req.minOsIos != null && caps.osVersion != null) {
    if (compareVersions(caps.osVersion, req.minOsIos) < 0) {
      hardFailures.push(`Needs iOS ${req.minOsIos}+`);
    }
  }

  if (req.minNpuTops != null && (caps.npuTops == null || caps.npuTops < req.minNpuTops)) {
    softWarnings.push(
      caps.npuTops == null
        ? 'No NPU detected — training may be slow'
        : `NPU below ${req.minNpuTops} TOPS — training may be slow`,
    );
  }
  if (req.requiresWifi === true && caps.onWifi === false) {
    softWarnings.push('Wi-Fi recommended for this project');
  }
  if (req.minBatteryPct != null && caps.batteryPct != null && caps.batteryPct < req.minBatteryPct) {
    softWarnings.push(`Battery below ${req.minBatteryPct}%`);
  }

  return { eligible: hardFailures.length === 0, hardFailures, softWarnings };
}

/**
 * Compact display helper for the picker. `marker` is a plain-text suffix meant
 * to be appended to an item's display name (no emoji — renders consistently in
 * native controls and can be styled by the consumer).
 */
export function eligibilitySummary(r: EligibilityResult): { marker: string; lines: string[] } {
  if (!r.eligible) return { marker: ' — unsupported', lines: r.hardFailures };
  if (r.softWarnings.length > 0) return { marker: ' — limited', lines: r.softWarnings };
  return { marker: ' — recommended', lines: [] };
}

/** Compare dotted version strings (e.g. "16.0"). <0 if a<b, 0 equal, >0 if a>b. */
function compareVersions(a: string, b: string): number {
  const pa = a.split('.').map((x) => parseInt(x, 10) || 0);
  const pb = b.split('.').map((x) => parseInt(x, 10) || 0);
  const n = Math.max(pa.length, pb.length);
  for (let i = 0; i < n; i++) {
    const d = (pa[i] ?? 0) - (pb[i] ?? 0);
    if (d !== 0) return d;
  }
  return 0;
}
