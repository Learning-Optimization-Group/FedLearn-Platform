// Device-class detection -> maximum supported on-device model tier (A6 §M-H2).
//
// The 100M tier is NEVER offered on a phone: its ~2 GB transient zeroth-order working set OOMs
// mid-tier Android, and ~100 forward passes/round means minutes/round (15-LLD §10). 100M is a
// benchmark artifact, not a deployable mobile config — demote it to the server/desktop tier.
import DeviceInfo from 'react-native-device-info';

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
