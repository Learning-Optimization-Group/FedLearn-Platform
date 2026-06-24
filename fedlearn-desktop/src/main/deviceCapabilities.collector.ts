import * as os from 'os';
import * as fs from 'fs';
import * as path from 'path';
import type { DeviceCapabilities, DeviceOs } from '../shared/deviceCapabilities.types';

/** Maps Node's process/os platform to the shared DeviceOs. */
function osName(): DeviceOs {
  switch (os.platform()) {
    case 'darwin': return 'macos';
    case 'win32': return 'windows';
    default: return 'linux';
  }
}

/** Free space (GB) at the user's home volume, via stdlib fs.statfsSync. Undefined on failure. */
function freeStorageGb(): number | undefined {
  try {
    // On Windows statfs needs a drive root; elsewhere the home dir is fine.
    const target = os.platform() === 'win32' ? path.parse(os.homedir()).root : os.homedir();
    const st = fs.statfsSync(target);
    // bavail = blocks available to an unprivileged user; bsize = block size.
    return (Number(st.bavail) * Number(st.bsize)) / 1024 ** 3;
  } catch {
    return undefined;
  }
}

/**
 * Collects this desktop's device capabilities for the eligibility self-gate.
 * Node stdlib only (no Electron import) so it is unit-testable. RAM/storage/OS
 * are read; NPU TOPS, battery, and wifi are not probed on desktop (left undefined),
 * which the eligibility rule treats as "unknown" (soft, never a hard failure).
 */
export function collectDeviceCapabilities(): DeviceCapabilities {
  return {
    ramGb: os.totalmem() / 1024 ** 3,
    freeStorageGb: freeStorageGb(),
    osName: osName(),
    osVersion: os.release(),
    npuTops: undefined,
    batteryPct: undefined,
    onWifi: undefined,
  };
}
