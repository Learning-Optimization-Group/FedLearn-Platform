import type { RuntimeCompatibility } from './nativeCore';

// Bump when the JavaScript/native execution surface changes incompatibly. This is deliberately
// separate from the FL server protocol, which is negotiated during registration.
export const BRIDGE_ABI_VERSION = 3;
export const SERVER_PROTOCOL_VERSION = 2;

export class NativeCompatibilityError extends Error {
  constructor() {
    super('The installed native training core does not match this app. Rebuild and reinstall the app before training.');
    this.name = 'NativeCompatibilityError';
  }
}

export async function assertNativeCompatibility(core: {
  getRuntimeCompatibility?: () => Promise<RuntimeCompatibility>;
}): Promise<void> {
  if (typeof core.getRuntimeCompatibility !== 'function') throw new NativeCompatibilityError();
  try {
    const versions = await core.getRuntimeCompatibility();
    if (versions?.bridgeAbiVersion !== BRIDGE_ABI_VERSION ||
        versions?.protocolVersion !== SERVER_PROTOCOL_VERSION) {
      throw new NativeCompatibilityError();
    }
  } catch {
    throw new NativeCompatibilityError();
  }
}
