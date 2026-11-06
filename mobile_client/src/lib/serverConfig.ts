// Persisted backend base URL for the control plane (/api/*). Mirrors the desktop's electron-store
// server config so a device can be pointed at a demo / AWS / Tailscale backend without a rebuild.
// Stored in encrypted storage (same backing as authStore/clientId — a URL isn't a secret, but it's
// the storage the app already ships). configureApi() is (re)bound on load and on every save.
import EncryptedStorage from 'react-native-encrypted-storage';
import { configureApi } from './restClient';
import { normalizeBackendOrigin } from './backendOrigin';
import { getBackendScope } from './backendScope';
import { clearSessionIfMatches, forgetTokenForRecovery, getSession, migrateLegacyToken } from './authStore';
import { forgetSavedCredentialsForRecovery, migrateLegacySavedCredentials } from './credentialStore';
import { clearNativeCookies } from './nativeCookies';

const KEY = 'fedlearn.serverBaseUrl';
let transitions: Promise<void> = Promise.resolve();

function transition<T>(action: () => Promise<T>): Promise<T> {
  const result = transitions.then(action);
  transitions = result.then(() => undefined, () => undefined);
  return result;
}

// The committed EC2 demo backend (frontend .env.ec2demo → VITE_PROXY_TARGET). The mobile restClient
// calls absolute `/api/...` paths, so the base URL is the host root (no `/api` suffix).
// Keep this an https origin: release builds enforce TLS-only (network_security_config.xml), so a
// plaintext default would fail there anyway. For a local demo, point a DEBUG build at the LAN
// host once via the in-app Settings screen — it persists to storage; never commit a LAN IP here.
export const DEFAULT_BASE_URL = 'https://fedlearn.duckdns.org';

/** Trim, drop any trailing slash, and require an http(s) origin. Throws on invalid input. */
export function normalizeBaseUrl(raw: string): string {
  return normalizeBackendOrigin(raw);
}

export async function getServerBaseUrl(): Promise<string> {
  const active = getBackendScope();
  if (active) return active.origin;
  try {
    const v = await EncryptedStorage.getItem(KEY);
    return normalizeBaseUrl(v && v.trim() ? v : DEFAULT_BASE_URL);
  } catch {
    return DEFAULT_BASE_URL;
  }
}

/** Persist a new base URL and rebind the axios client so every subsequent call uses it. */
export async function setServerBaseUrl(url: string): Promise<string> {
  const clean = normalizeBaseUrl(url);
  return transition(async () => {
    const previous = getBackendScope();
    if (!previous) {
      const saved = await EncryptedStorage.getItem(KEY);
      const issuer = normalizeBaseUrl(saved && saved.trim() ? saved : DEFAULT_BASE_URL);
      await migrateLegacyToken(issuer);
      await migrateLegacySavedCredentials(issuer);
    }
    const session = previous ? await getSession(previous.origin) : null;
    if (!previous || previous.origin !== clean) await clearNativeCookies();
    await EncryptedStorage.setItem(KEY, clean);
    configureApi(clean);
    if (previous?.origin !== clean && session) await clearSessionIfMatches(session);
    return clean;
  });
}

/** Explicit signed-out recovery clears sign-in data before replacing an unreadable/invalid issuer. */
export async function recoverServerConfig(url: string): Promise<string> {
  const origin = normalizeBaseUrl(url);
  return transition(async () => {
    await clearNativeCookies();
    await forgetTokenForRecovery();
    await forgetSavedCredentialsForRecovery();
    await EncryptedStorage.setItem(KEY, origin);
    configureApi(origin);
    return origin;
  });
}

/** Load the persisted URL (or the default) and bind the REST client. Call once at startup. */
export async function initServerConfig(): Promise<string> {
  return transition(async () => {
    // Do not bind a legacy credential to a fallback host when its saved setting cannot be read.
    const saved = await EncryptedStorage.getItem(KEY);
    const url = normalizeBaseUrl(saved && saved.trim() ? saved : DEFAULT_BASE_URL);
    await migrateLegacyToken(url);
    await migrateLegacySavedCredentials(url);
    await clearNativeCookies();
    configureApi(url);
    return url;
  });
}
