// Persisted backend base URL for the control plane (/api/*). Mirrors the desktop's electron-store
// server config so a device can be pointed at a demo / AWS / Tailscale backend without a rebuild.
// Stored in encrypted storage (same backing as authStore/clientId — a URL isn't a secret, but it's
// the storage the app already ships). configureApi() is (re)bound on load and on every save.
import EncryptedStorage from 'react-native-encrypted-storage';
import { configureApi } from './restClient';

const KEY = 'fedlearn.serverBaseUrl';

// The committed EC2 demo backend (frontend .env.ec2demo → VITE_PROXY_TARGET). The mobile restClient
// calls absolute `/api/...` paths, so the base URL is the host root (no `/api` suffix).
export const DEFAULT_BASE_URL = 'https://fedlearn.duckdns.org';

/** Trim, drop any trailing slash, and require an http(s) origin. Throws on invalid input. */
export function normalizeBaseUrl(raw: string): string {
  const t = (raw ?? '').trim().replace(/\/+$/, '');
  if (!/^https?:\/\/.+/i.test(t)) {
    throw new Error('Enter a full URL starting with http:// or https://');
  }
  return t;
}

export async function getServerBaseUrl(): Promise<string> {
  try {
    const v = await EncryptedStorage.getItem(KEY);
    return v && v.trim() ? v.trim() : DEFAULT_BASE_URL;
  } catch {
    return DEFAULT_BASE_URL;
  }
}

/** Persist a new base URL and rebind the axios client so every subsequent call uses it. */
export async function setServerBaseUrl(url: string): Promise<string> {
  const clean = normalizeBaseUrl(url);
  await EncryptedStorage.setItem(KEY, clean);
  configureApi(clean);
  return clean;
}

/** Load the persisted URL (or the default) and bind the REST client. Call once at startup. */
export async function initServerConfig(): Promise<string> {
  const url = await getServerBaseUrl();
  configureApi(url);
  return url;
}
