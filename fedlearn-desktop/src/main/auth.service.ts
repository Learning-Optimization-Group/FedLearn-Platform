// =============================================================================
// FedLearn Desktop — Auth Service
// =============================================================================
// JWT storage and backend API calls, confined to the Main Process.
// Per Section 5.2: the JWT token NEVER leaves Main Process.
// Renderer receives only { success: boolean }.
//
// Uses electron-store + safeStorage for encrypted JWT persistence.
// The backend's AuthController sets JWT in a Set-Cookie header
// (ResponseCookie with httpOnly=true). For the desktop client, we
// extract the token from the response and store it locally with
// OS-level encryption.
// =============================================================================

import Store from 'electron-store';
import { safeStorage } from 'electron';
import type { BrowserWindow } from 'electron';
import { AxiosError } from 'axios';
import log from 'electron-log';
import { http, installUnauthorizedHandler } from './http';

/** Renderer channel Main pushes to when a session goes from valid to invalid. */
export const SESSION_EXPIRED_CHANNEL = 'auth:session-expired';

interface AuthStore {
  encryptedJwt: string;
  expiresAt: number;
  username: string;
}

/** Keys persisted in the encrypted electron-store file (see constructor). */
interface AuthStoreSchema {
  serverUrl: string;
  auth: AuthStore;
  // "Save password" opt-in: a safeStorage-encrypted, base64 JSON {username,password} blob.
  savedCredentials: string;
}

/** Held only in main-process memory when OS-level encryption is unavailable. */
interface SessionMemory {
  jwt: string;
  expiresAt: number;
  username: string;
}

const SERVER_URL_KEY = 'serverUrl';

const AUTH_STORE_KEY = 'auth';
const CREDENTIALS_KEY = 'savedCredentials';
const JWT_EXPIRY_MS = 24 * 60 * 60 * 1000; // 24 hours — matches backend's maxAge

// Default backend URL matches the existing frontend's axiosConfig.ts pattern
const DEFAULT_API_BASE_URL = 'http://localhost:8081/api';

export class AuthService {
  private store: Store<AuthStoreSchema>;
  private apiBaseUrl: string;
  /**
   * Holds the JWT in process memory when {@link safeStorage} cannot
   * encrypt (typically headless Linux without a keyring). On those hosts
   * we deliberately do NOT persist to disk — the previous design wrote
   * reversible base64, which is not encryption and would leak the token
   * to anyone with read access to the userData directory.
   *
   * Trade-off: the user must re-authenticate on every app launch when
   * encryption is unavailable. That's the correct security posture.
   */
  private sessionMemory: SessionMemory | null = null;

  /**
   * The window to notify when a session goes from valid to invalid (DE-8).
   * Optional so AuthService stays constructible in isolation (unit tests,
   * or any future headless use) without a real BrowserWindow.
   */
  private readonly mainWindow: BrowserWindow | null;

  constructor(mainWindow: BrowserWindow | null = null) {
    this.mainWindow = mainWindow;

    // clearInvalidConfig recovers from unreadable state — e.g. a store file
    // written by an older build that used a different encryptionKey. Without
    // this, a SyntaxError here would propagate up through registerIpcHandlers
    // and prevent mainWindow.loadFile from running, leaving a black window.
    this.store = new Store<AuthStoreSchema>({
      name: 'fedlearn-auth',
      clearInvalidConfig: true,
    });

    // Load persisted server URL, or fall back to env var / localhost default
    const savedUrl = this.store.get(SERVER_URL_KEY) as string | undefined;
    this.apiBaseUrl = savedUrl || process.env.FEDLEARN_API_URL || DEFAULT_API_BASE_URL;
    log.info(`[AuthService] Initialized with API base URL: ${this.apiBaseUrl}`);

    // DE-8: a 401 from any authenticated backend call (inference, client
    // projects, generation, ...) means the session is no longer valid server
    // side — the shared `http` instance carries the single active handler for
    // the whole app, so every service gets this for free.
    installUnauthorizedHandler(() => this.handleSessionExpired());
  }

  /**
   * Update the backend API URL and persist it for future launches.
   */
  setApiUrl(url: string): void {
    const changed = url !== this.apiBaseUrl;
    this.apiBaseUrl = url;
    this.store.set(SERVER_URL_KEY, url);
    log.info(`[AuthService] API base URL updated to: ${url}`);
    if (changed) {
      // Security: a JWT (and the login credentials) are minted by ONE backend and must NEVER be sent to
      // a different host. Repointing the server URL — whether a legitimate switch or a compromised
      // renderer calling setServerUrl('https://attacker...') — invalidates the current session, so clear
      // it here. Otherwise the very next authenticated call would ship the Bearer token to the newly-set
      // host. handleSessionExpired() clears the store + in-memory session and signals the renderer to
      // re-auth (guarded so it only signals when there actually was a session). Startup loads apiBaseUrl
      // directly in the constructor, not via setApiUrl, so persisted sessions survive a normal relaunch.
      this.handleSessionExpired();
    }
  }

  /**
   * Returns the current backend API URL.
   */
  getApiUrl(): string {
    return this.apiBaseUrl;
  }

  /**
   * Authenticates with the backend API and stores the JWT securely.
   * The JWT is encrypted using Electron's safeStorage (OS keychain on macOS,
   * DPAPI on Windows, libsecret on Linux).
   *
   * @returns true if authentication succeeded, false otherwise
   */
  async login(username: string, password: string): Promise<boolean> {
    try {
      log.info(`[AuthService] Attempting login for user: ${username}`);

      const response = await http.post(
        `${this.apiBaseUrl}/auth/login`,
        { username, password },
        {
          headers: { 'Content-Type': 'application/json' },
          // We need to capture the Set-Cookie header to extract the JWT
          withCredentials: true,
          // Prevent axios from automatically handling cookies
          maxRedirects: 0,
          validateStatus: (status) => status < 500,
        },
      );

      if (response.status !== 200) {
        log.warn(`[AuthService] Login failed with status ${response.status}`);
        return false;
      }

      // Extract JWT — prefer the accessToken in the response body (always
      // available), fall back to parsing the Set-Cookie header.
      let jwt: string | null = null;

      // 1. Check response body (backend returns { accessToken, username, email })
      if (response.data && typeof response.data.accessToken === 'string') {
        jwt = response.data.accessToken;
        log.info('[AuthService] JWT extracted from response body');
      }

      // 2. Fallback: Set-Cookie header
      if (!jwt) {
        const setCookieHeaders = response.headers['set-cookie'];
        if (setCookieHeaders) {
          for (const cookie of setCookieHeaders) {
            const match = cookie.match(/jwtToken=([^;]+)/);
            if (match) {
              jwt = match[1];
              log.info('[AuthService] JWT extracted from Set-Cookie header');
              break;
            }
          }
        }
      }

      if (!jwt) {
        log.error('[AuthService] No JWT found in Set-Cookie response header');
        return false;
      }

      // Encrypt and store the JWT using OS-level encryption
      this.storeJwt(jwt, username);
      log.info(`[AuthService] Login successful for user: ${username}`);
      return true;
    } catch (err: unknown) {
      if (err instanceof AxiosError) {
        log.error(`[AuthService] Login request failed: ${err.message}`);
        if (err.response) {
          log.error(`[AuthService] Response status: ${err.response.status}`);
        }
      } else {
        const message = err instanceof Error ? err.message : 'Unknown error';
        log.error(`[AuthService] Login error: ${message}`);
      }
      return false;
    }
  }

  /**
   * Clears the stored JWT from both the encrypted store and the in-memory
   * fallback. Safe to call repeatedly.
   */
  logout(): void {
    this.store.delete(AUTH_STORE_KEY);
    this.sessionMemory = null;
    log.info('[AuthService] JWT cleared from store');
  }

  /**
   * Persist the login credentials for the "Save password" opt-in, encrypted with
   * {@link safeStorage} (OS keychain). When OS encryption is unavailable we refuse to
   * persist — the same posture as the JWT: never write a reversible secret to disk.
   *
   * @returns true if the credentials were stored, false if encryption was unavailable.
   */
  saveCredentials(username: string, password: string): boolean {
    if (!safeStorage.isEncryptionAvailable()) {
      log.warn('[AuthService] safeStorage unavailable — refusing to persist saved credentials.');
      this.store.delete(CREDENTIALS_KEY);
      return false;
    }
    const encrypted = safeStorage.encryptString(JSON.stringify({ username, password }));
    this.store.set(CREDENTIALS_KEY, encrypted.toString('base64'));
    log.info('[AuthService] Saved credentials encrypted via safeStorage (OS keychain)');
    return true;
  }

  /**
   * Load the saved credentials for pre-filling the login form, decrypting via
   * {@link safeStorage}. Returns null if none are stored or the blob can no longer be
   * decrypted (e.g. the OS keychain key changed) — in which case the stale blob is scrubbed.
   */
  getSavedCredentials(): { username: string; password: string } | null {
    try {
      const blob = this.store.get(CREDENTIALS_KEY) as string | undefined;
      if (!blob) return null;
      const decrypted = safeStorage.decryptString(Buffer.from(blob, 'base64'));
      const parsed = JSON.parse(decrypted) as { username?: unknown; password?: unknown };
      if (typeof parsed.username === 'string' && typeof parsed.password === 'string') {
        return { username: parsed.username, password: parsed.password };
      }
      this.store.delete(CREDENTIALS_KEY);
      return null;
    } catch {
      log.warn('[AuthService] Could not decrypt saved credentials — scrubbing the stale blob.');
      this.store.delete(CREDENTIALS_KEY);
      return null;
    }
  }

  /** Forget any saved credentials (unchecked "Save password"). */
  clearSavedCredentials(): void {
    this.store.delete(CREDENTIALS_KEY);
  }

  /**
   * Checks if a valid (non-expired) JWT is available — either from the
   * encrypted on-disk store (preferred) or from the in-memory session
   * fallback used when OS encryption is unavailable.
   */
  isAuthenticated(): boolean {
    try {
      // 1. In-memory session (used when safeStorage was unavailable at login).
      if (this.sessionMemory) {
        if (Date.now() > this.sessionMemory.expiresAt) {
          log.info('[AuthService] In-memory JWT has expired');
          this.logout();
          return false;
        }
        return this.sessionMemory.jwt.length > 0;
      }

      // 2. Encrypted on-disk store.
      const authData = this.store.get(AUTH_STORE_KEY) as AuthStore | undefined;

      if (!authData || !authData.encryptedJwt) {
        return false;
      }

      if (Date.now() > authData.expiresAt) {
        log.info('[AuthService] Stored JWT has expired');
        this.logout();
        return false;
      }

      // Verify we can decrypt it (safeStorage key may have changed).
      try {
        const decrypted = safeStorage.decryptString(Buffer.from(authData.encryptedJwt, 'base64'));
        return decrypted.length > 0;
      } catch {
        log.warn('[AuthService] Failed to decrypt stored JWT — keychain may have changed');
        this.logout();
        return false;
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[AuthService] Auth check error: ${message}`);
      return false;
    }
  }

  /**
   * Returns the Authorization header for internal use by other Main Process services.
   * This method is NEVER exposed to the Renderer.
   */
  getAuthHeader(): string | null {
    try {
      // In-memory session takes priority — if present, on-disk is empty by design.
      if (this.sessionMemory) {
        // DE-8: proactive expiry — a request is never armed with a token whose
        // expiresAt has already passed. Treat it as an expired session (clear +
        // signal the renderer to re-auth) instead of sending a doomed request
        // that would just 401 downstream.
        if (Date.now() > this.sessionMemory.expiresAt) {
          this.handleSessionExpired();
          return null;
        }
        return `Bearer ${this.sessionMemory.jwt}`;
      }

      const authData = this.store.get(AUTH_STORE_KEY) as AuthStore | undefined;

      if (!authData || !authData.encryptedJwt) {
        return null;
      }

      if (Date.now() > authData.expiresAt) {
        this.handleSessionExpired();
        return null;
      }

      const jwt = safeStorage.decryptString(Buffer.from(authData.encryptedJwt, 'base64'));
      return `Bearer ${jwt}`;
    } catch {
      log.warn('[AuthService] Failed to retrieve auth header');
      return null;
    }
  }

  /**
   * Single entry point for "the session just went from valid to invalid" —
   * reached either from the shared `http` 401 interceptor (server said no) or
   * from {@link getAuthHeader}'s proactive expiry check (we already know it's
   * stale before asking). Clears the session like an explicit logout, then
   * pushes a one-shot event so the renderer swaps the dashboard for the login
   * screen instead of leaving the user looking at an opaque per-call error.
   *
   * Guarded on {@link hasStoredSession} so a burst of 401s / expiry checks in
   * flight at once (several in-flight requests failing together) still clears
   * once and signals exactly once, not once per caller.
   */
  private handleSessionExpired(): void {
    const hadSession = this.hasStoredSession();
    this.logout();
    if (hadSession) {
      log.info('[AuthService] Session expired — clearing JWT and signalling renderer to re-auth');
      this.emitSessionExpired();
    }
  }

  /** True when either storage backend currently holds a session to clear. */
  private hasStoredSession(): boolean {
    if (this.sessionMemory) {
      return true;
    }
    const authData = this.store.get(AUTH_STORE_KEY) as AuthStore | undefined;
    return !!(authData && authData.encryptedJwt);
  }

  /** Pushes the re-auth signal to the renderer, if a window is attached. */
  private emitSessionExpired(): void {
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.webContents.send(SESSION_EXPIRED_CHANNEL);
    }
  }

  /**
   * Stores the JWT either in the OS-encrypted on-disk store (preferred) or,
   * if {@link safeStorage} cannot encrypt, in process memory only.
   *
   * The previous implementation wrote reversible base64 to disk when
   * encryption was unavailable. Base64 is not encryption — anything with
   * read access to {@code app.getPath('userData')} could lift the token
   * trivially. The new behaviour: refuse to persist, force the user to
   * re-authenticate per launch on hosts without a keyring. This is the
   * correct trade-off; persistence is a convenience, not a requirement.
   */
  private storeJwt(jwt: string, username: string): void {
    const expiresAt = Date.now() + JWT_EXPIRY_MS;

    if (safeStorage.isEncryptionAvailable()) {
      const encrypted = safeStorage.encryptString(jwt);
      const authData: AuthStore = {
        encryptedJwt: encrypted.toString('base64'),
        expiresAt,
        username,
      };
      this.store.set(AUTH_STORE_KEY, authData);
      // Clear any prior in-memory session so the on-disk path is the single
      // source of truth once encryption becomes available again.
      this.sessionMemory = null;
      log.info('[AuthService] JWT encrypted via safeStorage (OS keychain)');
      return;
    }

    // No OS encryption — fall back to in-memory only.
    log.warn(
      '[AuthService] safeStorage unavailable on this host (no OS keyring). '
        + 'JWT will be held in process memory only; user must re-authenticate '
        + 'on next launch.',
    );
    // Belt-and-suspenders: if the on-disk store somehow already holds an
    // unencrypted token from a prior install, scrub it.
    this.store.delete(AUTH_STORE_KEY);
    this.sessionMemory = { jwt, expiresAt, username };
  }
}
