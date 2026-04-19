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
import axios, { AxiosError } from 'axios';
import log from 'electron-log';

interface AuthStore {
  encryptedJwt: string;
  expiresAt: number;
  username: string;
}

const SERVER_URL_KEY = 'serverUrl';

const AUTH_STORE_KEY = 'auth';
const JWT_EXPIRY_MS = 24 * 60 * 60 * 1000; // 24 hours — matches backend's maxAge

// Default backend URL matches the existing frontend's axiosConfig.ts pattern
const DEFAULT_API_BASE_URL = 'http://localhost:8081/api';

export class AuthService {
  private store: any;
  private apiBaseUrl: string;

  constructor() {
    this.store = new Store({
      name: 'fedlearn-auth',
    });

    // Load persisted server URL, or fall back to env var / localhost default
    const savedUrl = this.store.get(SERVER_URL_KEY) as string | undefined;
    this.apiBaseUrl = savedUrl || process.env.FEDLEARN_API_URL || DEFAULT_API_BASE_URL;
    log.info(`[AuthService] Initialized with API base URL: ${this.apiBaseUrl}`);
  }

  /**
   * Update the backend API URL and persist it for future launches.
   */
  setApiUrl(url: string): void {
    this.apiBaseUrl = url;
    this.store.set(SERVER_URL_KEY, url);
    log.info(`[AuthService] API base URL updated to: ${url}`);
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

      const response = await axios.post(
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
   * Clears the stored JWT from the encrypted store.
   */
  logout(): void {
    this.store.delete(AUTH_STORE_KEY);
    log.info('[AuthService] JWT cleared from store');
  }

  /**
   * Checks if a valid (non-expired) JWT exists in the encrypted store.
   */
  isAuthenticated(): boolean {
    try {
      const authData = this.store.get(AUTH_STORE_KEY) as AuthStore | undefined;

      if (!authData || !authData.encryptedJwt) {
        return false;
      }

      // Check expiration
      if (Date.now() > authData.expiresAt) {
        log.info('[AuthService] Stored JWT has expired');
        this.logout();
        return false;
      }

      // Verify we can decrypt it (safeStorage key may have changed)
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
      const authData = this.store.get(AUTH_STORE_KEY) as AuthStore | undefined;

      if (!authData || !authData.encryptedJwt) {
        return null;
      }

      if (Date.now() > authData.expiresAt) {
        this.logout();
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
   * Encrypts and stores the JWT token using Electron's safeStorage API.
   * safeStorage uses the OS keychain (macOS Keychain, Windows DPAPI, Linux libsecret).
   */
  private storeJwt(jwt: string, username: string): void {
    let encryptedJwt: string;

    if (safeStorage.isEncryptionAvailable()) {
      const encrypted = safeStorage.encryptString(jwt);
      encryptedJwt = encrypted.toString('base64');
      log.info('[AuthService] JWT encrypted via safeStorage (OS keychain)');
    } else {
      // Fallback: store as base64 (less secure, but functional on some Linux DEs)
      encryptedJwt = Buffer.from(jwt).toString('base64');
      log.warn('[AuthService] safeStorage unavailable — JWT stored with base64 encoding only');
    }

    const authData: AuthStore = {
      encryptedJwt,
      expiresAt: Date.now() + JWT_EXPIRY_MS,
      username,
    };

    this.store.set(AUTH_STORE_KEY, authData);
  }
}
