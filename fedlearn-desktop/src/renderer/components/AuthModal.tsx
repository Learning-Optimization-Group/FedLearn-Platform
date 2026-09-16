// =============================================================================
// FedLearn Desktop — AuthModal Component
// =============================================================================
// Login form with server URL configuration.
// Calls window.fedLearnAPI.login() — JWT is confined to Main Process.
// Server URL is persisted via IPC so users only need to set it once.
// =============================================================================

import React, { useState, useCallback, useEffect } from 'react';
import { Network, ChevronDown, ChevronRight, Check, AlertTriangle, Eye, EyeOff } from 'lucide-react';
import { isPlaintextRemoteUrl } from '../../shared/urlSecurity';

interface AuthModalProps {
  onLoginSuccess: () => void;
}

const AuthModal: React.FC<AuthModalProps> = ({ onLoginSuccess }) => {
  const [serverUrl, setServerUrl] = useState('http://localhost:8081');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showServerConfig, setShowServerConfig] = useState(false);
  const [serverSaved, setServerSaved] = useState(false);
  // DE-13: remote plaintext http:// is refused by Main unless the user
  // explicitly opts in. insecureWarning holds the refusal/override message;
  // allowInsecure remembers the acknowledgement for the current URL only.
  const [insecureWarning, setInsecureWarning] = useState('');
  const [allowInsecure, setAllowInsecure] = useState(false);
  // Show/hide password (eye toggle) and the "Save password" opt-in.
  const [showPassword, setShowPassword] = useState(false);
  const [savePassword, setSavePassword] = useState(false);

  // Load saved server URL on mount
  useEffect(() => {
    const loadUrl = async () => {
      try {
        const result = await window.fedLearnAPI.getServerUrl();
        if (result.success && result.url) {
          // Strip /api suffix for display
          const displayUrl = result.url.replace(/\/api$/, '');
          setServerUrl(displayUrl);
        }
      } catch {
        // Use default
      }
    };
    loadUrl();
  }, []);

  // Pre-fill from saved credentials ("Save password" opt-in) on mount.
  useEffect(() => {
    const loadCreds = async () => {
      try {
        const result = await window.fedLearnAPI.getSavedCredentials();
        if (result.success && result.username && result.password) {
          setUsername(result.username);
          setPassword(result.password);
          setSavePassword(true);
        }
      } catch {
        // Nothing saved — leave the form empty.
      }
    };
    loadCreds();
  }, []);

  const saveServer = useCallback(
    async (overrideInsecure: boolean): Promise<boolean> => {
      if (!serverUrl.trim()) {
        setError('Please enter a server URL.');
        return false;
      }

      try {
        const result = await window.fedLearnAPI.setServerUrl(
          serverUrl.trim(),
          overrideInsecure ? { allowInsecureHttp: true } : undefined,
        );
        if (result.success) {
          setError('');
          // Accepted via override — keep the plaintext warning visible.
          setInsecureWarning(result.warning ?? '');
          return true;
        }
        if (result.code === 'INSECURE_HTTP') {
          setShowServerConfig(true);
          setInsecureWarning(result.error || 'This server uses unencrypted HTTP.');
          return false;
        }
        setError(result.error || 'Failed to save server URL.');
        return false;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(`Failed to save server URL: ${message}`);
        return false;
      }
    },
    [serverUrl],
  );

  const handleSaveServer = useCallback(async () => {
    if (await saveServer(allowInsecure)) {
      setServerSaved(true);
      setTimeout(() => setServerSaved(false), 2000);
    }
  }, [saveServer, allowInsecure]);

  const handleAllowInsecure = useCallback(async () => {
    setAllowInsecure(true);
    if (await saveServer(true)) {
      setServerSaved(true);
      setTimeout(() => setServerSaved(false), 2000);
    }
  }, [saveServer]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      setError('');

      if (!username.trim() || !password.trim()) {
        setError('Please enter both username and password.');
        return;
      }

      setIsLoading(true);

      try {
        // Ensure server URL is saved before login attempt. If Main refuses it
        // (remote plaintext http:// without acknowledgement), do NOT proceed —
        // login would silently go to the previously stored URL.
        const urlResult = await window.fedLearnAPI.setServerUrl(
          serverUrl.trim(),
          allowInsecure ? { allowInsecureHttp: true } : undefined,
        );
        if (!urlResult.success) {
          if (urlResult.code === 'INSECURE_HTTP') {
            setShowServerConfig(true);
            setInsecureWarning(urlResult.error || 'This server uses unencrypted HTTP.');
          } else {
            setError(urlResult.error || 'Invalid server URL.');
          }
          return;
        }
        setInsecureWarning(urlResult.warning ?? '');

        const result = await window.fedLearnAPI.login(username, password);

        if (result.success) {
          // Persist or forget the password per the opt-in — only after a successful sign-in.
          if (savePassword) {
            await window.fedLearnAPI.saveCredentials(username, password);
          } else {
            await window.fedLearnAPI.clearSavedCredentials();
          }
          onLoginSuccess();
        } else {
          setError('Invalid credentials. Please try again.');
        }
      } catch (err) {
        console.error('[AuthModal] Login error:', err);
        const message = err instanceof Error ? err.message : String(err);
        setError(`Connection failed: ${message}`);
      } finally {
        setIsLoading(false);
      }
    },
    [username, password, serverUrl, allowInsecure, savePassword, onLoginSuccess],
  );

  return (
    <div className="auth-overlay">
      <div className="auth-modal">
        <div className="auth-content">
          {/* Logo */}
          <div className="auth-logo">
            <span className="auth-logo-icon">
              <Network strokeWidth={1.5} size={22} />
            </span>
            <h1 className="auth-title">Fed<span style={{ color: 'var(--accent)' }}>Learn</span></h1>
            <p className="auth-subtitle">Train AI together. Share nothing.</p>
          </div>

          {/* Login Form */}
          <form className="auth-form" onSubmit={handleSubmit}>
            {/* Server URL — collapsible */}
            <div className="form-group">
              <button
                type="button"
                className="server-toggle"
                onClick={() => setShowServerConfig(!showServerConfig)}
              >
                <span className="server-toggle-icon">
                  {showServerConfig ? (
                    <ChevronDown strokeWidth={1.5} size={16} />
                  ) : (
                    <ChevronRight strokeWidth={1.5} size={16} />
                  )}
                </span>
                <span className="server-toggle-label">Server</span>
                <span className="server-toggle-url">{serverUrl}</span>
              </button>

              {showServerConfig && (
                <div className="server-config">
                  <div className="server-input-row">
                    <input
                      id="auth-server-url"
                      className="form-input"
                      type="text"
                      value={serverUrl}
                      onChange={(e) => {
                        setServerUrl(e.target.value);
                        setServerSaved(false);
                        // A different URL needs a fresh transport decision.
                        setAllowInsecure(false);
                        setInsecureWarning('');
                      }}
                      placeholder="http://your-server:8081"
                      disabled={isLoading}
                      maxLength={512}
                    />
                    <button
                      type="button"
                      className="btn btn-sm btn-secondary"
                      onClick={handleSaveServer}
                      disabled={isLoading}
                    >
                      {serverSaved ? <Check strokeWidth={1.5} size={16} /> : 'Save'}
                    </button>
                  </div>
                  <p className="server-hint">/api is appended automatically</p>

                  {insecureWarning && (
                    <div className="auth-warning" role="alert">
                      <span className="error-icon">
                        <AlertTriangle strokeWidth={1.5} size={16} />
                      </span>
                      <span>{insecureWarning}</span>
                      {!allowInsecure && (
                        <button
                          type="button"
                          className="btn btn-sm btn-secondary"
                          onClick={handleAllowInsecure}
                          disabled={isLoading}
                        >
                          Use HTTP anyway
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="auth-username">
                Username
              </label>
              <input
                id="auth-username"
                className="form-input"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                autoComplete="username"
                disabled={isLoading}
                maxLength={256}
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="auth-password">
                Password
              </label>
              <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                <input
                  id="auth-password"
                  className="form-input"
                  style={{ paddingRight: 40, width: '100%' }}
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  autoComplete="current-password"
                  disabled={isLoading}
                  maxLength={256}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  aria-pressed={showPassword}
                  disabled={isLoading}
                  style={{
                    position: 'absolute',
                    right: 8,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: 'none',
                    border: 'none',
                    padding: 4,
                    cursor: isLoading ? 'default' : 'pointer',
                    color: 'var(--fg-muted, #6b7280)',
                  }}
                >
                  {showPassword ? <EyeOff strokeWidth={1.5} size={16} /> : <Eye strokeWidth={1.5} size={16} />}
                </button>
              </div>
            </div>

            {/* Save password — opt-in, encrypted at rest via the OS keychain (safeStorage). */}
            <div className="form-group">
              <label
                htmlFor="auth-save-password"
                style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}
              >
                <input
                  id="auth-save-password"
                  type="checkbox"
                  checked={savePassword}
                  onChange={(e) => setSavePassword(e.target.checked)}
                  disabled={isLoading}
                />
                <span className="form-label" style={{ marginBottom: 0 }}>Save password</span>
              </label>
            </div>

            {error && (
              <div className="auth-error" role="alert">
                <span className="error-icon">
                  <AlertTriangle strokeWidth={1.5} size={16} />
                </span>
                {error}
              </div>
            )}

            <button
              id="auth-submit"
              className="btn btn-primary btn-full"
              type="submit"
              disabled={isLoading}
            >
              {isLoading ? (
                <span className="btn-loading">
                  <span className="loading-spinner-sm" />
                  Authenticating...
                </span>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          {isPlaintextRemoteUrl(serverUrl) ? (
            <p className="auth-footer-text auth-footer-text--warning">
              Unencrypted connection — this server uses plaintext HTTP, so credentials are not
              protected in transit
            </p>
          ) : (
            <p className="auth-footer-text">
              Secure authentication via encrypted IPC bridge
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default AuthModal;
