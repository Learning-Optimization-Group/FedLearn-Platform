// =============================================================================
// FedLearn Desktop — AuthModal Component
// =============================================================================
// Login form with server URL configuration.
// Calls window.fedLearnAPI.login() — JWT is confined to Main Process.
// Server URL is persisted via IPC so users only need to set it once.
// =============================================================================

import React, { useState, useCallback, useEffect } from 'react';
import { Network, ChevronDown, ChevronRight, Check, AlertTriangle } from 'lucide-react';

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

  const handleSaveServer = useCallback(async () => {
    if (!serverUrl.trim()) {
      setError('Please enter a server URL.');
      return;
    }

    try {
      const result = await window.fedLearnAPI.setServerUrl(serverUrl.trim());
      if (result.success) {
        setServerSaved(true);
        setError('');
        setTimeout(() => setServerSaved(false), 2000);
      } else {
        setError(result.error || 'Failed to save server URL.');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(`Failed to save server URL: ${message}`);
    }
  }, [serverUrl]);

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
        // Ensure server URL is saved before login attempt
        await window.fedLearnAPI.setServerUrl(serverUrl.trim());

        const result = await window.fedLearnAPI.login(username, password);

        if (result.success) {
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
    [username, password, serverUrl, onLoginSuccess],
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
              <input
                id="auth-password"
                className="form-input"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                autoComplete="current-password"
                disabled={isLoading}
                maxLength={256}
              />
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

          <p className="auth-footer-text">
            Secure authentication via encrypted IPC bridge
          </p>
        </div>
      </div>
    </div>
  );
};

export default AuthModal;
