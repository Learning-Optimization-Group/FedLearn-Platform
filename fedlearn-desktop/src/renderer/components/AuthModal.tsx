// =============================================================================
// FedLearn Desktop — AuthModal Component
// =============================================================================
// Login form that calls window.fedLearnAPI.login().
// JWT is confined to Main Process — this component only receives { success }.
// =============================================================================

import React, { useState, useCallback } from 'react';

interface AuthModalProps {
  onLoginSuccess: () => void;
}

const AuthModal: React.FC<AuthModalProps> = ({ onLoginSuccess }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

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
        const result = await window.fedLearnAPI.login(username, password);

        if (result.success) {
          onLoginSuccess();
        } else {
          setError('Invalid credentials. Please try again.');
        }
      } catch {
        setError('Connection failed. Is the backend running?');
      } finally {
        setIsLoading(false);
      }
    },
    [username, password, onLoginSuccess],
  );

  return (
    <div className="auth-overlay">
      <div className="auth-modal">
        {/* Decorative background elements */}
        <div className="auth-glow auth-glow-1" />
        <div className="auth-glow auth-glow-2" />

        <div className="auth-content">
          {/* Logo */}
          <div className="auth-logo">
            <span className="auth-logo-icon">◆</span>
            <h1 className="auth-title">FedLearn</h1>
            <p className="auth-subtitle">Privacy-Preserving Federated Learning</p>
          </div>

          {/* Login Form */}
          <form className="auth-form" onSubmit={handleSubmit}>
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
                <span className="error-icon">⚠</span>
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
