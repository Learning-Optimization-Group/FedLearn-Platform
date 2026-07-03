import React, { useState, useEffect } from 'react';
import { Settings, AlertTriangle, Check } from 'lucide-react';

// Re-declare the window.fedLearnAPI interface locally for this component
// if it is not exported from App.tsx or available globally in this file's context.

interface SettingsModalProps {
  onClose: () => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({ onClose }) => {
  const [serverUrl, setServerUrl] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  // DE-13: remote plaintext http:// is refused by Main unless explicitly
  // acknowledged; the acknowledgement only holds for the current URL.
  const [insecureWarning, setInsecureWarning] = useState('');
  const [allowInsecure, setAllowInsecure] = useState(false);

  useEffect(() => {
    // Fetch the current server URL on mount
    const fetchUrl = async () => {
      try {
        const result = await window.fedLearnAPI.getServerUrl();
        if (result.success && result.url) {
          setServerUrl(result.url);
        }
      } catch (err) {
        console.error('Failed to fetch server URL', err);
      }
    };
    fetchUrl();
  }, []);

  const save = async (overrideInsecure: boolean) => {
    setIsSaving(true);
    setError('');
    setSuccessMsg('');

    try {
      const result = await window.fedLearnAPI.setServerUrl(
        serverUrl,
        overrideInsecure ? { allowInsecureHttp: true } : undefined,
      );
      if (result.success) {
        // Accepted via override — keep the plaintext warning visible.
        setInsecureWarning(result.warning ?? '');
        setSuccessMsg('Server URL updated successfully.');
        // Optionally close after a short delay
        setTimeout(() => onClose(), 1500);
      } else if (result.code === 'INSECURE_HTTP') {
        setInsecureWarning(result.error || 'This server uses unencrypted HTTP.');
      } else {
        setError(result.error || 'Failed to update server URL.');
      }
    } catch (err: any) {
      setError(err.message || 'Unknown error occurred.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    await save(allowInsecure);
  };

  const handleAllowInsecure = async () => {
    setAllowInsecure(true);
    await save(true);
  };

  return (
    <div className="auth-overlay" style={{ position: 'absolute', top: 0, left: 0, zIndex: 100 }}>
      <div className="auth-modal" style={{ width: '480px' }}>
        <div className="auth-content">
          <div className="auth-logo">
            <span className="auth-logo-icon"><Settings strokeWidth={1.5} size={24} /></span>
            <h2 className="auth-title">Settings</h2>
            <p className="auth-subtitle">Configure advanced properties</p>
          </div>

          <form className="auth-form" onSubmit={handleSave}>
            <div className="form-group">
              <label className="form-label" htmlFor="server-url">
                AWS Application Load Balancer Endpoint
              </label>
              <input
                id="server-url"
                className="form-input"
                type="text"
                value={serverUrl}
                onChange={(e) => {
                  setServerUrl(e.target.value);
                  // A different URL needs a fresh transport decision.
                  setAllowInsecure(false);
                  setInsecureWarning('');
                }}
                placeholder="https://your-aws-alb-url.com:8081"
                disabled={isSaving}
              />
            </div>

            {insecureWarning && (
              <div className="auth-warning" role="alert">
                <span className="error-icon"><AlertTriangle strokeWidth={1.5} size={16} /></span>
                <span>{insecureWarning}</span>
                {!allowInsecure && (
                  <button
                    type="button"
                    className="btn btn-sm btn-secondary"
                    onClick={handleAllowInsecure}
                    disabled={isSaving}
                  >
                    Use HTTP anyway
                  </button>
                )}
              </div>
            )}

            {error && (
              <div className="auth-error" role="alert">
                <span className="error-icon"><AlertTriangle strokeWidth={1.5} size={16} /></span>
                {error}
              </div>
            )}
            
            {successMsg && (
              <div className="validation-error" style={{ background: 'var(--surface-2)', color: 'var(--success)', borderColor: 'var(--success)' }} role="alert">
                <span className="error-icon"><Check strokeWidth={1.5} size={16} /></span>
                {successMsg}
              </div>
            )}

            <div className="action-buttons" style={{ display: 'flex', gap: '8px', marginTop: '16px' }}>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={onClose}
                disabled={isSaving}
                style={{ flex: 1 }}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={isSaving || !serverUrl.trim()}
                style={{ flex: 1 }}
              >
                {isSaving ? 'Saving...' : 'Save Configuration'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
