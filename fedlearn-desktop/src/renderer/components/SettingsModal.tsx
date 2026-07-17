// =============================================================================
// FedLearn Desktop — SettingsModal Component
// =============================================================================
// A real dialog: centered panel over a scrim, role="dialog" + aria-modal,
// Escape closes, close X in the header. Ledger design system — all styling
// comes from shared classes in styles.css and token vars; no inline styles.
// =============================================================================

import React, { useState, useEffect } from 'react';
import { AlertTriangle, Check, X } from 'lucide-react';

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

  // Escape closes the dialog.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

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
        setSuccessMsg('Server URL updated.');
        // Optionally close after a short delay
        setTimeout(() => onClose(), 1500);
      } else if (result.code === 'INSECURE_HTTP') {
        setInsecureWarning(result.error || 'This server uses unencrypted HTTP.');
      } else {
        setError(result.error || 'Failed to update server URL.');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred.');
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
    <div className="modal-overlay">
      <div
        className="modal-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-modal-title"
      >
        <div className="modal-header">
          <h2 className="modal-title" id="settings-modal-title">Settings</h2>
          <button
            type="button"
            className="modal-close"
            aria-label="Close settings"
            onClick={onClose}
          >
            <X strokeWidth={1.5} size={16} />
          </button>
        </div>

        <form className="modal-body" onSubmit={handleSave}>
          <div className="form-group">
            <label className="form-label" htmlFor="server-url">
              Server URL
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
              placeholder="https://server.example.com:8081"
              aria-describedby="server-url-help"
              disabled={isSaving}
              autoFocus
            />
            <p className="form-help" id="server-url-help">
              Address of the FedLearn server this app connects to.
            </p>
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
            <div className="validation-error" role="alert">
              <span className="error-icon"><AlertTriangle strokeWidth={1.5} size={16} /></span>
              {error}
            </div>
          )}

          {successMsg && (
            <div className="validation-success" role="status">
              <span className="error-icon"><Check strokeWidth={1.5} size={16} /></span>
              {successMsg}
            </div>
          )}

          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
              disabled={isSaving}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isSaving || !serverUrl.trim()}
            >
              {isSaving ? 'Saving…' : 'Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default SettingsModal;
