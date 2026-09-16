// =============================================================================
// FedLearn Desktop — SettingsSection Component
// =============================================================================
// The settings surface as a page section (supersedes the SettingsModal body):
//   - Server card: server URL with label + help, the DE-13 insecure-HTTP
//     acknowledge flow, success/error banners. Save is the one primary action.
//   - Updates card: manual "Check for updates" trigger. Update progress itself
//     renders in the shell-mounted UpdateBanner (its preload listeners have no
//     removal API, so that component must stay mounted at shell level — this
//     card only triggers the same checkForUpdates IPC).
//   - About card: app version (webpack DefinePlugin), detected platform.
//
// Ledger design system — token vars via shared classes plus sections.css; no
// inline styles. The optional onClose (dismissable hosts) is invoked shortly
// after a successful save; page-section hosts omit it and keep the banner.
// =============================================================================

import React, { useEffect, useState } from 'react';
import { AlertTriangle, Check, RefreshCw } from 'lucide-react';
import { describeDetection, type HardwareDetection } from './trainFlow';
import './sections.css';

// Injected at build time by webpack DefinePlugin (reads `version` from
// package.json); undefined under jest, hence the typeof guard.
declare const __APP_VERSION__: string;

export interface SettingsSectionProps {
  /**
   * Optional — for dismissable hosts: invoked 1.5s after a successful save.
   * As a page section, omit it: the success banner simply stays visible.
   */
  onClose?: () => void;
}

export const SettingsSection: React.FC<SettingsSectionProps> = ({ onClose }) => {
  const [serverUrl, setServerUrl] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  // DE-13: remote plaintext http:// is refused by Main unless explicitly
  // acknowledged; the acknowledgement only holds for the current URL.
  const [insecureWarning, setInsecureWarning] = useState('');
  const [allowInsecure, setAllowInsecure] = useState(false);

  const [updateCheck, setUpdateCheck] = useState<'idle' | 'requested' | 'failed'>('idle');
  const [updateCheckError, setUpdateCheckError] = useState('');

  const [detection, setDetection] = useState<HardwareDetection | null>(null);

  const version = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : 'dev';

  // Current server URL + best-effort platform info for the About card.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const result = await window.fedLearnAPI.getServerUrl();
        if (!cancelled && result.success && result.url) setServerUrl(result.url);
      } catch (err) {
        console.error('Failed to fetch server URL', err);
      }
    })();
    (async () => {
      try {
        const result = await window.fedLearnAPI.detectHardware();
        if (!cancelled && result.success && result.detection) setDetection(result.detection);
      } catch {
        // About rows fall back to "Unknown".
      }
    })();
    return () => {
      cancelled = true;
    };
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
        setSuccessMsg('Server URL updated.');
        if (onClose) setTimeout(() => onClose(), 1500);
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

  const handleCheckUpdates = async () => {
    setUpdateCheck('requested');
    setUpdateCheckError('');
    try {
      const result = await window.fedLearnAPI.checkForUpdates();
      if (!result.success) {
        setUpdateCheck('failed');
        setUpdateCheckError(result.error ?? 'Unknown error');
      }
    } catch (err) {
      setUpdateCheck('failed');
      setUpdateCheckError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  return (
    <div className="settings-section">
      <h2 className="settings-title" id="settings-section-title">Settings</h2>

      {/* ── Server ── */}
      <section className="panel settings-card" aria-labelledby="settings-server-title">
        <div className="panel-header">
          <h3 className="panel-title" id="settings-server-title">Server</h3>
        </div>
        <form className="settings-card-body" onSubmit={handleSave}>
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

          <div className="settings-actions">
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isSaving || !serverUrl.trim()}
            >
              {isSaving ? 'Saving…' : 'Save changes'}
            </button>
          </div>
        </form>
      </section>

      {/* ── Updates ── */}
      <section className="panel settings-card" aria-labelledby="settings-updates-title">
        <div className="panel-header">
          <h3 className="panel-title" id="settings-updates-title">Updates</h3>
        </div>
        <div className="settings-card-body">
          <p className="form-help">
            Updates download in the background; a notice appears at the top of the window when
            one is ready.
          </p>
          <div className="settings-actions">
            <button
              id="settings-check-updates-button"
              type="button"
              className="btn btn-secondary"
              onClick={() => { void handleCheckUpdates(); }}
            >
              <RefreshCw strokeWidth={1.5} size={16} />
              Check for updates
            </button>
          </div>
          {updateCheck === 'requested' && (
            <p className="form-help" role="status">
              Update check started — results appear in the update notice.
            </p>
          )}
          {updateCheck === 'failed' && (
            <div className="validation-error" role="alert">
              <span className="error-icon"><AlertTriangle strokeWidth={1.5} size={16} /></span>
              Update check failed: {updateCheckError}
            </div>
          )}
        </div>
      </section>

      {/* ── About ── */}
      <section className="panel settings-card" aria-labelledby="settings-about-title">
        <div className="panel-header">
          <h3 className="panel-title" id="settings-about-title">About</h3>
        </div>
        <div className="settings-card-body">
          <div className="about-row">
            <span className="about-row-label">App version</span>
            <span className="about-row-value">v{version}</span>
          </div>
          <div className="about-row">
            <span className="about-row-label">This device</span>
            <span className="about-row-value">
              {detection ? describeDetection(detection) : 'Unknown'}
            </span>
          </div>
          <p className="about-tagline">FedLearn — Train AI together. Share nothing.</p>
        </div>
      </section>
    </div>
  );
};

export default SettingsSection;
