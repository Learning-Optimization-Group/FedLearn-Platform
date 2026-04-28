import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import '../styles/Dashboard.css';
import '../styles/ClientsPage.css';

const SERVER_ROOT_URL =
    import.meta.env.VITE_SERVER_ROOT_URL || `http://${window.location.hostname}:8081`;

const SettingsPage: React.FC = () => {
    const { currentUser, logout } = useAuth();
    const [copied, setCopied] = useState<string | null>(null);

    useEffect(() => {
        if (!copied) return;
        const t = setTimeout(() => setCopied(null), 1500);
        return () => clearTimeout(t);
    }, [copied]);

    const copy = async (label: string, value: string) => {
        try {
            await navigator.clipboard.writeText(value);
            setCopied(label);
        } catch {
            /* no-op */
        }
    };

    return (
        <div className="dashboard">
            <header className="dashboard-header">
                <div>
                    <h1>Settings</h1>
                    <p className="section-subtitle">Session and orchestrator details.</p>
                </div>
            </header>

            <section className="settings-section">
                <h2 className="section-title">Account</h2>
                <div className="settings-card">
                    <div className="settings-row">
                        <span className="settings-label">Signed in as</span>
                        <span className="settings-value">
                            {currentUser?.username || '—'}
                        </span>
                    </div>
                    <div className="settings-row">
                        <span className="settings-label">Email</span>
                        <span className="settings-value">
                            <code>{currentUser?.email ?? '—'}</code>
                        </span>
                    </div>
                    <div className="settings-row">
                        <span className="settings-label">Role</span>
                        <span className="settings-value">
                            <code>{currentUser?.role ?? 'USER'}</code>
                        </span>
                    </div>
                    <div className="settings-actions">
                        <button className="btn-danger" onClick={logout}>
                            Log out
                        </button>
                    </div>
                </div>
            </section>

            <section className="settings-section">
                <h2 className="section-title">Orchestrator</h2>
                <div className="settings-card">
                    <div className="settings-row">
                        <span className="settings-label">Server URL</span>
                        <span className="settings-value">
                            <code>{SERVER_ROOT_URL}</code>
                            <button
                                className="copy-btn"
                                onClick={() => copy('server', SERVER_ROOT_URL)}
                                aria-label="Copy server URL"
                            >
                                {copied === 'server' ? 'Copied' : 'Copy'}
                            </button>
                        </span>
                    </div>
                    <div className="settings-row">
                        <span className="settings-label">Client bootstrap</span>
                        <span className="settings-value">
                            <code>python client-docker/entrypoint.py --server {SERVER_ROOT_URL}</code>
                        </span>
                    </div>
                </div>
            </section>

            <section className="settings-section">
                <h2 className="section-title">About</h2>
                <div className="settings-card">
                    <p className="settings-paragraph">
                        FedLearn-Platform orchestrates federated training across heterogeneous
                        edge devices (Jetson ARM64, Apple Silicon, x86/CUDA). For client
                        provisioning, visit the <a href="/clients">Clients</a> page.
                    </p>
                </div>
            </section>
        </div>
    );
};

export default SettingsPage;
