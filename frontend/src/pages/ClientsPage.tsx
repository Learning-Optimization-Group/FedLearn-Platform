import React, { useState, useEffect, useCallback } from 'react';
import * as api from '../services/apiServices';
import type { User, RegisterData } from '../services/apiServices';
import DiskLoader from '../components/DiskLoader';
import ConfirmDialog from '../components/ConfirmDialog';
import { createLogger } from '../lib/logger';
import '../styles/Dashboard.css';
import '../styles/ClientsPage.css';

const log = createLogger('ClientsPage');

interface CreateClientModalProps {
    onSubmit: (data: RegisterData) => Promise<void> | void;
    onCancel: () => void;
    isSubmitting: boolean;
}

const CreateClientModal: React.FC<CreateClientModalProps> = ({ onSubmit, onCancel, isSubmitting }) => {
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        onSubmit({ username, email, password });
    };

    return (
        <div
            className="modal-backdrop"
            role="dialog"
            aria-modal="true"
            aria-labelledby="create-client-title"
            onClick={(e) => {
                if (e.target === e.currentTarget) onCancel();
            }}
        >
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <h2 id="create-client-title">Add Client</h2>
                <p className="modal-subtitle">
                    Creates a client account that edge devices will use to authenticate
                    against this FedLearn server.
                </p>
                <form onSubmit={handleSubmit} className="client-form">
                    <label htmlFor="client-username">Username</label>
                    <input
                        id="client-username"
                        type="text"
                        value={username}
                        required
                        autoFocus
                        minLength={3}
                        onChange={(e) => setUsername(e.target.value)}
                        placeholder="e.g. node-edge-01"
                    />
                    <label htmlFor="client-email">Email</label>
                    <input
                        id="client-email"
                        type="email"
                        value={email}
                        required
                        onChange={(e) => setEmail(e.target.value)}
                        placeholder="e.g. edge01@fedlearn.internal"
                    />
                    <label htmlFor="client-password">Password</label>
                    <input
                        id="client-password"
                        type="password"
                        value={password}
                        required
                        minLength={8}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="At least 8 characters"
                    />
                    <div className="modal-actions">
                        <button type="button" className="btn-close" onClick={onCancel} disabled={isSubmitting}>
                            Cancel
                        </button>
                        <button
                            type="submit"
                            className="create-project-btn"
                            disabled={isSubmitting || !username || !email || password.length < 8}
                        >
                            {isSubmitting ? 'Creating…' : 'Create Client'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

const ClientsPage: React.FC = () => {
    const [users, setUsers] = useState<User[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [forbidden, setForbidden] = useState(false);
    const [success, setSuccess] = useState('');
    const [isCreateOpen, setIsCreateOpen] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);

    const loadUsers = useCallback(async () => {
        try {
            setIsLoading(true);
            const res = await api.fetchUsers();
            setUsers(Array.isArray(res.data) ? res.data : []);
            setError('');
            setForbidden(false);
        } catch (err: any) {
            // /api/users is admin-only on the backend. Distinguish "you don't
            // have permission" (403 — show a friendly empty state) from a real
            // failure (network / 5xx — show the generic error banner).
            if (err?.response?.status === 403) {
                setForbidden(true);
                setUsers([]);
                setError('');
            } else {
                log.error('fetchUsers failed', err);
                setError('Failed to fetch clients.');
            }
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        loadUsers();
    }, [loadUsers]);

    // Auto-dismiss success banner.
    useEffect(() => {
        if (!success) return;
        const t = setTimeout(() => setSuccess(''), 4000);
        return () => clearTimeout(t);
    }, [success]);

    const handleCreate = async (data: RegisterData) => {
        try {
            setIsSubmitting(true);
            await api.createUser(data);
            setIsCreateOpen(false);
            setSuccess(`Client "${data.username}" created.`);
            loadUsers();
        } catch (err: any) {
            const message =
                err?.response?.data?.message ||
                err?.response?.data ||
                'Failed to create client. The username or email may already exist.';
            setError(typeof message === 'string' ? message : 'Failed to create client.');
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleDelete = async (id: number) => {
        try {
            await api.deleteUser(id);
            setUsers((prev) => prev.filter((u) => u.id !== id));
            setSuccess('Client removed.');
        } catch (err) {
            log.error(`deleteUser failed for id ${id}`, err);
            setError('Failed to delete client.');
        } finally {
            setConfirmDeleteId(null);
        }
    };

    if (isLoading && users.length === 0) {
        return <DiskLoader message="Loading Clients…" />;
    }

    const deleteTarget = users.find((u) => u.id === confirmDeleteId);

    return (
        <div className="dashboard">
            <header className="dashboard-header">
                <div>
                    <h1>Client Management</h1>
                    <p className="section-subtitle">
                        Provision credentials for federated edge clients.
                    </p>
                </div>
                <button
                    className="create-project-btn"
                    onClick={() => setIsCreateOpen(true)}
                    aria-label="Add new client"
                >
                    + Add Client
                </button>
            </header>

            {error && (
                <div className="error-message" role="alert">
                    {error}
                    <button
                        type="button"
                        className="dismiss-btn"
                        aria-label="Dismiss error"
                        onClick={() => setError('')}
                    >
                        ×
                    </button>
                </div>
            )}
            {success && (
                <div className="success-message" role="status">
                    {success}
                </div>
            )}

            {forbidden ? (
                <div className="empty-state" role="status">
                    <p>Admin access required</p>
                    <p className="empty-sub">
                        Listing clients is restricted to admin accounts. Ask an administrator
                        to grant your account the <code>ADMIN</code> role if you need this
                        view.
                    </p>
                </div>
            ) : users.length === 0 ? (
                <div className="empty-state">
                    <p>No clients provisioned yet.</p>
                    <p className="empty-sub">
                        Add a client to hand credentials to an edge node so it can join a
                        federated round.
                    </p>
                </div>
            ) : (
                <div className="client-grid">
                    {users.map((user) => (
                        <div className="client-card" key={user.id}>
                            <div className="client-card-header">
                                <div className="client-avatar" aria-hidden>
                                    {user.username.slice(0, 2).toUpperCase()}
                                </div>
                                <div className="client-identity">
                                    <h3>{user.username}</h3>
                                    <span className="client-email">{user.email}</span>
                                </div>
                            </div>
                            <dl className="client-meta">
                                <div>
                                    <dt>Node ID</dt>
                                    <dd>
                                        <code>{user.id}</code>
                                    </dd>
                                </div>
                                {user.createdAt && (
                                    <div>
                                        <dt>Registered</dt>
                                        <dd>{new Date(user.createdAt).toLocaleDateString()}</dd>
                                    </div>
                                )}
                            </dl>
                            <div className="client-actions">
                                <button
                                    type="button"
                                    className="btn-danger"
                                    onClick={() => setConfirmDeleteId(user.id)}
                                    aria-label={`Delete client ${user.username}`}
                                >
                                    Delete
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {isCreateOpen && (
                <CreateClientModal
                    onSubmit={handleCreate}
                    onCancel={() => setIsCreateOpen(false)}
                    isSubmitting={isSubmitting}
                />
            )}

            {deleteTarget && (
                <ConfirmDialog
                    title="Delete client?"
                    message={`Permanently remove "${deleteTarget.username}"? The client will lose the ability to authenticate.`}
                    confirmLabel="Delete"
                    danger
                    onConfirm={() => handleDelete(deleteTarget.id)}
                    onCancel={() => setConfirmDeleteId(null)}
                />
            )}
        </div>
    );
};

export default ClientsPage;
