// =============================================================================
// FedLearn Frontend — AdminDashboard (role: PLATFORM_ADMIN)
// =============================================================================
// The platform-admin home: overview metrics, user/role management, the two
// approval queues (owner-promotion + project-deletion), and an all-projects
// table. 403 from any of these means "not allowed" (shouldn't happen for an
// admin) and is rendered inline rather than logging out.

import { useState, useEffect, useCallback } from 'react';
import { AlertCircle } from 'lucide-react';
import * as api from '../../services/apiServices';
import {
    errorMessage,
    type AdminOverview,
    type AdminUser,
    type AdminProject,
    type OwnerRequest,
    type DeletionRequest,
    type Role,
} from '../../services/apiServices';
import { Card, Button, StatusPill, StatGroup, Select, ConfirmDialog, type StatusKind } from '../ui';
import { PageHeader } from './PageHeader';
import { createLogger } from '../../lib/logger';

const log = createLogger('AdminDashboard');

const ROLE_OPTIONS: Role[] = ['USER', 'PROJECT_OWNER', 'PLATFORM_ADMIN'];
const ROLE_LABEL: Record<Role, string> = {
    USER: 'User',
    PROJECT_OWNER: 'Owner',
    PLATFORM_ADMIN: 'Admin',
};

function projectStatusKind(status: string): StatusKind {
    switch (status?.toUpperCase()) {
        case 'RUNNING':
            return 'running';
        case 'COMPLETED':
            return 'completed';
        case 'FAILED':
            return 'error';
        default:
            return 'idle';
    }
}

const sectionTitle = 'text-h4 font-semibold text-fg';
// SectionLabel-styled table header cell (the one uppercase micro-label).
const th = 'px-4 py-2.5 text-left text-[11px] font-semibold uppercase tracking-[0.08em] text-fg-muted';
const td = 'px-4 py-3 text-body text-fg align-middle';

export function AdminDashboard() {
    const [overview, setOverview] = useState<AdminOverview | null>(null);
    const [users, setUsers] = useState<AdminUser[]>([]);
    const [projects, setProjects] = useState<AdminProject[]>([]);
    const [ownerRequests, setOwnerRequests] = useState<OwnerRequest[]>([]);
    const [deletionRequests, setDeletionRequests] = useState<DeletionRequest[]>([]);
    const [error, setError] = useState('');
    const [savingUserId, setSavingUserId] = useState<number | null>(null);

    // Deletion approval needs an explicit confirm (it permanently deletes).
    const [confirmDeletion, setConfirmDeletion] = useState<DeletionRequest | null>(null);

    const loadAll = useCallback(async () => {
        const [ovr, usr, prj, own, del] = await Promise.allSettled([
            api.fetchAdminOverview(),
            api.fetchAdminUsers(),
            api.fetchAdminProjects(),
            api.fetchOwnerRequests('PENDING'),
            api.fetchDeletionRequests('PENDING'),
        ]);
        if (ovr.status === 'fulfilled') setOverview(ovr.value.data);
        if (usr.status === 'fulfilled') setUsers(usr.value.data);
        if (prj.status === 'fulfilled') setProjects(prj.value.data);
        if (own.status === 'fulfilled') setOwnerRequests(own.value.data);
        if (del.status === 'fulfilled') setDeletionRequests(del.value.data);

        const firstFailure = [ovr, usr, prj, own, del].find((r) => r.status === 'rejected');
        if (firstFailure && firstFailure.status === 'rejected') {
            setError(errorMessage(firstFailure.reason, 'Some admin data could not be loaded.'));
            log.warn('admin load partial failure', firstFailure.reason);
        } else {
            setError('');
        }
    }, []);

    useEffect(() => {
        loadAll();
    }, [loadAll]);

    const handleRoleChange = async (user: AdminUser, role: Role) => {
        if (role === user.role) return;
        setSavingUserId(user.id);
        setError('');
        try {
            const res = await api.updateUserRole(user.id, role);
            setUsers((prev) => prev.map((u) => (u.id === user.id ? res.data : u)));
            // Role changes shift the overview counts — refresh them.
            api.fetchAdminOverview().then((r) => setOverview(r.data)).catch(() => {});
        } catch (err) {
            // 409 = would demote the last admin. Surface the backend's message.
            setError(errorMessage(err, 'Could not change that role.'));
        } finally {
            setSavingUserId(null);
        }
    };

    const handleOwnerDecision = async (id: number, decision: 'APPROVED' | 'DENIED') => {
        try {
            await api.decideOwnerRequest(id, decision);
            setOwnerRequests((prev) => prev.filter((r) => r.id !== id));
            loadAll();
        } catch (err) {
            setError(errorMessage(err, 'Could not record that decision.'));
        }
    };

    const handleDeletionDecision = async (id: number, decision: 'APPROVED' | 'DENIED') => {
        try {
            await api.decideDeletionRequest(id, decision);
            setDeletionRequests((prev) => prev.filter((r) => r.id !== id));
            loadAll();
        } catch (err) {
            setError(errorMessage(err, 'Could not record that decision.'));
        }
    };

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
            <PageHeader title="Admin" subtitle="Manage users, requests, and every project." />

            <div className="flex-1 overflow-y-auto">
                <div className="mx-auto w-full max-w-[1400px] px-6 py-6 md:px-10 flex flex-col gap-8 reveal">
                    {error && (
                        <div className="flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
                            <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                            {error}
                        </div>
                    )}

                    {/* Overview stats — platform totals + pending activity */}
                    <section className="flex flex-col gap-4">
                        <StatGroup
                            stats={[
                                { label: 'Users', value: overview?.totalUsers ?? '—' },
                                { label: 'Owners', value: overview?.owners ?? '—' },
                                { label: 'Admins', value: overview?.admins ?? '—' },
                                { label: 'Projects', value: overview?.totalProjects ?? '—' },
                            ]}
                        />
                        <StatGroup
                            stats={[
                                { label: 'Running', value: overview?.runningProjects ?? '—' },
                                { label: 'Owner requests', value: overview?.pendingOwnerRequests ?? '—' },
                                { label: 'Deletion requests', value: overview?.pendingDeletionRequests ?? '—' },
                                { label: 'Access requests', value: overview?.pendingAccessRequests ?? '—' },
                            ]}
                        />
                    </section>

                    {/* Owner-promotion queue */}
                    <section className="flex flex-col gap-3">
                        <h2 className={sectionTitle}>Owner requests ({ownerRequests.length})</h2>
                        {ownerRequests.length === 0 ? (
                            <Card padding="lg" className="text-body text-fg-muted">No pending owner requests.</Card>
                        ) : (
                            <div className="flex flex-col gap-2">
                                {ownerRequests.map((r) => (
                                    <Card key={r.id} padding="md" className="flex items-center justify-between gap-4">
                                        <div className="min-w-0">
                                            <p className="text-body font-medium text-fg truncate">
                                                {r.username} <span className="text-fg-muted">· {r.email}</span>
                                            </p>
                                            {r.message && <p className="text-caption text-fg-muted truncate">{r.message}</p>}
                                        </div>
                                        <div className="flex items-center gap-2 flex-shrink-0">
                                            <Button size="sm" variant="secondary" onClick={() => handleOwnerDecision(r.id, 'APPROVED')}>
                                                Approve
                                            </Button>
                                            <Button
                                                size="sm"
                                                variant="ghost"
                                                className="text-danger hover:text-danger"
                                                onClick={() => handleOwnerDecision(r.id, 'DENIED')}
                                            >
                                                Deny
                                            </Button>
                                        </div>
                                    </Card>
                                ))}
                            </div>
                        )}
                    </section>

                    {/* Deletion queue */}
                    <section className="flex flex-col gap-3">
                        <h2 className={sectionTitle}>Deletion requests ({deletionRequests.length})</h2>
                        {deletionRequests.length === 0 ? (
                            <Card padding="lg" className="text-body text-fg-muted">No pending deletion requests.</Card>
                        ) : (
                            <div className="flex flex-col gap-2">
                                {deletionRequests.map((r) => (
                                    <Card key={r.id} padding="md" className="flex items-center justify-between gap-4">
                                        <div className="min-w-0">
                                            <p className="text-body font-medium text-fg truncate">
                                                {r.projectName} <span className="text-fg-muted">· by {r.requestedByUsername}</span>
                                            </p>
                                            {r.reason && <p className="text-caption text-fg-muted truncate">{r.reason}</p>}
                                        </div>
                                        <div className="flex items-center gap-2 flex-shrink-0">
                                            <Button size="sm" variant="secondary" onClick={() => setConfirmDeletion(r)}>
                                                Approve deletion
                                            </Button>
                                            <Button
                                                size="sm"
                                                variant="ghost"
                                                className="text-danger hover:text-danger"
                                                onClick={() => handleDeletionDecision(r.id, 'DENIED')}
                                            >
                                                Deny
                                            </Button>
                                        </div>
                                    </Card>
                                ))}
                            </div>
                        )}
                    </section>

                    {/* Users table */}
                    <section className="flex flex-col gap-3">
                        <h2 className={sectionTitle}>Users ({users.length})</h2>
                        <Card padding="none" className="overflow-hidden">
                            <div className="overflow-x-auto">
                                <table className="w-full border-collapse">
                                    <thead className="border-b border-hairline bg-surface-2">
                                        <tr>
                                            <th className={th}>User</th>
                                            <th className={th}>Email</th>
                                            <th className={th}>Owned</th>
                                            <th className={th}>Memberships</th>
                                            <th className={th}>Role</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {users.map((u) => (
                                            <tr key={u.id} className="border-b border-hairline last:border-0">
                                                <td className={`${td} font-medium`}>{u.username}</td>
                                                <td className={`${td} text-fg-muted`}>{u.email}</td>
                                                <td className={`${td} font-mono tabular-nums`}>{u.projectsOwned}</td>
                                                <td className={`${td} font-mono tabular-nums`}>{u.memberships}</td>
                                                <td className={td}>
                                                    <div className="w-40">
                                                        <Select
                                                            value={u.role}
                                                            aria-label={`Change role for ${u.username}`}
                                                            disabled={savingUserId === u.id}
                                                            onChange={(e) => handleRoleChange(u, e.target.value as Role)}
                                                        >
                                                            {ROLE_OPTIONS.map((r) => (
                                                                <option key={r} value={r}>{ROLE_LABEL[r]}</option>
                                                            ))}
                                                        </Select>
                                                    </div>
                                                </td>
                                            </tr>
                                        ))}
                                        {users.length === 0 && (
                                            <tr>
                                                <td className={`${td} text-fg-muted`} colSpan={5}>No users.</td>
                                            </tr>
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </Card>
                    </section>

                    {/* All projects table */}
                    <section className="flex flex-col gap-3">
                        <h2 className={sectionTitle}>All projects ({projects.length})</h2>
                        <Card padding="none" className="overflow-hidden">
                            <div className="overflow-x-auto">
                                <table className="w-full border-collapse">
                                    <thead className="border-b border-hairline bg-surface-2">
                                        <tr>
                                            <th className={th}>Project</th>
                                            <th className={th}>Owner</th>
                                            <th className={th}>Model</th>
                                            <th className={th}>Participants</th>
                                            <th className={th}>Visibility</th>
                                            <th className={th}>Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {projects.map((p) => (
                                            <tr key={p.id} className="border-b border-hairline last:border-0">
                                                <td className={`${td} font-medium`}>{p.name}</td>
                                                <td className={`${td} text-fg-muted`}>{p.ownerUsername}</td>
                                                <td className={`${td} text-fg-muted`}>{p.modelType}</td>
                                                <td className={`${td} font-mono tabular-nums`}>{p.participantCount}</td>
                                                <td className={`${td} text-fg-muted`}>{p.visibility}</td>
                                                <td className={td}>
                                                    <StatusPill status={projectStatusKind(p.status)}>{p.status}</StatusPill>
                                                </td>
                                            </tr>
                                        ))}
                                        {projects.length === 0 && (
                                            <tr>
                                                <td className={`${td} text-fg-muted`} colSpan={6}>No projects.</td>
                                            </tr>
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </Card>
                    </section>
                </div>
            </div>

            <ConfirmDialog
                open={!!confirmDeletion}
                title="Approve deletion?"
                message={
                    confirmDeletion
                        ? `Approving permanently deletes "${confirmDeletion.projectName}" and all its results. This can't be undone.`
                        : ''
                }
                confirmLabel="Delete permanently"
                cancelLabel="Cancel"
                danger
                onConfirm={() => {
                    if (confirmDeletion) handleDeletionDecision(confirmDeletion.id, 'APPROVED');
                    setConfirmDeletion(null);
                }}
                onCancel={() => setConfirmDeletion(null)}
            />
        </div>
    );
}
