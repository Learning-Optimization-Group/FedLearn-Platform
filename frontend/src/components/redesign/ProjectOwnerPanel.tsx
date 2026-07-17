// =============================================================================
// FedLearn Frontend — ProjectOwnerPanel (owner controls for one project)
// =============================================================================
// A single modal that bundles the owner-only controls the base ProjectCard
// doesn't have: visibility (3 tiers), join-request approvals, membership
// management, and "request deletion" (admin approves the actual delete).

import { useState, useEffect, useCallback } from 'react';
import { Check, X, UserPlus, Trash2, AlertCircle } from 'lucide-react';
import * as api from '../../services/apiServices';
import {
    errorMessage,
    isEmptyBody,
    VISIBILITY_HELP,
    type OwnedProject,
    type Visibility,
    type AccessRequest,
    type Membership,
    type DeletionRequest,
} from '../../services/apiServices';
import { Modal, Button, Input, Select, StatusPill, FormField, SectionLabel, ConfirmDialog } from '../ui';
import { createLogger } from '../../lib/logger';

const log = createLogger('ProjectOwnerPanel');

const VISIBILITY_ORDER: Visibility[] = ['PUBLIC', 'RESTRICTED', 'PRIVATE'];

interface ProjectOwnerPanelProps {
    open: boolean;
    project: OwnedProject | null;
    onClose: () => void;
    /** Fired after a change that the dashboard should reflect (e.g. visibility). */
    onChanged: () => void;
}

export function ProjectOwnerPanel({ open, project, onClose, onChanged }: ProjectOwnerPanelProps) {
    const [visibility, setVisibility] = useState<Visibility>('PRIVATE');
    const [requests, setRequests] = useState<AccessRequest[]>([]);
    const [members, setMembers] = useState<Membership[]>([]);
    const [deletion, setDeletion] = useState<DeletionRequest | null>(null);
    const [error, setError] = useState('');
    const [savingVisibility, setSavingVisibility] = useState(false);

    // Add-member form
    const [newUsername, setNewUsername] = useState('');
    const [newRole, setNewRole] = useState<'CLIENT' | 'MEMBER'>('CLIENT');

    // Request-deletion form
    const [deleteReason, setDeleteReason] = useState('');
    const [requestingDelete, setRequestingDelete] = useState(false);
    const [confirmRequestDelete, setConfirmRequestDelete] = useState(false);

    const projectId = project?.id ?? '';

    const loadAll = useCallback(async () => {
        if (!projectId) return;
        setError('');
        const [reqRes, memRes, delRes] = await Promise.allSettled([
            api.fetchAccessRequests(projectId, 'PENDING'),
            api.fetchMemberships(projectId),
            api.fetchProjectDeletionRequest(projectId),
        ]);
        if (reqRes.status === 'fulfilled') setRequests(reqRes.value.data);
        if (memRes.status === 'fulfilled') setMembers(memRes.value.data);
        if (delRes.status === 'fulfilled') {
            setDeletion(isEmptyBody(delRes.value.data) ? null : (delRes.value.data as DeletionRequest));
        }
        if (reqRes.status === 'rejected') log.warn('fetchAccessRequests failed', reqRes.reason);
    }, [projectId]);

    useEffect(() => {
        if (open && project) {
            setVisibility(project.visibility);
            setDeleteReason('');
            setNewUsername('');
            loadAll();
        }
    }, [open, project, loadAll]);

    if (!project) return null;

    const handleSaveVisibility = async (next: Visibility) => {
        setVisibility(next);
        setSavingVisibility(true);
        setError('');
        try {
            await api.updateProjectVisibility(project.id, { visibility: next });
            onChanged();
        } catch (err) {
            setError(errorMessage(err, 'Could not change visibility.'));
            setVisibility(project.visibility);
        } finally {
            setSavingVisibility(false);
        }
    };

    const handleDecide = async (requestId: number, decision: 'APPROVED' | 'DENIED') => {
        try {
            await api.decideAccessRequest(project.id, requestId, decision);
            setRequests((prev) => prev.filter((r) => r.id !== requestId));
            if (decision === 'APPROVED') {
                const memRes = await api.fetchMemberships(project.id);
                setMembers(memRes.data);
            }
        } catch (err) {
            setError(errorMessage(err, 'Could not record that decision.'));
        }
    };

    const handleAddMember = async () => {
        if (!newUsername.trim()) return;
        try {
            await api.addMembership(project.id, newUsername.trim(), newRole);
            setNewUsername('');
            const memRes = await api.fetchMemberships(project.id);
            setMembers(memRes.data);
        } catch (err) {
            setError(errorMessage(err, 'Could not add that member.'));
        }
    };

    const handleRemoveMember = async (userId: number) => {
        try {
            await api.removeMembership(project.id, userId);
            setMembers((prev) => prev.filter((m) => m.userId !== userId));
        } catch (err) {
            setError(errorMessage(err, 'Could not remove that member.'));
        }
    };

    const handleRequestDeletion = async () => {
        setRequestingDelete(true);
        setError('');
        try {
            const res = await api.submitDeletionRequest(project.id, deleteReason.trim() || undefined);
            setDeletion(res.data);
            setDeleteReason('');
        } catch (err) {
            setError(errorMessage(err, 'Could not submit a deletion request.'));
        } finally {
            setRequestingDelete(false);
        }
    };

    return (
        <Modal
            open={open}
            onClose={onClose}
            size="lg"
            title={`Manage “${project.name}”`}
        >
            <div className="flex flex-col gap-6">
                {error && (
                    <p className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-3 py-2.5 text-label text-danger">
                        <AlertCircle className="h-4 w-4 flex-shrink-0" strokeWidth={1.5} />
                        {error}
                    </p>
                )}

                {/* Visibility */}
                <section className="flex flex-col gap-2">
                    <FormField label="Visibility" help={VISIBILITY_HELP[visibility]}>
                        <Select
                            value={visibility}
                            disabled={savingVisibility}
                            onChange={(e) => handleSaveVisibility(e.target.value as Visibility)}
                        >
                            {VISIBILITY_ORDER.map((v) => (
                                <option key={v} value={v}>
                                    {v.charAt(0) + v.slice(1).toLowerCase()}
                                </option>
                            ))}
                        </Select>
                    </FormField>
                </section>

                {/* Access requests */}
                <section className="flex flex-col gap-2">
                    <SectionLabel>Join requests ({requests.length})</SectionLabel>
                    {requests.length === 0 ? (
                        <p className="text-body text-fg-muted">No pending requests.</p>
                    ) : (
                        <ul className="flex flex-col gap-2">
                            {requests.map((r) => (
                                <li
                                    key={r.id}
                                    className="flex items-center justify-between gap-3 rounded-md border border-hairline bg-surface-2 px-3 py-2"
                                >
                                    <div className="min-w-0">
                                        <p className="text-body font-medium text-fg truncate">{r.username}</p>
                                        {r.message && (
                                            <p className="text-caption text-fg-muted truncate">{r.message}</p>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-2 flex-shrink-0">
                                        <Button size="sm" variant="secondary" onClick={() => handleDecide(r.id, 'APPROVED')}>
                                            <Check className="h-3.5 w-3.5" strokeWidth={2} /> Approve
                                        </Button>
                                        <Button
                                            size="sm"
                                            variant="ghost"
                                            className="text-danger hover:text-danger"
                                            onClick={() => handleDecide(r.id, 'DENIED')}
                                        >
                                            <X className="h-3.5 w-3.5" strokeWidth={2} /> Deny
                                        </Button>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    )}
                </section>

                {/* Members */}
                <section className="flex flex-col gap-2">
                    <SectionLabel>Members ({members.length})</SectionLabel>
                    {members.length > 0 && (
                        <ul className="flex flex-col gap-2">
                            {members.map((m) => (
                                <li
                                    key={m.userId}
                                    className="flex items-center justify-between gap-3 rounded-md border border-hairline bg-surface-2 px-3 py-2"
                                >
                                    <div className="min-w-0">
                                        <p className="text-body font-medium text-fg truncate">{m.username}</p>
                                        <p className="text-caption text-fg-muted">
                                            {m.role}
                                            {m.partitionId != null && ` · partition ${m.partitionId}`}
                                        </p>
                                    </div>
                                    <button
                                        onClick={() => handleRemoveMember(m.userId)}
                                        className="rounded-md p-1.5 text-fg-muted transition-colors hover:bg-surface-3 hover:text-danger"
                                        title="Remove member"
                                        aria-label={`Remove ${m.username}`}
                                    >
                                        <Trash2 className="h-4 w-4" strokeWidth={1.5} />
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}
                    <div className="flex items-end gap-2">
                        <FormField label="Add by username" className="flex-1">
                            <Input
                                value={newUsername}
                                onChange={(e) => setNewUsername(e.target.value)}
                                placeholder="username"
                            />
                        </FormField>
                        <FormField label="Role" className="w-32">
                            <Select value={newRole} onChange={(e) => setNewRole(e.target.value as 'CLIENT' | 'MEMBER')}>
                                <option value="CLIENT">Client</option>
                                <option value="MEMBER">Member</option>
                            </Select>
                        </FormField>
                        <Button variant="secondary" onClick={handleAddMember} disabled={!newUsername.trim()}>
                            <UserPlus className="h-[18px] w-[18px]" strokeWidth={1.5} /> Add member
                        </Button>
                    </div>
                </section>

                {/* Request deletion */}
                <section className="flex flex-col gap-2 border-t border-hairline pt-5">
                    <SectionLabel>Danger zone</SectionLabel>
                    {deletion ? (
                        <div className="flex items-center gap-2 rounded-md border border-warning/30 bg-warning/10 px-3 py-2.5 text-body font-medium text-warning">
                            <StatusPill status="pending">Pending</StatusPill>
                            Deletion requested — awaiting platform-admin approval.
                        </div>
                    ) : (
                        <>
                            <p className="text-caption text-fg-muted">
                                Deleting a project is permanent and must be approved by a platform admin.
                            </p>
                            <FormField label="Reason (optional)">
                                <Input
                                    value={deleteReason}
                                    onChange={(e) => setDeleteReason(e.target.value)}
                                    placeholder="Why should this project be deleted?"
                                />
                            </FormField>
                            <Button
                                variant="ghost"
                                onClick={() => setConfirmRequestDelete(true)}
                                disabled={requestingDelete}
                                className="self-start text-danger hover:text-danger"
                            >
                                <Trash2 className="h-[18px] w-[18px]" strokeWidth={1.5} />
                                {requestingDelete ? 'Requesting…' : 'Request deletion'}
                            </Button>
                        </>
                    )}
                </section>
            </div>

            {/* Destructive confirm — solid danger fill, never weaker than Cancel. */}
            <ConfirmDialog
                open={confirmRequestDelete}
                title="Request project deletion?"
                message={`Deleting “${project.name}” is permanent. A platform admin has to approve the request before anything is removed.`}
                confirmLabel="Request deletion"
                cancelLabel="Cancel"
                danger
                onConfirm={() => {
                    setConfirmRequestDelete(false);
                    handleRequestDeletion();
                }}
                onCancel={() => setConfirmRequestDelete(false)}
            />
        </Modal>
    );
}
