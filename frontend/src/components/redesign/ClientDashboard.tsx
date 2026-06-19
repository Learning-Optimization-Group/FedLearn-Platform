// =============================================================================
// FedLearn Frontend — ClientDashboard (role: USER)
// =============================================================================
// What a plain USER can do on the web: ask to become a project owner, and
// discover projects to join/request access to. Actual training happens in the
// desktop app — surfaced as a note here.

import { useState, useEffect, useCallback } from 'react';
import { ArrowUpCircle, AlertCircle, Compass, Monitor, Send } from 'lucide-react';
import * as api from '../../services/apiServices';
import {
    errorStatus,
    errorMessage,
    isEmptyBody,
    type OwnerRequest,
    type DiscoverableProject,
} from '../../services/apiServices';
import { Button, Card, StatusPill } from '../ui';
import { createLogger } from '../../lib/logger';

const log = createLogger('ClientDashboard');

function DiscoverCard({
    project,
    onAction,
}: {
    project: DiscoverableProject;
    onAction: (project: DiscoverableProject) => void;
}) {
    const isPublic = project.visibility === 'PUBLIC';
    const requested = project.myRequestStatus === 'PENDING';
    const joined = project.myRequestStatus === 'APPROVED';

    return (
        <Card padding="lg" className="flex flex-col gap-3">
            <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                    <h3 className="text-h4 font-display font-semibold tracking-tight text-fg truncate">
                        {project.name}
                    </h3>
                    <p className="text-caption text-fg-muted mt-0.5 truncate">
                        by {project.ownerUsername} · {project.modelType}
                    </p>
                </div>
                <span className="text-caption font-medium uppercase tracking-wider text-fg-subtle flex-shrink-0">
                    {project.visibility}
                </span>
            </div>

            {project.description && (
                <p className="text-body text-fg-muted line-clamp-2">{project.description}</p>
            )}

            <div className="mt-auto pt-2">
                {joined ? (
                    <StatusPill status="completed">Joined</StatusPill>
                ) : requested ? (
                    <StatusPill status="pending">Request pending</StatusPill>
                ) : project.myRequestStatus === 'DENIED' ? (
                    <StatusPill status="error">Request denied</StatusPill>
                ) : (
                    <Button variant="secondary" className="w-full" onClick={() => onAction(project)}>
                        {isPublic ? 'Join project' : 'Request access'}
                    </Button>
                )}
            </div>
        </Card>
    );
}

export function ClientDashboard() {
    // ── Owner-promotion request state ──────────────────────────────────────
    const [ownerRequest, setOwnerRequest] = useState<OwnerRequest | null>(null);
    const [message, setMessage] = useState('');
    const [submitting, setSubmitting] = useState(false);
    const [requestError, setRequestError] = useState('');

    // ── Discover state ─────────────────────────────────────────────────────
    const [discoverable, setDiscoverable] = useState<DiscoverableProject[]>([]);
    const [discoverError, setDiscoverError] = useState('');

    const loadOwnerRequest = useCallback(async () => {
        try {
            const res = await api.fetchMyOwnerRequest();
            setOwnerRequest(isEmptyBody(res.data) ? null : (res.data as OwnerRequest));
        } catch (err) {
            log.warn('fetchMyOwnerRequest failed', err);
        }
    }, []);

    const loadDiscoverable = useCallback(async () => {
        try {
            const res = await api.fetchDiscoverableProjects();
            setDiscoverable(Array.isArray(res.data) ? res.data : []);
            setDiscoverError('');
        } catch (err) {
            setDiscoverError(errorMessage(err, 'Could not load projects to discover.'));
        }
    }, []);

    useEffect(() => {
        loadOwnerRequest();
        loadDiscoverable();
    }, [loadOwnerRequest, loadDiscoverable]);

    const handleSubmitOwnerRequest = async () => {
        setSubmitting(true);
        setRequestError('');
        try {
            const res = await api.submitOwnerRequest(message.trim() || undefined);
            setOwnerRequest(res.data);
            setMessage('');
        } catch (err) {
            // 409 = already pending / already an owner — surface the backend copy.
            setRequestError(errorMessage(err, 'Could not submit your request.'));
            if (errorStatus(err) === 409) loadOwnerRequest();
        } finally {
            setSubmitting(false);
        }
    };

    const handleDiscoverAction = async (project: DiscoverableProject) => {
        try {
            await api.requestProjectAccess(project.id);
            // Refresh so the card flips to "joined" (PUBLIC) or "pending" (RESTRICTED).
            loadDiscoverable();
        } catch (err) {
            setDiscoverError(errorMessage(err, 'Could not request access to that project.'));
        }
    };

    const hasPending = ownerRequest?.status === 'PENDING';
    const isApproved = ownerRequest?.status === 'APPROVED';

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
            <header className="flex items-center justify-between gap-4 px-6 md:px-10 h-20 border-b border-hairline bg-canvas/80 backdrop-blur-xl sticky top-0 z-20">
                <div>
                    <h1 className="text-h3 font-display font-semibold tracking-tight text-fg">Overview</h1>
                    <p className="text-label text-fg-muted mt-0.5">Join projects and train on your devices.</p>
                </div>
            </header>

            <div className="flex-1 overflow-y-auto px-6 md:px-10 py-8 relative z-10 bg-canvas">
                <div className="max-w-[1600px] mx-auto flex flex-col gap-8">
                    {/* Request owner access */}
                    <Card padding="lg" glow className="flex flex-col gap-4">
                        <div className="flex items-start gap-3">
                            <div className="grid h-10 w-10 flex-shrink-0 place-items-center rounded-card border border-hairline bg-surface-1">
                                <ArrowUpCircle className="h-5 w-5 text-accent" strokeWidth={1.5} />
                            </div>
                            <div>
                                <h2 className="text-h4 font-display font-semibold text-fg">Become a project owner</h2>
                                <p className="text-body text-fg-muted mt-1">
                                    Owners can create projects, set who can join, and manage participants. Ask a
                                    platform admin to upgrade your account.
                                </p>
                            </div>
                        </div>

                        {isApproved ? (
                            <div className="flex items-center gap-2 rounded-md border border-success/30 bg-success/10 px-4 py-3 text-body font-medium text-success">
                                <StatusPill status="completed">Approved</StatusPill>
                                You're now a project owner — sign out and back in to see your owner dashboard.
                            </div>
                        ) : hasPending ? (
                            <div className="flex items-center gap-2 rounded-md border border-warning/30 bg-warning/10 px-4 py-3 text-body font-medium text-warning">
                                <StatusPill status="pending">Pending</StatusPill>
                                Your request is awaiting review by a platform admin.
                            </div>
                        ) : (
                            <div className="flex flex-col gap-3">
                                {ownerRequest?.status === 'DENIED' && (
                                    <div className="flex items-center gap-2 text-label text-fg-muted">
                                        <StatusPill status="error">Previously denied</StatusPill>
                                        You can submit a new request.
                                    </div>
                                )}
                                <textarea
                                    value={message}
                                    onChange={(e) => setMessage(e.target.value)}
                                    placeholder="Optional: tell the admins why you'd like to own projects."
                                    rows={3}
                                    className="w-full resize-none bg-surface-2 border border-hairline rounded-md px-3 py-2 text-body text-fg placeholder:text-fg-subtle transition-[border-color,box-shadow] duration-[140ms] hover:border-line focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
                                />
                                {requestError && (
                                    <p className="flex items-center gap-2 text-label text-danger">
                                        <AlertCircle className="h-4 w-4 flex-shrink-0" strokeWidth={1.5} />
                                        {requestError}
                                    </p>
                                )}
                                <Button onClick={handleSubmitOwnerRequest} disabled={submitting} className="self-start">
                                    <Send className="h-[18px] w-[18px]" strokeWidth={1.5} />
                                    {submitting ? 'Sending…' : 'Request owner access'}
                                </Button>
                            </div>
                        )}
                    </Card>

                    {/* Desktop-app note */}
                    <div className="flex items-start gap-3 rounded-card border border-hairline bg-surface-1 px-4 py-3 text-body text-fg-muted">
                        <Monitor className="h-5 w-5 flex-shrink-0 text-fg-subtle mt-0.5" strokeWidth={1.5} />
                        <span>
                            Training runs in the <span className="text-fg font-medium">FedLearn desktop app</span>.
                            Join a project below, then open it in the desktop app to contribute your device.
                        </span>
                    </div>

                    {/* Discover projects */}
                    <div className="flex flex-col gap-4">
                        <div className="flex items-center gap-2">
                            <Compass className="h-5 w-5 text-accent" strokeWidth={1.5} />
                            <h2 className="text-h4 font-display font-semibold text-fg">Discover projects</h2>
                        </div>

                        {discoverError && (
                            <div className="flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
                                <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                                {discoverError}
                            </div>
                        )}

                        {discoverable.length > 0 ? (
                            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                                {discoverable.map((p) => (
                                    <DiscoverCard key={p.id} project={p} onAction={handleDiscoverAction} />
                                ))}
                            </div>
                        ) : (
                            <Card padding="lg" className="text-center text-body text-fg-muted">
                                No public or discoverable projects right now. Check back soon.
                            </Card>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
