// =============================================================================
// FedLearn Frontend — ClientDashboard (role: USER)
// =============================================================================
// What a plain USER can do on the web: ask to become a project owner, and
// discover projects to join/request access to. Actual training happens in the
// desktop app — surfaced as a note here.

import { useState, useEffect, useCallback } from 'react';
import { AlertCircle, Compass, Monitor, Send } from 'lucide-react';
import * as api from '../../services/apiServices';
import {
    errorStatus,
    errorMessage,
    isEmptyBody,
    type OwnerRequest,
    type DiscoverableProject,
} from '../../services/apiServices';
import { Button, Card, FormField, SectionLabel, StatusPill } from '../ui';
import { PageHeader } from './PageHeader';
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
                    <h3 className="text-h4 font-semibold text-fg truncate">{project.name}</h3>
                    <p className="text-caption text-fg-muted mt-0.5 truncate">
                        by {project.ownerUsername} · {project.modelType}
                    </p>
                </div>
                <SectionLabel className="flex-shrink-0">{project.visibility}</SectionLabel>
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
            <PageHeader title="Overview" subtitle="Join projects and train on your devices." />

            <div className="flex-1 overflow-y-auto">
                <div className="mx-auto w-full max-w-[1400px] px-6 py-6 md:px-10 flex flex-col gap-8 reveal">
                    {/* Request owner access */}
                    <Card padding="lg" className="flex flex-col gap-4">
                        <div>
                            <h2 className="text-h4 font-semibold text-fg">Become a project owner</h2>
                            <p className="text-body text-fg-muted mt-1">
                                Owners can create projects, set who can join, and manage participants. Ask a
                                platform admin to upgrade your account.
                            </p>
                        </div>

                        {isApproved ? (
                            <div className="flex items-center gap-3 rounded-md border border-hairline bg-surface-2 px-4 py-3 text-body text-fg-muted">
                                <StatusPill status="completed">Approved</StatusPill>
                                You're now a project owner — your owner dashboard will appear shortly (switch tabs and back to refresh now).
                            </div>
                        ) : hasPending ? (
                            <div className="flex items-center gap-3 rounded-md border border-hairline bg-surface-2 px-4 py-3 text-body text-fg-muted">
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
                                <FormField
                                    label="Message to admins (optional)"
                                    help="Tell the admins why you'd like to own projects."
                                    error={requestError || undefined}
                                >
                                    <textarea
                                        value={message}
                                        onChange={(e) => setMessage(e.target.value)}
                                        rows={3}
                                        className="w-full resize-none bg-surface-2 border border-hairline rounded-md px-3 py-2 text-body text-fg placeholder:text-fg-subtle transition-[border-color,box-shadow] duration-[140ms] hover:border-line focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
                                    />
                                </FormField>
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
                            <Compass className="h-5 w-5 text-fg-muted" strokeWidth={1.5} />
                            <h2 className="text-h4 font-semibold text-fg">Discover projects</h2>
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
