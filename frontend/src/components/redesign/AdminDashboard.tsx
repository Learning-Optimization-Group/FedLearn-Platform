// =============================================================================
// FedLearn Frontend — AdminDashboard (role: PLATFORM_ADMIN)
// =============================================================================
// The platform-admin home, deliberately BOUNDED: nothing unbounded ever renders
// here. The approval queues cap at 5 visible rows with an inline expander, the
// platform stats deep-link into the directories, and the old full user/project
// tables are replaced by compact 5-row directory cards fed by the paginated
// search endpoints (never the full legacy lists). Directory work (search,
// role + status management) lives on /nodes and /admin/projects.
// 403 from any of these means "not allowed" (shouldn't happen for an admin) and
// is rendered inline rather than logging out.

import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { AlertCircle, ArrowRight, ArrowUpRight, CheckCircle2 } from 'lucide-react';
import * as api from '../../services/apiServices';
import {
    errorMessage,
    type AdminOverview,
    type AdminUser,
    type AdminProject,
    type OwnerRequest,
    type DeletionRequest,
    type Paged,
} from '../../services/apiServices';
import { Card, Button, StatusPill, StatGroup, ConfirmDialog, SectionLabel, toStatusKind } from '../ui';
import { PageHeader } from './PageHeader';
import { createLogger } from '../../lib/logger';

const log = createLogger('AdminDashboard');

// Hard cap on rows a queue shows before the inline expander takes over.
const QUEUE_CAP = 5;
// Page size the compact directory cards request from the search endpoints.
const RECENT_CAP = 5;

function formatDate(iso?: string): string {
    if (!iso) return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleDateString();
}

const sectionTitle = 'text-h4 font-semibold text-fg';

/** Value slot for a linked platform stat — the number itself is the link. */
function StatLink({ to, value, label }: { to: string; value: number | string; label: string }) {
    return (
        <Link
            to={to}
            aria-label={label}
            className="group inline-flex items-center gap-1.5 rounded-sm text-fg transition-colors hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface-1"
        >
            {value}
            <ArrowUpRight
                className="h-4 w-4 text-fg-subtle transition-colors group-hover:text-accent"
                strokeWidth={1.5}
            />
        </Link>
    );
}

/** Card header for the compact directory cards: title + "View all →". */
function RecentCardHeader({ title, to, linkLabel }: { title: string; to: string; linkLabel: string }) {
    return (
        <div className="flex items-center justify-between border-b border-hairline px-4 py-3">
            <span className="text-body font-semibold text-fg">{title}</span>
            <Link
                to={to}
                className="inline-flex items-center gap-1 text-label font-medium text-fg-muted transition-colors hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent rounded-sm"
            >
                {linkLabel}
                <ArrowRight className="h-3.5 w-3.5" strokeWidth={1.5} />
            </Link>
        </div>
    );
}

/** Inline per-queue expander: "Show all N" / "Show fewer". */
function QueueExpander({
    total,
    expanded,
    onToggle,
    label,
}: {
    total: number;
    expanded: boolean;
    onToggle: () => void;
    label: string;
}) {
    if (total <= QUEUE_CAP) return null;
    return (
        <Button variant="ghost" size="sm" className="self-start" onClick={onToggle} aria-label={label}>
            {expanded ? 'Show fewer' : `Show all ${total}`}
        </Button>
    );
}

export function AdminDashboard() {
    const [overview, setOverview] = useState<AdminOverview | null>(null);
    const [usersPage, setUsersPage] = useState<Paged<AdminUser> | null>(null);
    const [projectsPage, setProjectsPage] = useState<Paged<AdminProject> | null>(null);
    const [ownerRequests, setOwnerRequests] = useState<OwnerRequest[]>([]);
    const [deletionRequests, setDeletionRequests] = useState<DeletionRequest[]>([]);
    const [error, setError] = useState('');
    const [showAllOwnerRequests, setShowAllOwnerRequests] = useState(false);
    const [showAllDeletionRequests, setShowAllDeletionRequests] = useState(false);

    // Deletion approval needs an explicit confirm (it permanently deletes).
    const [confirmDeletion, setConfirmDeletion] = useState<DeletionRequest | null>(null);

    const loadAll = useCallback(async () => {
        const [ovr, usr, prj, own, del] = await Promise.allSettled([
            api.fetchAdminOverview(),
            api.searchAdminUsers({ page: 0, size: RECENT_CAP }),
            api.searchAdminProjects({ page: 0, size: RECENT_CAP }),
            api.fetchOwnerRequests('PENDING'),
            api.fetchDeletionRequests('PENDING'),
        ]);
        if (ovr.status === 'fulfilled') setOverview(ovr.value.data);
        if (usr.status === 'fulfilled') setUsersPage(usr.value.data);
        if (prj.status === 'fulfilled') setProjectsPage(prj.value.data);
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

    // First page (size RECENT_CAP) from each search endpoint — the server
    // orders these (username / name asc), so the cards make no "newest" claim.
    const recentUsers = usersPage?.items ?? [];
    const recentProjects = projectsPage?.items ?? [];

    const visibleOwnerRequests = showAllOwnerRequests ? ownerRequests : ownerRequests.slice(0, QUEUE_CAP);
    const visibleDeletionRequests = showAllDeletionRequests
        ? deletionRequests
        : deletionRequests.slice(0, QUEUE_CAP);

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
            <PageHeader title="Admin" subtitle="Review requests and monitor the platform." />

            <div className="flex-1 overflow-y-auto">
                <div className="mx-auto w-full max-w-[1400px] px-6 py-6 md:px-10 flex flex-col gap-8 reveal">
                    {error && (
                        <div className="flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
                            <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                            {error}
                        </div>
                    )}

                    {/* 1 · Needs attention — the queues an admin must act on.
                        Each queue renders at most QUEUE_CAP rows; the inline
                        expander reveals the rest without leaving the page. */}
                    <section className="flex flex-col gap-3">
                        <h2 className={sectionTitle}>Needs attention</h2>
                        {ownerRequests.length === 0 && deletionRequests.length === 0 ? (
                            <p className="flex items-center gap-2 text-body text-fg-muted">
                                <CheckCircle2 className="w-4 h-4 text-success" strokeWidth={1.5} />
                                Nothing waiting for review.
                            </p>
                        ) : (
                            <div className="flex flex-col gap-5">
                                {ownerRequests.length > 0 && (
                                    <div className="flex flex-col gap-2">
                                        <SectionLabel>Owner requests · {ownerRequests.length}</SectionLabel>
                                        {visibleOwnerRequests.map((r) => (
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
                                        <QueueExpander
                                            total={ownerRequests.length}
                                            expanded={showAllOwnerRequests}
                                            onToggle={() => setShowAllOwnerRequests((v) => !v)}
                                            label={
                                                showAllOwnerRequests
                                                    ? 'Show fewer owner requests'
                                                    : `Show all ${ownerRequests.length} owner requests`
                                            }
                                        />
                                    </div>
                                )}
                                {deletionRequests.length > 0 && (
                                    <div className="flex flex-col gap-2">
                                        <SectionLabel>Deletion requests · {deletionRequests.length}</SectionLabel>
                                        {visibleDeletionRequests.map((r) => (
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
                                        <QueueExpander
                                            total={deletionRequests.length}
                                            expanded={showAllDeletionRequests}
                                            onToggle={() => setShowAllDeletionRequests((v) => !v)}
                                            label={
                                                showAllDeletionRequests
                                                    ? 'Show fewer deletion requests'
                                                    : `Show all ${deletionRequests.length} deletion requests`
                                            }
                                        />
                                    </div>
                                )}
                            </div>
                        )}
                    </section>

                    {/* 2 · Platform health — durable totals that deep-link into
                        the matching directory. Pending counts live in the
                        queues above, not duplicated here. */}
                    <section className="flex flex-col gap-3">
                        <h2 className={sectionTitle}>Platform</h2>
                        <StatGroup
                            stats={[
                                {
                                    label: 'Users',
                                    value: (
                                        <StatLink
                                            to="/nodes"
                                            value={overview?.totalUsers ?? '—'}
                                            label="View all users"
                                        />
                                    ),
                                },
                                { label: 'Owners', value: overview?.owners ?? '—' },
                                {
                                    label: 'Projects',
                                    value: (
                                        <StatLink
                                            to="/admin/projects"
                                            value={overview?.totalProjects ?? '—'}
                                            label="View all projects"
                                        />
                                    ),
                                },
                                {
                                    label: 'Running now',
                                    value: (
                                        <StatLink
                                            to="/admin/projects?status=RUNNING"
                                            value={overview?.runningProjects ?? '—'}
                                            label="View running projects"
                                        />
                                    ),
                                },
                            ]}
                        />
                    </section>

                    {/* 3 · Directory snapshots — two glanceable read-only cards
                        showing the first search page, titled "x of N" from the
                        paginated envelope. Role and status management live in
                        the directories, not here. */}
                    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                        <section className="flex flex-col gap-3">
                            <h2 className={sectionTitle}>Users</h2>
                            <Card padding="none" className="overflow-hidden">
                                <RecentCardHeader
                                    title={usersPage ? `${recentUsers.length} of ${usersPage.total}` : 'Users'}
                                    to="/nodes"
                                    linkLabel="View all"
                                />
                                {recentUsers.length === 0 ? (
                                    <p className="px-4 py-6 text-body text-fg-muted">No users yet.</p>
                                ) : (
                                    <ul className="divide-y divide-hairline">
                                        {recentUsers.map((u) => (
                                            <li key={u.id} className="flex items-center justify-between gap-4 px-4 py-3">
                                                <div className="min-w-0">
                                                    <p className="text-body font-medium text-fg truncate">{u.username}</p>
                                                    <p className="text-caption text-fg-muted truncate">{u.email}</p>
                                                </div>
                                                <span className="flex-shrink-0 text-caption text-fg-muted font-mono tabular-nums">
                                                    {formatDate(u.createdAt)}
                                                </span>
                                            </li>
                                        ))}
                                    </ul>
                                )}
                            </Card>
                        </section>

                        <section className="flex flex-col gap-3">
                            <h2 className={sectionTitle}>Projects</h2>
                            <Card padding="none" className="overflow-hidden">
                                <RecentCardHeader
                                    title={projectsPage ? `${recentProjects.length} of ${projectsPage.total}` : 'Projects'}
                                    to="/admin/projects"
                                    linkLabel="View all"
                                />
                                {recentProjects.length === 0 ? (
                                    <p className="px-4 py-6 text-body text-fg-muted">No projects yet.</p>
                                ) : (
                                    <ul className="divide-y divide-hairline">
                                        {recentProjects.map((p) => (
                                            <li key={p.id} className="flex items-center justify-between gap-4 px-4 py-3">
                                                <div className="min-w-0">
                                                    <p className="text-body font-medium text-fg truncate">{p.name}</p>
                                                    <p className="text-caption text-fg-muted truncate">
                                                        {p.ownerUsername} · {p.modelType}
                                                    </p>
                                                </div>
                                                <StatusPill status={toStatusKind(p.status)} className="flex-shrink-0">
                                                    {p.status}
                                                </StatusPill>
                                            </li>
                                        ))}
                                    </ul>
                                )}
                            </Card>
                        </section>
                    </div>
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
