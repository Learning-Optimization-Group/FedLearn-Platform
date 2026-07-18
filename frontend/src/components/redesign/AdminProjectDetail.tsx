// =============================================================================
// FedLearn Frontend — AdminProjectDetail (role: PLATFORM_ADMIN)
// =============================================================================
// Drilldown for one project from the admin directory. Composes existing
// fetchers: GET /projects/:id (core fields), GET /admin/projects (owner /
// visibility / participants — the admin list is the only surface that carries
// them), memberships and results. Actions: open the live log overlay, and stop
// the FL server (confirm-gated, only while RUNNING). The back link restores the
// directory's exact URL state (carried in location.state by the list view).

import { useCallback, useEffect, useState } from 'react';
import { Link, useLocation, useParams } from 'react-router-dom';
import { AlertCircle, ArrowLeft, Check, Copy, ScrollText, Square } from 'lucide-react';
import * as api from '../../services/apiServices';
import {
    errorMessage,
    type AdminProject,
    type Membership,
    type Project,
    type ProjectResult,
} from '../../services/apiServices';
import { Button, Card, ConfirmDialog, MetricTile, SectionLabel, Skeleton, StatusPill, type StatusKind } from '../ui';
import { PageHeader } from './PageHeader';
import { LogViewerV2 } from './LogViewer';
import { createLogger } from '../../lib/logger';

const log = createLogger('AdminProjectDetail');

function projectStatusKind(status?: string): StatusKind {
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

/** Mono id with a copy-to-clipboard affordance. */
function CopyableId({ value, label }: { value: string; label: string }) {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(value);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            log.warn('clipboard copy failed', err);
        }
    };

    return (
        <span className="inline-flex min-w-0 items-center gap-1.5">
            <span className="truncate font-mono text-label text-fg" title={value}>
                {value}
            </span>
            <button
                type="button"
                onClick={handleCopy}
                aria-label={copied ? `${label} copied` : `Copy ${label}`}
                className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md text-fg-muted transition-colors duration-[120ms] hover:bg-surface-2 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
                {copied ? (
                    <Check className="h-3.5 w-3.5 text-success" strokeWidth={1.5} />
                ) : (
                    <Copy className="h-3.5 w-3.5" strokeWidth={1.5} />
                )}
            </button>
        </span>
    );
}

/** One label/value pair in the Overview grid. */
function OverviewItem({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <div className="flex min-w-0 flex-col gap-1">
            <span className="text-caption uppercase tracking-wide text-fg-muted">{label}</span>
            <span className="min-w-0 text-body text-fg">{children}</span>
        </div>
    );
}

export function AdminProjectDetail() {
    const { projectId } = useParams<{ projectId: string }>();
    const location = useLocation();
    // The list view passes its full URL (path + query) so Back restores the
    // exact search/filter/page state. Deep links fall back to the bare list.
    const backTo = (location.state as { from?: string } | null)?.from ?? '/admin/projects';

    const [project, setProject] = useState<Project | null>(null);
    const [adminMeta, setAdminMeta] = useState<AdminProject | null>(null);
    const [members, setMembers] = useState<Membership[] | null>(null);
    const [results, setResults] = useState<ProjectResult[] | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    const [showLogs, setShowLogs] = useState(false);
    const [confirmStop, setConfirmStop] = useState(false);
    const [isStopping, setIsStopping] = useState(false);

    const load = useCallback(async () => {
        if (!projectId) return;
        setIsLoading(true);
        const [prj, adminList, mem, res] = await Promise.allSettled([
            api.fetchProject(projectId),
            api.fetchAdminProjects(),
            api.fetchMemberships(projectId),
            api.fetchProjectResults(projectId),
        ]);
        if (prj.status === 'fulfilled') setProject(prj.value.data);
        if (adminList.status === 'fulfilled') {
            setAdminMeta(adminList.value.data.find((p) => p.id === projectId) ?? null);
        }
        // Members / results stay null on failure so their sections can degrade
        // to a quiet line instead of a page-level error.
        if (mem.status === 'fulfilled') setMembers(mem.value.data);
        if (res.status === 'fulfilled') setResults(res.value.data);

        if (prj.status === 'rejected' && adminList.status === 'rejected') {
            setError(errorMessage(prj.reason, 'Could not load this project.'));
            log.warn('project detail load failed', prj.reason);
        } else {
            setError('');
        }
        setIsLoading(false);
    }, [projectId]);

    useEffect(() => {
        load();
    }, [load]);

    const handleStop = async () => {
        if (!projectId) return;
        setIsStopping(true);
        try {
            const res = await api.stopProjectServer(projectId);
            setProject((prev) => (prev ? { ...prev, ...res.data } : res.data));
            setAdminMeta((prev) => (prev ? { ...prev, status: res.data.status } : prev));
        } catch (err) {
            setError(errorMessage(err, 'Could not stop the server.'));
        } finally {
            setIsStopping(false);
        }
    };

    const name = project?.name ?? adminMeta?.name ?? 'Project';
    const status = project?.status ?? adminMeta?.status;
    const isRunning = status === 'RUNNING';
    const notFound = !isLoading && !project && !adminMeta;

    const latest = results && results.length > 0 ? results[results.length - 1] : null;
    const rounds = latest?.serverRound ?? (results ? results.length : null);

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
            <PageHeader
                title={name}
                subtitle={adminMeta ? `Owned by ${adminMeta.ownerUsername}` : 'Project detail'}
            >
                {status && <StatusPill status={projectStatusKind(status)}>{status}</StatusPill>}
                <Button variant="secondary" size="sm" onClick={() => setShowLogs(true)} disabled={!projectId}>
                    <ScrollText className="h-3.5 w-3.5" strokeWidth={1.5} />
                    Open logs
                </Button>
                {isRunning && (
                    <Button
                        variant="secondary"
                        size="sm"
                        className="text-danger hover:text-danger"
                        disabled={isStopping}
                        onClick={() => setConfirmStop(true)}
                    >
                        <Square className="h-3.5 w-3.5" strokeWidth={1.5} />
                        {isStopping ? 'Stopping…' : 'Stop server'}
                    </Button>
                )}
            </PageHeader>

            <div className="flex-1 overflow-y-auto">
                <div className="mx-auto w-full max-w-[1400px] px-6 py-6 md:px-10 flex flex-col gap-6">
                    <Link
                        to={backTo}
                        className="inline-flex w-fit items-center gap-1.5 rounded-sm text-label font-medium text-fg-muted transition-colors hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                    >
                        <ArrowLeft className="h-4 w-4" strokeWidth={1.5} />
                        All projects
                    </Link>

                    {error && (
                        <div className="flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
                            <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                            {error}
                        </div>
                    )}

                    {isLoading ? (
                        <div className="flex flex-col gap-6">
                            <Card padding="lg" className="flex flex-col gap-3">
                                <Skeleton className="h-4 w-24" />
                                <Skeleton className="h-4 w-full max-w-md" />
                                <Skeleton className="h-4 w-full max-w-sm" />
                            </Card>
                            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                                <Card padding="lg"><Skeleton className="h-24 w-full" /></Card>
                                <Card padding="lg"><Skeleton className="h-24 w-full" /></Card>
                            </div>
                        </div>
                    ) : notFound ? (
                        <div className="flex flex-col items-center justify-center gap-4 pt-16 text-center md:pt-24">
                            <div className="max-w-sm">
                                <p className="text-h4 font-semibold text-fg">Project not found</p>
                                <p className="mt-1 text-caption text-fg-muted">
                                    It may have been deleted, or the link is wrong.
                                </p>
                            </div>
                        </div>
                    ) : (
                        <>
                            {/* Overview */}
                            <section className="flex flex-col gap-3">
                                <SectionLabel>Overview</SectionLabel>
                                <Card padding="lg">
                                    <div className="grid grid-cols-1 gap-x-8 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
                                        <OverviewItem label="Model">
                                            {project
                                                ? `${project.modelType} · ${project.modelName}`
                                                : adminMeta?.modelType ?? '—'}
                                        </OverviewItem>
                                        <OverviewItem label="Optimizer">{project?.optimizer ?? '—'}</OverviewItem>
                                        <OverviewItem label="Visibility">{adminMeta?.visibility ?? '—'}</OverviewItem>
                                        <OverviewItem label="Participants">
                                            <span className="font-mono tabular-nums">
                                                {adminMeta?.participantCount ?? '—'}
                                            </span>
                                        </OverviewItem>
                                        <OverviewItem label="Server port">
                                            <span className="font-mono tabular-nums">
                                                {project?.serverPort ?? '—'}
                                            </span>
                                        </OverviewItem>
                                        {projectId && (
                                            <OverviewItem label="Project ID">
                                                <CopyableId value={projectId} label="project ID" />
                                            </OverviewItem>
                                        )}
                                    </div>
                                </Card>
                            </section>

                            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                                {/* Members */}
                                <section className="flex flex-col gap-3">
                                    <SectionLabel>Members</SectionLabel>
                                    <Card padding="none" className="overflow-hidden">
                                        {members === null ? (
                                            <p className="px-4 py-6 text-body text-fg-muted">
                                                Members could not be loaded.
                                            </p>
                                        ) : members.length === 0 ? (
                                            <p className="px-4 py-6 text-body text-fg-muted">No members yet.</p>
                                        ) : (
                                            <ul className="divide-y divide-hairline">
                                                {members.map((m) => (
                                                    <li
                                                        key={m.userId}
                                                        className="flex items-center justify-between gap-4 px-4 py-3"
                                                    >
                                                        <span className="min-w-0 truncate text-body font-medium text-fg">
                                                            {m.username}
                                                        </span>
                                                        <span className="flex-shrink-0 rounded-pill border border-hairline bg-surface-2 px-2.5 py-0.5 text-caption font-medium text-fg-muted">
                                                            {m.role}
                                                        </span>
                                                    </li>
                                                ))}
                                            </ul>
                                        )}
                                    </Card>
                                </section>

                                {/* Latest results */}
                                <section className="flex flex-col gap-3">
                                    <SectionLabel>Latest results</SectionLabel>
                                    <Card padding="lg">
                                        {results === null ? (
                                            <p className="text-body text-fg-muted">Results could not be loaded.</p>
                                        ) : latest === null ? (
                                            <p className="text-body text-fg-muted">No training results yet.</p>
                                        ) : (
                                            <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
                                                <MetricTile label="Rounds" value={rounds ?? '—'} />
                                                <MetricTile
                                                    label="Final accuracy"
                                                    value={`${(latest.accuracy * 100).toFixed(2)}%`}
                                                />
                                                <MetricTile label="Final loss" value={latest.loss.toFixed(4)} />
                                            </div>
                                        )}
                                    </Card>
                                </section>
                            </div>
                        </>
                    )}
                </div>
            </div>

            {showLogs && projectId && (
                <LogViewerV2 projectId={projectId} onClose={() => setShowLogs(false)} />
            )}

            <ConfirmDialog
                open={confirmStop}
                title="Stop this server?"
                message={`Stopping "${name}" ends the current training run. Connected clients are disconnected and the round in progress is lost.`}
                confirmLabel="Stop server"
                cancelLabel="Cancel"
                danger
                onConfirm={() => {
                    setConfirmStop(false);
                    handleStop();
                }}
                onCancel={() => setConfirmStop(false)}
            />
        </div>
    );
}
