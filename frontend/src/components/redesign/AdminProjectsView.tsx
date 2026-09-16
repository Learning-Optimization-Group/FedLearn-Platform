// =============================================================================
// FedLearn Frontend — AdminProjectsView (role: PLATFORM_ADMIN)
// =============================================================================
// Search-first, server-paginated directory of EVERY project on the platform,
// backed by GET /admin/projects/search. Filter + page state lives in the URL
// (useSearchParams) so any view is shareable and the dashboard can deep-link
// (e.g. ?status=RUNNING). Row click drills into /admin/projects/:id, carrying
// the current URL so the detail's back link restores this exact view.

import { useEffect, useState } from 'react';
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import { AlertCircle, FolderKanban, Search } from 'lucide-react';
import * as api from '../../services/apiServices';
import {
    errorMessage,
    type AdminProject,
    type Paged,
    type Visibility,
} from '../../services/apiServices';
import { Button, Card, Input, Select, Skeleton, StatusPill, toStatusKind } from '../ui';
import { PageHeader } from './PageHeader';

const PAGE_SIZE = 25;
const SEARCH_DEBOUNCE_MS = 300;

const STATUS_OPTIONS = ['INITIALIZING', 'CREATED', 'RUNNING', 'STOPPED', 'COMPLETED', 'FAILED'];
const VISIBILITY_OPTIONS: Visibility[] = ['PUBLIC', 'RESTRICTED', 'PRIVATE'];

// SectionLabel-styled table header cell (the one uppercase micro-label).
const th = 'px-4 py-2.5 text-left text-[11px] font-semibold uppercase tracking-[0.08em] text-fg-muted';
const td = 'px-4 py-3 text-body text-fg align-middle';

export function AdminProjectsView() {
    const [searchParams, setSearchParams] = useSearchParams();
    const location = useLocation();
    const navigate = useNavigate();

    // URL is the single source of truth for the view state. `page` is 1-based
    // in the URL for readability; the API takes 0-based pages.
    const q = searchParams.get('q') ?? '';
    const status = searchParams.get('status') ?? '';
    const visibility = searchParams.get('visibility') ?? '';
    const page = Math.max(1, parseInt(searchParams.get('page') ?? '1', 10) || 1);

    const [searchInput, setSearchInput] = useState(q);
    const [data, setData] = useState<Paged<AdminProject> | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    const updateParams = (patch: Record<string, string | null>, replace = false) => {
        setSearchParams(
            (prev) => {
                const next = new URLSearchParams(prev);
                for (const [key, value] of Object.entries(patch)) {
                    if (value === null || value === '') next.delete(key);
                    else next.set(key, value);
                }
                return next;
            },
            { replace },
        );
    };

    // Keep the box in sync when the URL changes underneath us (back/forward,
    // deep links). While typing this is a no-op: the debounce below is what
    // wrote q, and it always writes the box's own value.
    useEffect(() => {
        setSearchInput(q);
    }, [q]);

    // Debounced search → URL. Every q change resets to page 1 (replace, so
    // each keystroke doesn't grow the history stack).
    useEffect(() => {
        const timer = setTimeout(() => {
            if (searchInput === q) return;
            updateParams({ q: searchInput || null, page: null }, true);
        }, SEARCH_DEBOUNCE_MS);
        return () => clearTimeout(timer);
        // updateParams is re-created per render but only touches setSearchParams.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [searchInput, q]);

    // Fetch whenever the URL state changes.
    useEffect(() => {
        let cancelled = false;
        setIsLoading(true);
        api.searchAdminProjects({
            q: q || undefined,
            status: status || undefined,
            visibility: (visibility || undefined) as Visibility | undefined,
            page: page - 1,
            size: PAGE_SIZE,
        })
            .then((res) => {
                if (cancelled) return;
                setData(res.data);
                setError('');
            })
            .catch((err) => {
                if (cancelled) return;
                setError(errorMessage(err, 'Could not load projects.'));
            })
            .finally(() => {
                if (!cancelled) setIsLoading(false);
            });
        return () => {
            cancelled = true;
        };
    }, [q, status, visibility, page]);

    const openProject = (projectId: string) => {
        navigate(`/admin/projects/${projectId}`, {
            state: { from: `${location.pathname}${location.search}` },
        });
    };

    const items = data?.items ?? [];
    const total = data?.total ?? 0;
    const rangeStart = total === 0 ? 0 : (page - 1) * PAGE_SIZE + 1;
    const rangeEnd = total === 0 ? 0 : rangeStart + items.length - 1;
    const hasPrev = page > 1;
    const hasNext = page * PAGE_SIZE < total;

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
            <PageHeader title="Projects" subtitle="Every project on the platform." />

            <div className="flex-1 overflow-y-auto">
                <div className="mx-auto w-full max-w-[1400px] px-6 py-6 md:px-10 flex flex-col gap-4">
                    {error && (
                        <div className="flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
                            <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                            {error}
                        </div>
                    )}

                    {/* Toolbar: search + filters. All of it round-trips through
                        the URL so the view is shareable. */}
                    <div className="flex flex-col gap-3 md:flex-row md:items-center">
                        <div className="relative w-full md:max-w-sm">
                            <Search
                                className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-fg-muted"
                                strokeWidth={1.5}
                            />
                            <Input
                                type="search"
                                value={searchInput}
                                onChange={(e) => setSearchInput(e.target.value)}
                                placeholder="Search by project or owner…"
                                aria-label="Search projects by name or owner"
                                className="pl-9"
                            />
                        </div>
                        <div className="flex gap-3">
                            <div className="w-44">
                                <Select
                                    value={status}
                                    aria-label="Filter by status"
                                    onChange={(e) => updateParams({ status: e.target.value || null, page: null })}
                                >
                                    <option value="">All statuses</option>
                                    {STATUS_OPTIONS.map((s) => (
                                        <option key={s} value={s}>
                                            {s.charAt(0) + s.slice(1).toLowerCase()}
                                        </option>
                                    ))}
                                </Select>
                            </div>
                            <div className="w-44">
                                <Select
                                    value={visibility}
                                    aria-label="Filter by visibility"
                                    onChange={(e) => updateParams({ visibility: e.target.value || null, page: null })}
                                >
                                    <option value="">All visibility</option>
                                    {VISIBILITY_OPTIONS.map((v) => (
                                        <option key={v} value={v}>
                                            {v.charAt(0) + v.slice(1).toLowerCase()}
                                        </option>
                                    ))}
                                </Select>
                            </div>
                        </div>
                    </div>

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
                                    {isLoading ? (
                                        [0, 1, 2, 3, 4].map((i) => (
                                            <tr key={i} className="border-b border-hairline last:border-0">
                                                <td className={td}><Skeleton className="h-4 w-40" /></td>
                                                <td className={td}><Skeleton className="h-4 w-24" /></td>
                                                <td className={td}><Skeleton className="h-4 w-20" /></td>
                                                <td className={td}><Skeleton className="h-4 w-8" /></td>
                                                <td className={td}><Skeleton className="h-4 w-20" /></td>
                                                <td className={td}><Skeleton className="h-5 w-24 rounded-pill" /></td>
                                            </tr>
                                        ))
                                    ) : items.length === 0 ? (
                                        <tr>
                                            <td className={`${td} text-fg-muted`} colSpan={6}>
                                                <span className="inline-flex items-center gap-2">
                                                    <FolderKanban className="h-4 w-4" strokeWidth={1.5} />
                                                    {q || status || visibility
                                                        ? 'No projects match these filters.'
                                                        : 'No projects yet.'}
                                                </span>
                                            </td>
                                        </tr>
                                    ) : (
                                        items.map((p) => (
                                            <tr
                                                key={p.id}
                                                role="link"
                                                tabIndex={0}
                                                aria-label={`Open project ${p.name}`}
                                                onClick={() => openProject(p.id)}
                                                onKeyDown={(e) => {
                                                    if (e.key === 'Enter' || e.key === ' ') {
                                                        e.preventDefault();
                                                        openProject(p.id);
                                                    }
                                                }}
                                                className="border-b border-hairline last:border-0 cursor-pointer transition-colors hover:bg-surface-2 focus-visible:outline-none focus-visible:bg-surface-2"
                                            >
                                                <td className={`${td} font-medium`}>{p.name}</td>
                                                <td className={`${td} text-fg-muted`}>{p.ownerUsername}</td>
                                                <td className={`${td} text-fg-muted`}>{p.modelType}</td>
                                                <td className={`${td} font-mono tabular-nums`}>{p.participantCount}</td>
                                                <td className={`${td} text-fg-muted`}>{p.visibility}</td>
                                                <td className={td}>
                                                    <StatusPill status={toStatusKind(p.status)}>{p.status}</StatusPill>
                                                </td>
                                            </tr>
                                        ))
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </Card>

                    {/* Pager */}
                    <div className="flex items-center justify-between">
                        <span className="text-label text-fg-muted font-mono tabular-nums">
                            {total === 0 ? '0 of 0' : `${rangeStart}–${rangeEnd} of ${total}`}
                        </span>
                        <div className="flex items-center gap-2">
                            <Button
                                variant="secondary"
                                size="sm"
                                disabled={!hasPrev || isLoading}
                                onClick={() => updateParams({ page: page - 1 <= 1 ? null : String(page - 1) })}
                            >
                                Prev
                            </Button>
                            <Button
                                variant="secondary"
                                size="sm"
                                disabled={!hasNext || isLoading}
                                onClick={() => updateParams({ page: String(page + 1) })}
                            >
                                Next
                            </Button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
