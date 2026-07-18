// =============================================================================
// FedLearn Frontend — Audit Log explorer (role: PLATFORM_ADMIN)
// =============================================================================
// Paginated, filterable view over /api/admin/audit-events (newest first,
// 50/page). Filter + page state is URL-encoded so a filtered view can be
// shared/bookmarked; the three text filters are debounced before committing to
// the URL, date bounds commit immediately. Each row expands in place to show
// the event's pretty-printed metadata JSON in a code well.

import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
    ScrollText,
    AlertCircle,
    ChevronRight,
    ChevronLeft,
    Copy,
    Check,
} from 'lucide-react';
import * as api from '../../services/apiServices';
import { errorMessage, type AuditEventItem, type Paged } from '../../services/apiServices';
import { Card, Button, Input, Skeleton } from '../ui';
import { PageHeader } from './PageHeader';
import { cn } from '../../lib/utils';

const PAGE_SIZE = 50;
const DEBOUNCE_MS = 350;

// Table cell styles — same vocabulary as the other admin tables.
const th = 'px-4 py-2.5 text-left text-[11px] font-semibold uppercase tracking-[0.08em] text-fg-muted';
const td = 'px-4 py-3 text-body text-fg align-middle';

// ─── Time formatting ─────────────────────────────────────────────────────

/** Coarse relative form ("3h ago"); the exact instant lives in the title attr. */
function formatRelative(iso: string): string {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    const seconds = Math.round((Date.now() - d.getTime()) / 1000);
    if (seconds < 45) return 'just now';
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.round(hours / 24);
    if (days < 30) return `${days}d ago`;
    return d.toLocaleDateString();
}

function formatExact(iso: string): string {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

/**
 * A native date input's value (yyyy-mm-dd, local) → ISO-8601 instant for the
 * API. `endOfDay` makes the "to" bound inclusive of its whole day.
 */
function dateToInstant(value: string, endOfDay: boolean): string | undefined {
    if (!value) return undefined;
    const d = new Date(`${value}T${endOfDay ? '23:59:59.999' : '00:00:00.000'}`);
    return Number.isNaN(d.getTime()) ? undefined : d.toISOString();
}

/** Pretty-print the metadata payload; non-JSON falls back to the raw string. */
function prettyMetadata(metadata?: string): string {
    if (!metadata) return 'No metadata recorded for this event.';
    try {
        return JSON.stringify(JSON.parse(metadata), null, 2);
    } catch {
        return metadata;
    }
}

// ─── Small pieces ────────────────────────────────────────────────────────

/** Icon-only copy affordance for target ids. */
function CopyIdButton({ value }: { value: string }) {
    const [copied, setCopied] = useState(false);

    useEffect(() => {
        if (!copied) return;
        const t = setTimeout(() => setCopied(false), 1500);
        return () => clearTimeout(t);
    }, [copied]);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(value);
            setCopied(true);
        } catch {
            /* no-op */
        }
    };

    return (
        <button
            type="button"
            onClick={handleCopy}
            title={copied ? 'Copied' : 'Copy id'}
            aria-label={copied ? 'Copied' : `Copy id ${value}`}
            className={cn(
                'inline-flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md',
                'text-fg-subtle hover:text-fg hover:bg-surface-2',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent',
                copied && 'text-success hover:text-success',
            )}
        >
            {copied
                ? <Check strokeWidth={1.5} className="h-3.5 w-3.5" />
                : <Copy strokeWidth={1.5} className="h-3.5 w-3.5" />}
        </button>
    );
}

function AuditRow({
    event,
    expanded,
    onToggle,
}: {
    event: AuditEventItem;
    expanded: boolean;
    onToggle: () => void;
}) {
    return (
        <>
            <tr className={cn('border-b border-hairline', expanded ? 'bg-surface-2/50' : 'last:border-0')}>
                <td className="w-10 pl-3 py-3 align-middle">
                    <button
                        type="button"
                        onClick={onToggle}
                        aria-expanded={expanded}
                        aria-label={expanded ? 'Hide event details' : 'Show event details'}
                        className={cn(
                            'inline-flex h-6 w-6 items-center justify-center rounded-md',
                            'text-fg-muted hover:text-fg hover:bg-surface-2',
                            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent',
                        )}
                    >
                        <ChevronRight
                            strokeWidth={1.5}
                            className={cn('h-4 w-4 transition-transform duration-[140ms]', expanded && 'rotate-90')}
                        />
                    </button>
                </td>
                <td className={cn(td, 'whitespace-nowrap')}>
                    <span title={formatExact(event.occurredAt)} className="text-label text-fg">
                        {formatRelative(event.occurredAt)}
                    </span>
                </td>
                <td className={td}>
                    <span className="text-label text-fg">{event.actorUsername ?? '—'}</span>
                </td>
                <td className={td}>
                    <span className="font-mono text-caption text-fg">{event.action}</span>
                </td>
                <td className={td}>
                    {event.targetType || event.targetId ? (
                        <span className="flex items-center gap-1.5 min-w-0">
                            {event.targetType && (
                                <span className="text-caption text-fg-muted flex-shrink-0">{event.targetType}</span>
                            )}
                            {event.targetId && (
                                <>
                                    <span
                                        title={event.targetId}
                                        className="font-mono text-caption text-fg truncate max-w-[160px]"
                                    >
                                        {event.targetId}
                                    </span>
                                    <CopyIdButton value={event.targetId} />
                                </>
                            )}
                        </span>
                    ) : (
                        <span className="text-label text-fg-subtle">—</span>
                    )}
                </td>
                <td className={cn(td, 'whitespace-nowrap')}>
                    <span className="font-mono text-caption text-fg-muted">{event.requestIp ?? '—'}</span>
                </td>
            </tr>
            {expanded && (
                <tr className="border-b border-hairline last:border-0">
                    <td colSpan={6} className="px-4 pb-4 pt-1">
                        <pre className="bg-code-well border border-hairline rounded-lg font-mono text-label text-code-fg p-4 overflow-x-auto whitespace-pre-wrap break-all">
                            {prettyMetadata(event.metadata)}
                        </pre>
                    </td>
                </tr>
            )}
        </>
    );
}

// ─── View ────────────────────────────────────────────────────────────────

export function AuditLogView() {
    const [searchParams, setSearchParams] = useSearchParams();

    // Text filters are typed into local drafts, then committed to the URL after
    // a pause — the URL (not the drafts) is what drives fetching.
    const [drafts, setDrafts] = useState({
        actor: searchParams.get('actor') ?? '',
        action: searchParams.get('action') ?? '',
        targetType: searchParams.get('targetType') ?? '',
    });

    const [data, setData] = useState<Paged<AuditEventItem> | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [expandedId, setExpandedId] = useState<string | null>(null);

    // External URL changes (back/forward, shared link) re-seed the drafts. Our
    // own debounced commit lands here too, but then URL === drafts and the
    // update is a no-op, so typing is never clobbered.
    useEffect(() => {
        const fromUrl = {
            actor: searchParams.get('actor') ?? '',
            action: searchParams.get('action') ?? '',
            targetType: searchParams.get('targetType') ?? '',
        };
        setDrafts((d) =>
            d.actor === fromUrl.actor && d.action === fromUrl.action && d.targetType === fromUrl.targetType
                ? d
                : fromUrl,
        );
    }, [searchParams]);

    // Debounced commit: drafts → URL (resetting to page 0).
    useEffect(() => {
        const t = setTimeout(() => {
            const unchanged =
                (searchParams.get('actor') ?? '') === drafts.actor &&
                (searchParams.get('action') ?? '') === drafts.action &&
                (searchParams.get('targetType') ?? '') === drafts.targetType;
            if (unchanged) return;
            setSearchParams(
                (prev) => {
                    const next = new URLSearchParams(prev);
                    for (const key of ['actor', 'action', 'targetType'] as const) {
                        if (drafts[key]) next.set(key, drafts[key]);
                        else next.delete(key);
                    }
                    next.delete('page');
                    return next;
                },
                { replace: true },
            );
        }, DEBOUNCE_MS);
        return () => clearTimeout(t);
    }, [drafts, searchParams, setSearchParams]);

    /** Set (or clear) one URL param immediately — dates and paging. */
    const setParam = (key: string, value: string, resetPage = true) => {
        setSearchParams((prev) => {
            const next = new URLSearchParams(prev);
            if (value) next.set(key, value);
            else next.delete(key);
            if (resetPage && key !== 'page') next.delete('page');
            return next;
        });
    };

    const fromDate = searchParams.get('from') ?? '';
    const toDate = searchParams.get('to') ?? '';
    const page = Math.max(0, Number.parseInt(searchParams.get('page') ?? '0', 10) || 0);
    const hasFilters = Boolean(
        searchParams.get('actor') || searchParams.get('action') || searchParams.get('targetType') || fromDate || toDate,
    ) || Boolean(drafts.actor || drafts.action || drafts.targetType);

    const queryParams = useMemo(
        () => ({
            actor: searchParams.get('actor') ?? undefined,
            action: searchParams.get('action') ?? undefined,
            targetType: searchParams.get('targetType') ?? undefined,
            from: dateToInstant(searchParams.get('from') ?? '', false),
            to: dateToInstant(searchParams.get('to') ?? '', true),
            page,
            size: PAGE_SIZE,
        }),
        [searchParams, page],
    );

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        api.fetchAuditEvents(queryParams)
            .then((res) => {
                if (cancelled) return;
                setData(res.data);
                setError('');
                setExpandedId(null);
            })
            .catch((err) => {
                if (!cancelled) setError(errorMessage(err, 'Could not load the audit log.'));
            })
            .finally(() => {
                if (!cancelled) setLoading(false);
            });
        return () => {
            cancelled = true;
        };
    }, [queryParams]);

    const clearFilters = () => {
        setDrafts({ actor: '', action: '', targetType: '' });
        setSearchParams({}, { replace: true });
    };

    const items = data?.items ?? [];
    const total = data?.total ?? 0;
    const rangeStart = total === 0 ? 0 : page * PAGE_SIZE + 1;
    const rangeEnd = page * PAGE_SIZE + items.length;
    const hasNext = rangeEnd < total;

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
            <PageHeader title="Audit log" subtitle="Every audited platform action, newest first." />

            <div className="flex-1 overflow-y-auto">
                <div className="mx-auto w-full max-w-[1400px] px-6 py-6 md:px-10 reveal">
                    {/* Filter row */}
                    <div className="mb-4 flex flex-wrap items-center gap-3">
                        <Input
                            value={drafts.actor}
                            onChange={(e) => setDrafts((d) => ({ ...d, actor: e.target.value }))}
                            placeholder="Actor username"
                            aria-label="Filter by actor username"
                            className="w-44"
                        />
                        <Input
                            value={drafts.action}
                            onChange={(e) => setDrafts((d) => ({ ...d, action: e.target.value }))}
                            placeholder="Action"
                            aria-label="Filter by action"
                            className="w-44"
                        />
                        <Input
                            value={drafts.targetType}
                            onChange={(e) => setDrafts((d) => ({ ...d, targetType: e.target.value }))}
                            placeholder="Target type"
                            aria-label="Filter by target type"
                            className="w-44"
                        />
                        <Input
                            type="date"
                            value={fromDate}
                            onChange={(e) => setParam('from', e.target.value)}
                            aria-label="From date"
                            className="w-40"
                        />
                        <Input
                            type="date"
                            value={toDate}
                            onChange={(e) => setParam('to', e.target.value)}
                            aria-label="To date"
                            className="w-40"
                        />
                        <Button variant="ghost" size="sm" onClick={clearFilters} disabled={!hasFilters}>
                            Clear filters
                        </Button>
                    </div>

                    {error && (
                        <div className="mb-6 flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
                            <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                            {error}
                        </div>
                    )}

                    {loading ? (
                        <Card padding="none" className="overflow-hidden">
                            <div className="flex flex-col gap-3 p-4">
                                {[0, 1, 2, 3, 4, 5].map((i) => (
                                    <Skeleton key={i} className="h-8 w-full" />
                                ))}
                            </div>
                        </Card>
                    ) : !error && items.length === 0 ? (
                        <div className="flex flex-col items-center justify-center text-center gap-4 pt-16 md:pt-24">
                            <div className="grid h-12 w-12 place-items-center rounded-pill bg-surface-2 text-fg-muted">
                                <ScrollText className="h-6 w-6" strokeWidth={1.5} />
                            </div>
                            <div className="max-w-sm">
                                <p className="text-h4 font-semibold text-fg">No audit events</p>
                                <p className="text-caption text-fg-muted mt-1">
                                    {hasFilters
                                        ? 'No events match these filters. Try widening the date range or clearing a filter.'
                                        : 'Audited platform actions will show up here as they happen.'}
                                </p>
                            </div>
                        </div>
                    ) : !error ? (
                        <Card padding="none" className="overflow-hidden">
                            <div className="overflow-x-auto">
                                <table className="w-full border-collapse">
                                    <thead className="border-b border-hairline bg-surface-2">
                                        <tr>
                                            <th className="w-10 pl-3 py-2.5">
                                                <span className="sr-only">Details</span>
                                            </th>
                                            <th className={th}>Time</th>
                                            <th className={th}>Actor</th>
                                            <th className={th}>Action</th>
                                            <th className={th}>Target</th>
                                            <th className={th}>IP</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {items.map((event) => (
                                            <AuditRow
                                                key={event.id}
                                                event={event}
                                                expanded={expandedId === event.id}
                                                onToggle={() =>
                                                    setExpandedId((cur) => (cur === event.id ? null : event.id))
                                                }
                                            />
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            {/* Pagination footer */}
                            <div className="flex items-center justify-between gap-4 border-t border-hairline px-4 py-3">
                                <span className="text-caption text-fg-muted">
                                    Showing {rangeStart}–{rangeEnd} of {total}
                                </span>
                                <div className="flex items-center gap-2">
                                    <Button
                                        variant="secondary"
                                        size="sm"
                                        disabled={page === 0}
                                        onClick={() => setParam('page', page > 1 ? String(page - 1) : '', false)}
                                    >
                                        <ChevronLeft strokeWidth={1.5} className="w-3.5 h-3.5" />
                                        Previous
                                    </Button>
                                    <Button
                                        variant="secondary"
                                        size="sm"
                                        disabled={!hasNext}
                                        onClick={() => setParam('page', String(page + 1), false)}
                                    >
                                        Next
                                        <ChevronRight strokeWidth={1.5} className="w-3.5 h-3.5" />
                                    </Button>
                                </div>
                            </div>
                        </Card>
                    ) : null}
                </div>
            </div>
        </div>
    );
}

export default AuditLogView;
