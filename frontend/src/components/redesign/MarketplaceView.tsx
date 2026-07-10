// =============================================================================
// FedLearn Frontend — Adapter Marketplace (FE-12)
// =============================================================================
// Browses PUBLISHED LORA_ADAPTER bundles the caller may see. Visibility is
// org-scoped server-side, so an empty feed is a normal, honest result — not an
// error. Each entry surfaces its recipe, its content address, its license
// (marketplace-load-bearing — shown prominently), its provenance base model,
// and its parsed eval card (via the shared RegistryEvalCard).
//
// Cookie auth only — every request rides the shared axios instance's
// `withCredentials`. No token handling here.

import { useEffect, useState } from 'react';
import { Store, Fingerprint, AlertCircle, Clock, Scale, GitBranch } from 'lucide-react';
import * as registry from '../../services/artifactService';
import type { ArtifactDto } from '../../services/artifactService';
import { Card, Skeleton } from '../ui';
import { PageHeader } from './PageHeader';
import { RegistryEvalCard } from './RegistryEvalCard';

/** Short, human-scannable form of a 64-hex content address. */
function shortSha(sha: string): string {
    return sha.length > 12 ? sha.slice(0, 12) : sha;
}

/** Format an ISO instant for display; falls back to the raw string if unparseable. */
function formatDate(iso: string | null): string {
    if (!iso) return '—';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString();
}

/** A small neutral badge — reused for the kind chip. */
function Chip({ children }: { children: React.ReactNode }) {
    return (
        <span className="text-caption font-medium px-2.5 py-0.5 rounded-pill bg-surface-2 border border-hairline text-fg-muted">
            {children}
        </span>
    );
}

/** License is marketplace-load-bearing: give it a prominent, accent-toned badge. */
function LicenseBadge({ licenseTag }: { licenseTag: string | null }) {
    return (
        <span
            className="inline-flex items-center gap-1.5 text-label font-semibold px-3 py-1 rounded-pill bg-accent/10 border border-accent/30 text-accent"
            aria-label={`License ${licenseTag ?? 'unspecified'}`}
        >
            <Scale className="w-3.5 h-3.5" strokeWidth={1.75} />
            {licenseTag ?? 'License unspecified'}
        </span>
    );
}

function MarketplaceCard({ adapter }: { adapter: ArtifactDto }) {
    return (
        <Card padding="lg" className="flex flex-col gap-4">
            {/* Header */}
            <div className="flex items-start gap-3">
                <span className="icon-tile flex-shrink-0">
                    <Store strokeWidth={1.5} className="w-5 h-5" />
                </span>
                <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                        <h3 className="text-h4 font-display text-fg truncate">
                            {adapter.recipeKey ?? 'LORA adapter'}
                        </h3>
                        <Chip>{adapter.kind}</Chip>
                    </div>
                    <span className="font-mono text-caption text-fg-subtle flex items-center gap-1 mt-1">
                        <Fingerprint className="w-3 h-3" strokeWidth={1.5} />
                        {shortSha(adapter.blobSha256)}
                    </span>
                </div>
            </div>

            {/* License — surfaced prominently. */}
            <div>
                <LicenseBadge licenseTag={adapter.licenseTag} />
            </div>

            {/* Provenance + published time */}
            <dl className="rounded-card border border-hairline bg-surface-1 divide-y divide-hairline">
                <div className="flex items-start justify-between gap-4 px-4 py-2.5">
                    <dt className="text-label text-fg-muted flex items-center gap-1.5">
                        <GitBranch className="w-3.5 h-3.5" strokeWidth={1.5} /> Base model
                    </dt>
                    <dd className="text-label text-fg text-right break-all">
                        {adapter.baseModelRef ?? '—'}
                    </dd>
                </div>
                <div className="flex items-start justify-between gap-4 px-4 py-2.5">
                    <dt className="text-label text-fg-muted flex items-center gap-1.5">
                        <Clock className="w-3.5 h-3.5" strokeWidth={1.5} /> Published
                    </dt>
                    <dd className="text-label text-fg text-right break-all">
                        {formatDate(adapter.publishedAt)}
                    </dd>
                </div>
            </dl>

            <RegistryEvalCard evalCardJson={adapter.evalCardJson} />
        </Card>
    );
}

export function MarketplaceView() {
    const [adapters, setAdapters] = useState<ArtifactDto[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const list = await registry.listMarketplace();
                if (!cancelled) setAdapters(list);
            } catch {
                if (!cancelled) setError('Failed to load the adapter marketplace.');
            } finally {
                if (!cancelled) setLoading(false);
            }
        })();
        return () => {
            cancelled = true;
        };
    }, []);

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
            <PageHeader
                title="Marketplace"
                subtitle="Published LoRA adapters you can browse across your organization."
            />

            <div className="flex-1 overflow-y-auto px-6 md:px-10 py-8 bg-canvas">
                {error && (
                    <div className="mb-6 flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
                        <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                        {error}
                    </div>
                )}

                {loading ? (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {[0, 1].map((i) => (
                            <Card key={i} padding="lg" className="flex flex-col gap-4">
                                <div className="flex items-center gap-3">
                                    <Skeleton className="h-11 w-11 rounded-xl" />
                                    <Skeleton className="h-5 w-40" />
                                </div>
                                <Skeleton className="h-6 w-28" />
                                <Skeleton className="h-24 w-full" />
                            </Card>
                        ))}
                    </div>
                ) : !error && adapters.length === 0 ? (
                    <div className="flex flex-col items-center justify-center text-center gap-5 mt-16 md:mt-24">
                        <div className="grid h-20 w-20 place-items-center rounded-card border border-hairline bg-surface-1">
                            <Store className="w-9 h-9 text-fg-subtle" strokeWidth={1.25} />
                        </div>
                        <div className="max-w-sm">
                            <p className="text-h4 font-display text-fg">No published adapters yet</p>
                            <p className="text-body text-fg-muted mt-1.5">
                                When an owner publishes a LoRA adapter to the marketplace, it shows up
                                here for your organization to browse.
                            </p>
                        </div>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {adapters.map((a) => (
                            <MarketplaceCard key={a.id} adapter={a} />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
