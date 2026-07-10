// =============================================================================
// FedLearn Frontend — Model Registry / Adapter Catalog (FE-11)
// =============================================================================
// Lists a project's versioned, content-addressed model artifacts and, for a
// selected artifact, shows its provenance, its parsed eval card, its lineage
// chain, and a download affordance for the immutable weights blob.
//
// Cookie auth only — every request rides the shared axios instance's
// `withCredentials`. No token handling here.

import { useEffect, useMemo, useState } from 'react';
import { Package, Download, Fingerprint, AlertCircle, Loader2, Clock, ScrollText } from 'lucide-react';
import { cn } from '../../lib/utils';
import * as api from '../../services/apiServices';
import { errorMessage } from '../../services/apiServices';
import type { Project } from '../../services/apiServices';
import * as registry from '../../services/artifactService';
import type { ArtifactDto, LineageNode } from '../../services/artifactService';
import { Card, Button, Select, Skeleton } from '../ui';
import { BrandMark } from '../brand';
import { PageHeader } from './PageHeader';
import { RegistryEvalCard } from './RegistryEvalCard';
import { RegistryLineage } from './RegistryLineage';

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

/** A small neutral badge — reused for kind / license chips. */
function Chip({ children }: { children: React.ReactNode }) {
    return (
        <span className="text-caption font-medium px-2.5 py-0.5 rounded-pill bg-surface-2 border border-hairline text-fg-muted">
            {children}
        </span>
    );
}

export function RegistryView() {
    const [projects, setProjects] = useState<Project[]>([]);
    const [loadingProjects, setLoadingProjects] = useState(true);
    const [selectedProjectId, setSelectedProjectId] = useState<string>('');

    const [artifacts, setArtifacts] = useState<ArtifactDto[]>([]);
    const [loadingArtifacts, setLoadingArtifacts] = useState(false);
    const [artifactsError, setArtifactsError] = useState('');

    const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null);
    const [lineage, setLineage] = useState<LineageNode[]>([]);
    const [loadingLineage, setLoadingLineage] = useState(false);

    const [downloadingId, setDownloadingId] = useState<string | null>(null);
    // Top-of-page banner: projects-load failure or a download failure.
    const [pageError, setPageError] = useState('');

    // Load the caller's projects once; default to the first one.
    useEffect(() => {
        (async () => {
            try {
                const res = await api.fetchProjects();
                const list = Array.isArray(res.data) ? res.data : [];
                setProjects(list);
                if (list.length > 0) setSelectedProjectId(list[0].id);
            } catch {
                setPageError('Failed to load your projects.');
            } finally {
                setLoadingProjects(false);
            }
        })();
    }, []);

    // (Re)load artifacts whenever the selected project changes.
    useEffect(() => {
        if (!selectedProjectId) return;
        let cancelled = false;
        setLoadingArtifacts(true);
        setArtifactsError('');
        setSelectedArtifactId(null);
        setLineage([]);
        setArtifacts([]);
        (async () => {
            try {
                const list = await registry.listArtifacts(selectedProjectId);
                if (!cancelled) setArtifacts(list);
            } catch {
                if (!cancelled) setArtifactsError('Failed to load artifacts for this project.');
            } finally {
                if (!cancelled) setLoadingArtifacts(false);
            }
        })();
        return () => {
            cancelled = true;
        };
    }, [selectedProjectId]);

    const selected = useMemo(
        () => artifacts.find((a) => a.id === selectedArtifactId) ?? null,
        [artifacts, selectedArtifactId],
    );

    const selectArtifact = async (artifact: ArtifactDto) => {
        setSelectedArtifactId(artifact.id);
        setLineage([]);
        setLoadingLineage(true);
        try {
            const nodes = await registry.getLineage(artifact.id);
            setLineage(nodes);
        } catch {
            // Lineage is a non-fatal add-on; an empty chain renders honestly.
            setLineage([]);
        } finally {
            setLoadingLineage(false);
        }
    };

    const handleDownload = async (artifact: ArtifactDto) => {
        setDownloadingId(artifact.id);
        setPageError('');
        try {
            const blob = await registry.downloadBlob(artifact.id);
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = `${shortSha(artifact.blobSha256)}.safetensors`;
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            URL.revokeObjectURL(url);
        } catch (e: unknown) {
            setPageError(errorMessage(e, 'Download failed. Please try again.'));
        } finally {
            setDownloadingId(null);
        }
    };

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
            <PageHeader
                title="Registry"
                subtitle="Versioned, content-addressed model artifacts and their lineage."
            />

            <div className="flex-1 overflow-y-auto px-6 md:px-10 py-8 bg-canvas">
                {pageError && (
                    <div className="mb-6 flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
                        <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                        {pageError}
                    </div>
                )}

                {loadingProjects ? (
                    <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] gap-6">
                        <Card padding="lg" className="flex flex-col gap-4">
                            <Skeleton className="h-5 w-40" />
                            <Skeleton className="h-24 w-full" />
                            <Skeleton className="h-24 w-full" />
                        </Card>
                        <Card padding="lg" className="flex flex-col gap-4">
                            <Skeleton className="h-5 w-32" />
                            <Skeleton className="h-40 w-full" />
                        </Card>
                    </div>
                ) : projects.length === 0 ? (
                    <div className="flex flex-col items-center justify-center text-center gap-5 mt-16 md:mt-24">
                        <div className="grid h-20 w-20 place-items-center rounded-card border border-hairline bg-surface-1">
                            <BrandMark size={48} />
                        </div>
                        <div className="max-w-sm">
                            <p className="text-h4 font-display text-fg">No projects yet</p>
                            <p className="text-body text-fg-muted mt-1.5">
                                Create a project and its model artifacts will show up here.
                            </p>
                        </div>
                    </div>
                ) : (
                    <div className="flex flex-col gap-6">
                        {/* Project selector */}
                        <label className="flex flex-col gap-1.5 max-w-sm">
                            <span className="text-caption uppercase tracking-wide font-semibold text-fg-muted">
                                Project
                            </span>
                            <Select
                                aria-label="Project"
                                value={selectedProjectId}
                                onChange={(e) => setSelectedProjectId(e.target.value)}
                            >
                                {projects.map((p) => (
                                    <option key={p.id} value={p.id}>
                                        {p.name}
                                    </option>
                                ))}
                            </Select>
                        </label>

                        {artifactsError && (
                            <div className="flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
                                <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                                {artifactsError}
                            </div>
                        )}

                        {loadingArtifacts ? (
                            <div className="flex flex-col gap-3">
                                {[0, 1, 2].map((i) => (
                                    <Card key={i} padding="md" className="flex flex-col gap-2">
                                        <Skeleton className="h-4 w-24" />
                                        <Skeleton className="h-4 w-48" />
                                    </Card>
                                ))}
                            </div>
                        ) : !artifactsError && artifacts.length === 0 ? (
                            <div className="flex flex-col items-center justify-center text-center gap-5 mt-10 md:mt-16">
                                <div className="grid h-20 w-20 place-items-center rounded-card border border-hairline bg-surface-1">
                                    <Package className="w-9 h-9 text-fg-subtle" strokeWidth={1.25} />
                                </div>
                                <div className="max-w-sm">
                                    <p className="text-h4 font-display text-fg">No artifacts yet</p>
                                    <p className="text-body text-fg-muted mt-1.5">
                                        This project hasn't published any model artifacts. They appear here
                                        once a run produces a checkpoint or adapter.
                                    </p>
                                </div>
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] gap-6">
                                {/* ── Artifact list ── */}
                                <div className="flex flex-col gap-3">
                                    {artifacts.map((a) => {
                                        const isActive = a.id === selectedArtifactId;
                                        return (
                                            <Card
                                                key={a.id}
                                                padding="md"
                                                interactive
                                                role="button"
                                                tabIndex={0}
                                                aria-pressed={isActive}
                                                aria-label={`View artifact ${a.kind} ${shortSha(a.blobSha256)}`}
                                                onClick={() => selectArtifact(a)}
                                                onKeyDown={(e) => {
                                                    if (e.key === 'Enter' || e.key === ' ') {
                                                        e.preventDefault();
                                                        selectArtifact(a);
                                                    }
                                                }}
                                                className={cn(
                                                    'flex flex-col gap-2',
                                                    isActive && 'border-accent/50 bg-surface-2',
                                                )}
                                            >
                                                <div className="flex items-center justify-between gap-2">
                                                    <Chip>{a.kind}</Chip>
                                                    <span className="font-mono text-caption text-fg-subtle flex items-center gap-1">
                                                        <Fingerprint className="w-3 h-3" strokeWidth={1.5} />
                                                        {shortSha(a.blobSha256)}
                                                    </span>
                                                </div>
                                                <div className="flex items-center justify-between gap-2">
                                                    <span className="text-label text-fg truncate">
                                                        {a.recipeKey ?? 'No recipe'}
                                                    </span>
                                                    <span className="text-caption text-fg-muted flex items-center gap-1 flex-shrink-0">
                                                        <Clock className="w-3 h-3" strokeWidth={1.5} />
                                                        {formatDate(a.createdAt)}
                                                    </span>
                                                </div>
                                                {a.licenseTag && (
                                                    <div>
                                                        <Chip>{a.licenseTag}</Chip>
                                                    </div>
                                                )}
                                            </Card>
                                        );
                                    })}
                                </div>

                                {/* ── Detail panel ── */}
                                <div>
                                    {!selected ? (
                                        <Card padding="lg" className="flex flex-1 flex-col items-center justify-center text-center gap-2 min-h-[240px]">
                                            <Package className="w-8 h-8 text-fg-subtle" strokeWidth={1.25} />
                                            <p className="text-body text-fg-muted">
                                                Select an artifact to see its provenance, eval card, and lineage.
                                            </p>
                                        </Card>
                                    ) : (
                                        <Card padding="lg" className="flex flex-col gap-5">
                                            {/* Header */}
                                            <div className="flex items-start gap-3">
                                                <span className="icon-tile flex-shrink-0">
                                                    <Package strokeWidth={1.5} className="w-5 h-5" />
                                                </span>
                                                <div className="min-w-0 flex-1">
                                                    <div className="flex items-center gap-2">
                                                        <h3 className="text-h4 font-display text-fg">{selected.kind}</h3>
                                                    </div>
                                                    <p className="text-caption font-mono text-fg-subtle break-all mt-0.5">
                                                        {selected.blobSha256}
                                                    </p>
                                                </div>
                                                <Button
                                                    variant="secondary"
                                                    size="sm"
                                                    onClick={() => handleDownload(selected)}
                                                    disabled={downloadingId === selected.id}
                                                    className="flex-shrink-0"
                                                >
                                                    {downloadingId === selected.id ? (
                                                        <>
                                                            <Loader2 className="w-4 h-4 animate-spin" strokeWidth={2} /> Downloading…
                                                        </>
                                                    ) : (
                                                        <>
                                                            <Download className="w-4 h-4" strokeWidth={1.5} /> Download
                                                        </>
                                                    )}
                                                </Button>
                                            </div>

                                            {/* Provenance / metadata */}
                                            <div className="flex flex-col gap-2">
                                                <span className="text-caption uppercase tracking-wide font-semibold text-fg-muted flex items-center gap-1.5">
                                                    <ScrollText className="w-3.5 h-3.5" strokeWidth={1.5} /> Provenance
                                                </span>
                                                <dl className="rounded-card border border-hairline bg-surface-1 divide-y divide-hairline">
                                                    {(
                                                        [
                                                            ['Recipe', selected.recipeKey ?? '—'],
                                                            ['Base model', selected.baseModelRef ?? '—'],
                                                            ['License', selected.licenseTag ?? '—'],
                                                            ['Run', selected.runId ?? '—'],
                                                            ['Created', formatDate(selected.createdAt)],
                                                            ['Created by', selected.createdBy != null ? String(selected.createdBy) : '—'],
                                                        ] as const
                                                    ).map(([label, value]) => (
                                                        <div key={label} className="flex items-start justify-between gap-4 px-4 py-2.5">
                                                            <dt className="text-label text-fg-muted">{label}</dt>
                                                            <dd className="text-label text-fg text-right break-all">{value}</dd>
                                                        </div>
                                                    ))}
                                                </dl>
                                            </div>

                                            <RegistryEvalCard evalCardJson={selected.evalCardJson} />
                                            <RegistryLineage nodes={lineage} loading={loadingLineage} />
                                        </Card>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
