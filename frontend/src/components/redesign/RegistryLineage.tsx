// =============================================================================
// FedLearn Frontend — Registry lineage chain (FE-11)
// =============================================================================
// Renders an artifact's ancestor chain as an ordered, top-to-bottom list. The
// backend returns the chain root-most first; we render it verbatim.

import { GitBranch, Loader2 } from 'lucide-react';
import type { LineageNode } from '../../services/artifactService';

interface RegistryLineageProps {
    nodes: LineageNode[];
    loading: boolean;
}

/** Short, human-scannable form of a 64-hex content address. */
function shortSha(sha: string): string {
    return sha.length > 12 ? sha.slice(0, 12) : sha;
}

export function RegistryLineage({ nodes, loading }: RegistryLineageProps) {
    return (
        <div className="flex flex-col gap-2">
            <span className="text-caption uppercase tracking-wide font-semibold text-fg-muted flex items-center gap-1.5">
                <GitBranch className="w-3.5 h-3.5" strokeWidth={1.5} /> Lineage
            </span>
            {loading ? (
                <div className="flex items-center gap-2 text-label text-fg-muted">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" strokeWidth={2} /> Loading lineage…
                </div>
            ) : nodes.length === 0 ? (
                <div className="rounded-card border border-hairline bg-surface-1 px-4 py-3 text-label text-fg-muted">
                    No ancestors — this is a root artifact.
                </div>
            ) : (
                <ol className="flex flex-col gap-2">
                    {nodes.map((node, i) => (
                        <li
                            key={node.id}
                            className="flex items-center gap-3 rounded-card border border-hairline bg-surface-1 px-4 py-2.5"
                        >
                            <span className="grid h-6 w-6 flex-shrink-0 place-items-center rounded-pill bg-surface-2 border border-hairline text-caption font-mono tabular-nums text-fg-muted">
                                {i + 1}
                            </span>
                            <span className="text-caption font-medium px-2 py-0.5 rounded-pill bg-surface-2 border border-hairline text-fg-muted">
                                {node.kind}
                            </span>
                            <span className="font-mono text-caption text-fg-subtle truncate">
                                {shortSha(node.sha256)}
                            </span>
                            {node.licenseTag && (
                                <span className="ml-auto text-caption text-fg-muted">{node.licenseTag}</span>
                            )}
                        </li>
                    ))}
                </ol>
            )}
        </div>
    );
}
