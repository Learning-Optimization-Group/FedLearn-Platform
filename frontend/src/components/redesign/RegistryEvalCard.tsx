// =============================================================================
// FedLearn Frontend — Registry eval card (FE-11)
// =============================================================================
// Renders an artifact's raw eval card. `evalCardJson` is an untrusted RAW JSON
// string straight off the wire — it may be null, malformed, or not an object.
// Parsing is fully defensive: anything that isn't a plain JSON object renders
// the honest "No eval card" empty state instead of throwing.

import { SectionLabel } from '../ui';

interface RegistryEvalCardProps {
    evalCardJson: string | null;
}

/** Parse the raw eval-card string into a flat object, or null if absent/invalid. */
function parseEvalCard(raw: string | null): Record<string, unknown> | null {
    if (!raw) return null;
    try {
        const parsed: unknown = JSON.parse(raw);
        if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
            return parsed as Record<string, unknown>;
        }
        return null;
    } catch {
        return null;
    }
}

/** Render an arbitrary JSON value as compact display text. */
function displayValue(value: unknown): string {
    if (value === null) return 'null';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
}

export function RegistryEvalCard({ evalCardJson }: RegistryEvalCardProps) {
    const parsed = parseEvalCard(evalCardJson);
    const entries = parsed ? Object.entries(parsed) : [];

    return (
        <div className="flex flex-col gap-2">
            <SectionLabel>Evaluation</SectionLabel>
            {entries.length === 0 ? (
                <div className="rounded-card border border-hairline bg-surface-1 px-4 py-3 text-label text-fg-muted">
                    No eval card
                </div>
            ) : (
                <dl className="rounded-card border border-hairline bg-surface-1 divide-y divide-hairline">
                    {entries.map(([key, value]) => (
                        <div key={key} className="flex items-start justify-between gap-4 px-4 py-2.5">
                            <dt className="text-caption text-fg-muted">{key}</dt>
                            <dd className="font-mono tabular-nums text-caption text-fg text-right break-all">
                                {displayValue(value)}
                            </dd>
                        </div>
                    ))}
                </dl>
            )}
        </div>
    );
}
