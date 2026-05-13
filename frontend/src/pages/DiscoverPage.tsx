import { useCallback, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Compass, Globe, Lock } from 'lucide-react';
import { cn } from '../lib/utils';
import * as api from '../services/apiServices';
import type { DiscoverProject } from '../services/apiServices';

function VisibilityBadge({ visibility }: { visibility: 'PUBLIC' | 'PRIVATE' }) {
    return (
        <span className={cn(
            'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold uppercase tracking-wider',
            visibility === 'PUBLIC'
                ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20'
                : 'bg-amber-500/10 text-amber-500 border border-amber-500/20'
        )}>
            {visibility === 'PUBLIC' ? <Globe className="w-3 h-3" /> : <Lock className="w-3 h-3" />}
            {visibility}
        </span>
    );
}

function RequestDialog({ projectId, onSuccess, onCancel }: { projectId: string; onSuccess: () => void; onCancel: () => void }) {
    const [message, setMessage] = useState('');
    const [loading, setLoading] = useState(false);

    const submit = async () => {
        setLoading(true);
        try {
            await api.createAccessRequest(projectId, message || undefined);
            onSuccess();
        } catch {
            // swallow; user stays on page
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="mt-3 flex flex-col gap-2">
            <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Optional message to the owner..."
                rows={2}
                maxLength={1000}
                className="w-full rounded-xl px-3 py-2 text-[13px] resize-none"
                style={{ backgroundColor: 'var(--background-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
            />
            <div className="flex gap-2">
                <button
                    onClick={submit}
                    disabled={loading}
                    className="flex-1 py-2 rounded-xl text-[13px] font-semibold text-white"
                    style={{ backgroundColor: 'var(--accent-primary)' }}
                >
                    {loading ? 'Sending…' : 'Send Request'}
                </button>
                <button
                    onClick={onCancel}
                    className="flex-1 py-2 rounded-xl text-[13px] font-medium text-(--text-secondary) border"
                    style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--background-secondary)' }}
                >
                    Cancel
                </button>
            </div>
        </div>
    );
}

function DiscoverCard({ project, onJoined }: { project: DiscoverProject; onJoined: (id: string) => void }) {
    const [showDialog, setShowDialog] = useState(false);
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState(project.myRequestStatus);

    const handleJoin = async () => {
        setLoading(true);
        try {
            await api.createAccessRequest(project.id);
            onJoined(project.id);
        } catch {
            // swallow
        } finally {
            setLoading(false);
        }
    };

    const handleRequestSuccess = () => {
        setStatus('PENDING');
        setShowDialog(false);
    };

    const ctaButton = () => {
        if (status === 'APPROVED') {
            return <span className="text-[13px] font-semibold text-emerald-500">Joined</span>;
        }
        if (status === 'PENDING') {
            return <span className="text-[13px] font-semibold text-amber-500">Request Pending</span>;
        }
        if (status === 'DENIED') {
            return (
                <button
                    onClick={() => setShowDialog(true)}
                    className="text-[13px] font-semibold text-(--text-secondary) border rounded-xl px-3 py-1.5"
                    style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--background-secondary)' }}
                >
                    Re-request
                </button>
            );
        }
        if (project.visibility === 'PUBLIC') {
            return (
                <button
                    onClick={handleJoin}
                    disabled={loading}
                    className="text-[13px] font-semibold text-white rounded-xl px-4 py-1.5"
                    style={{ backgroundColor: 'var(--accent-primary)' }}
                >
                    {loading ? 'Joining…' : 'Join'}
                </button>
            );
        }
        return (
            <button
                onClick={() => setShowDialog((v) => !v)}
                className="text-[13px] font-semibold rounded-xl px-4 py-1.5 border"
                style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--background-secondary)', color: 'var(--text-primary)' }}
            >
                Request Access
            </button>
        );
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-3xl p-5 flex flex-col gap-3"
            style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}
        >
            <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                    <h3 className="text-[17px] font-semibold tracking-tight text-(--text-primary) truncate">{project.name}</h3>
                    <p className="text-[12px] text-(--text-secondary) mt-0.5">by {project.ownerUsername} · {project.modelType}</p>
                </div>
                <VisibilityBadge visibility={project.visibility} />
            </div>

            {project.description && (
                <p className="text-[13px] text-(--text-secondary) line-clamp-2">{project.description}</p>
            )}

            {project.lastAccuracy != null && (
                <p className="text-[12px] text-(--text-secondary)">
                    Latest accuracy: <span className="font-semibold text-(--text-primary)">{(project.lastAccuracy * 100).toFixed(1)}%</span>
                </p>
            )}

            <div className="flex items-center justify-end mt-1">{ctaButton()}</div>

            {showDialog && (
                <RequestDialog
                    projectId={project.id}
                    onSuccess={handleRequestSuccess}
                    onCancel={() => setShowDialog(false)}
                />
            )}
        </motion.div>
    );
}

export default function DiscoverPage() {
    const [projects, setProjects] = useState<DiscoverProject[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    const load = useCallback(async () => {
        try {
            setIsLoading(true);
            const res = await api.fetchDiscover();
            setProjects(Array.isArray(res.data) ? res.data : []);
            setError('');
        } catch {
            setError('Failed to load discoverable projects.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    const handleJoined = (id: string) => {
        setProjects((prev) =>
            prev.map((p) => p.id === id ? { ...p, myRequestStatus: 'APPROVED' as const } : p)
        );
    };

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden font-sans">
            <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px) saturate(160%)' }}>
                <div className="flex items-center gap-3">
                    <Compass className="w-6 h-6 text-(--accent-primary)" />
                    <div>
                        <h1 className="font-display text-3xl font-semibold tracking-tight text-(--text-primary)">Discover Projects</h1>
                        <p className="text-sm text-(--text-secondary) mt-1">Browse publicly visible projects and request access to private ones.</p>
                    </div>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto px-8 py-8">
                {error && (
                    <div className="mb-6 px-5 py-3 rounded-2xl text-sm font-medium" style={{ backgroundColor: 'color-mix(in srgb, #ef4444 12%, transparent)', color: '#ef4444', border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)' }}>
                        {error}
                    </div>
                )}
                {isLoading ? (
                    <div className="flex items-center justify-center h-64 text-(--text-secondary)">Loading…</div>
                ) : projects.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-64 text-(--text-secondary) gap-2">
                        <p className="text-[16px] font-medium text-(--text-primary)">No discoverable projects found.</p>
                        <p className="text-[14px]">Projects become visible here when their owners make them public.</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 max-w-[1700px] mx-auto">
                        {projects.map((p) => (
                            <DiscoverCard key={p.id} project={p} onJoined={handleJoined} />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
