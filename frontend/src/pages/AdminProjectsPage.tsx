import { useCallback, useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ShieldCheck, Globe, Lock, ExternalLink } from 'lucide-react';
import { cn } from '../lib/utils';
import { useAuth } from '../context/AuthContext';
import * as api from '../services/apiServices';
import type { Project } from '../services/apiServices';

function statusDot(status: Project['status']) {
    if (status === 'RUNNING') return 'bg-blue-500 animate-pulse';
    if (status === 'COMPLETED') return 'bg-emerald-500';
    if (status === 'FAILED') return 'bg-rose-500';
    return 'bg-(--text-secondary)';
}

export default function AdminProjectsPage() {
    const { currentUser } = useAuth();
    const navigate = useNavigate();
    const [projects, setProjects] = useState<Project[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        if (currentUser?.role !== 'ADMIN') {
            navigate('/dashboard', { replace: true });
        }
    }, [currentUser, navigate]);

    const load = useCallback(async () => {
        try {
            setIsLoading(true);
            const res = await api.fetchAdminProjects();
            setProjects(Array.isArray(res.data) ? res.data : []);
            setError('');
        } catch {
            setError('Failed to load projects.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden font-sans">
            <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px) saturate(160%)' }}>
                <div className="flex items-center gap-3">
                    <ShieldCheck className="w-6 h-6 text-(--accent-primary)" />
                    <div>
                        <h1 className="font-display text-3xl font-semibold tracking-tight text-(--text-primary)">All Projects</h1>
                        <p className="text-sm text-(--text-secondary) mt-1">{projects.length} total projects across all users</p>
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
                ) : (
                    <div className="max-w-5xl flex flex-col gap-2">
                        {projects.map((p) => (
                            <motion.div
                                key={p.id}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="flex items-center gap-4 px-5 py-4 rounded-2xl"
                                style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}
                            >
                                <div className={cn('w-2 h-2 rounded-full shrink-0', statusDot(p.status))} />
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2">
                                        <span className="text-[15px] font-semibold text-(--text-primary) truncate">{p.name}</span>
                                        {p.visibility === 'PUBLIC'
                                            ? <Globe className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                                            : <Lock className="w-3.5 h-3.5 text-amber-500 shrink-0" />
                                        }
                                    </div>
                                    <p className="text-[12px] text-(--text-secondary) mt-0.5">
                                        {p.modelType} · {p.modelName} · {p.status}
                                    </p>
                                </div>
                                <Link
                                    to={`/projects/${p.id}`}
                                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[13px] font-medium border transition-all hover:bg-(--background-secondary) text-(--text-secondary) hover:text-(--text-primary)"
                                    style={{ borderColor: 'var(--border-color)' }}
                                >
                                    <ExternalLink className="w-3.5 h-3.5" />
                                    View
                                </Link>
                            </motion.div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
