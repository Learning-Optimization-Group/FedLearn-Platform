import { useCallback, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Inbox } from 'lucide-react';
import { cn } from '../lib/utils';
import * as api from '../services/apiServices';
import type { AccessRequest } from '../services/apiServices';

function StatusChip({ status }: { status: AccessRequest['status'] }) {
    return (
        <span className={cn(
            'inline-flex items-center px-2.5 py-0.5 rounded-full text-[12px] font-semibold uppercase tracking-wider',
            status === 'PENDING' && 'bg-amber-500/10 text-amber-500 border border-amber-500/20',
            status === 'APPROVED' && 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20',
            status === 'DENIED' && 'bg-rose-500/10 text-rose-500 border border-rose-500/20'
        )}>
            {status}
        </span>
    );
}

export default function MyRequestsPage() {
    const [requests, setRequests] = useState<AccessRequest[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    const load = useCallback(async () => {
        try {
            setIsLoading(true);
            const res = await api.fetchMyAccessRequests();
            setRequests(Array.isArray(res.data) ? res.data : []);
            setError('');
        } catch {
            setError('Failed to load requests.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden font-sans">
            <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px) saturate(160%)' }}>
                <div className="flex items-center gap-3">
                    <Inbox className="w-6 h-6 text-(--accent-primary)" />
                    <div>
                        <h1 className="font-display text-3xl font-semibold tracking-tight text-(--text-primary)">My Requests</h1>
                        <p className="text-sm text-(--text-secondary) mt-1">Access requests you have submitted to private projects.</p>
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
                ) : requests.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-64 text-(--text-secondary) gap-2">
                        <p className="text-[16px] font-medium text-(--text-primary)">No requests yet.</p>
                        <p className="text-[14px]">Use Discover to request access to private projects.</p>
                    </div>
                ) : (
                    <div className="flex flex-col gap-3 max-w-3xl">
                        {requests.map((req) => (
                            <motion.div
                                key={req.id}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="rounded-2xl px-5 py-4 flex items-center justify-between gap-4"
                                style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}
                            >
                                <div className="flex-1 min-w-0">
                                    <p className="text-[15px] font-semibold text-(--text-primary) truncate">{req.projectName}</p>
                                    <p className="text-[12px] text-(--text-secondary) mt-0.5">
                                        Requested {new Date(req.requestedAt).toLocaleDateString()}
                                        {req.decidedAt && ` · Decided ${new Date(req.decidedAt).toLocaleDateString()}`}
                                        {req.decidedByUsername && ` by ${req.decidedByUsername}`}
                                    </p>
                                    {req.message && (
                                        <p className="text-[12px] text-(--text-secondary) mt-1 italic line-clamp-1">"{req.message}"</p>
                                    )}
                                </div>
                                <StatusChip status={req.status} />
                            </motion.div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
