import { useCallback, useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, Globe, Lock, Plus, Trash2, Check, X } from 'lucide-react';
import { cn } from '../lib/utils';
import { useAuth } from '../context/AuthContext';
import * as api from '../services/apiServices';
import type { Project, Membership, AccessRequest, UserSearchResult } from '../services/apiServices';

type Tab = 'members' | 'clients' | 'requests' | 'model';

function VisibilityToggle({
    visibility,
    onToggle,
    canEdit,
}: {
    visibility: 'PUBLIC' | 'PRIVATE';
    onToggle: (v: 'PUBLIC' | 'PRIVATE') => void;
    canEdit: boolean;
}) {
    return (
        <div className="flex items-center gap-3">
            <span className="text-[13px] text-(--text-secondary) font-medium">Visibility:</span>
            {canEdit ? (
                <button
                    onClick={() => onToggle(visibility === 'PUBLIC' ? 'PRIVATE' : 'PUBLIC')}
                    className={cn(
                        'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[13px] font-semibold border transition-all',
                        visibility === 'PUBLIC'
                            ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20 hover:bg-emerald-500/20'
                            : 'bg-amber-500/10 text-amber-500 border-amber-500/20 hover:bg-amber-500/20'
                    )}
                >
                    {visibility === 'PUBLIC' ? <Globe className="w-3.5 h-3.5" /> : <Lock className="w-3.5 h-3.5" />}
                    {visibility}
                </button>
            ) : (
                <span className={cn(
                    'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[13px] font-semibold border',
                    visibility === 'PUBLIC'
                        ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20'
                        : 'bg-amber-500/10 text-amber-500 border-amber-500/20'
                )}>
                    {visibility === 'PUBLIC' ? <Globe className="w-3.5 h-3.5" /> : <Lock className="w-3.5 h-3.5" />}
                    {visibility}
                </span>
            )}
        </div>
    );
}

function UserSearchInput({
    label,
    onAdd,
}: {
    label: string;
    onAdd: (username: string) => Promise<void>;
}) {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<UserSearchResult[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (query.length < 2) { setResults([]); return; }
        const t = setTimeout(async () => {
            try {
                const res = await api.searchUsers(query);
                setResults(res.data);
            } catch {
                setResults([]);
            }
        }, 300);
        return () => clearTimeout(t);
    }, [query]);

    const handleAdd = async (username: string) => {
        setLoading(true);
        try {
            await onAdd(username);
            setQuery('');
            setResults([]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="relative">
            <div className="flex items-center gap-2">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder={`Add ${label} by username…`}
                    className="flex-1 rounded-xl px-3 py-2 text-[13px]"
                    style={{ backgroundColor: 'var(--background-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
                />
                <Plus className="w-4 h-4 text-(--text-secondary)" />
            </div>
            {results.length > 0 && (
                <div
                    className="absolute top-full mt-1 left-0 right-0 z-20 rounded-xl overflow-hidden shadow-lg"
                    style={{ backgroundColor: 'var(--background-card)', border: '1px solid var(--border-color)' }}
                >
                    {results.map((u) => (
                        <button
                            key={u.id}
                            onClick={() => handleAdd(u.username)}
                            disabled={loading}
                            className="w-full px-4 py-2.5 text-left text-[13px] text-(--text-primary) hover:bg-(--background-secondary) transition-colors"
                        >
                            {u.username}
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}

function MembershipRow({
    m,
    canRemove,
    onRemove,
}: {
    m: Membership;
    canRemove: boolean;
    onRemove: (userId: number) => void;
}) {
    const [confirm, setConfirm] = useState(false);
    return (
        <div
            className="flex items-center justify-between px-4 py-3 rounded-xl"
            style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}
        >
            <div>
                <span className="text-[14px] font-medium text-(--text-primary)">{m.username}</span>
                {m.partitionId != null && (
                    <span className="ml-2 text-[12px] text-(--text-secondary)">partition {m.partitionId}</span>
                )}
            </div>
            {canRemove && (
                <button
                    onClick={confirm ? () => onRemove(m.userId) : () => { setConfirm(true); setTimeout(() => setConfirm(false), 3000); }}
                    className={cn(
                        'p-1.5 rounded-lg transition-colors',
                        confirm ? 'text-rose-500 bg-rose-500/10' : 'text-(--text-secondary) hover:text-rose-500'
                    )}
                >
                    <Trash2 className="w-4 h-4" />
                </button>
            )}
        </div>
    );
}

function RequestRow({
    req,
    canDecide,
    onDecide,
}: {
    req: AccessRequest;
    canDecide: boolean;
    onDecide: (id: number, decision: 'APPROVED' | 'DENIED') => void;
}) {
    return (
        <div
            className="flex items-center justify-between px-4 py-3 rounded-xl gap-3"
            style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}
        >
            <div className="flex-1 min-w-0">
                <span className="text-[14px] font-medium text-(--text-primary)">{req.username}</span>
                <span className="text-[12px] text-(--text-secondary) ml-2">{new Date(req.requestedAt).toLocaleDateString()}</span>
                {req.message && <p className="text-[12px] text-(--text-secondary) mt-0.5 italic line-clamp-1">"{req.message}"</p>}
            </div>
            <span className={cn(
                'px-2 py-0.5 rounded-full text-[11px] font-semibold uppercase tracking-wider',
                req.status === 'PENDING' && 'bg-amber-500/10 text-amber-500',
                req.status === 'APPROVED' && 'bg-emerald-500/10 text-emerald-500',
                req.status === 'DENIED' && 'bg-rose-500/10 text-rose-500'
            )}>{req.status}</span>
            {canDecide && req.status === 'PENDING' && (
                <div className="flex items-center gap-1">
                    <button
                        onClick={() => onDecide(req.id, 'APPROVED')}
                        className="p-1.5 rounded-lg text-emerald-500 hover:bg-emerald-500/10 transition-colors"
                        title="Approve"
                    >
                        <Check className="w-4 h-4" />
                    </button>
                    <button
                        onClick={() => onDecide(req.id, 'DENIED')}
                        className="p-1.5 rounded-lg text-rose-500 hover:bg-rose-500/10 transition-colors"
                        title="Deny"
                    >
                        <X className="w-4 h-4" />
                    </button>
                </div>
            )}
        </div>
    );
}

export default function ProjectDetailPage() {
    const { projectId } = useParams<{ projectId: string }>();
    const navigate = useNavigate();
    const { currentUser } = useAuth();

    const [project, setProject] = useState<Project | null>(null);
    const [memberships, setMemberships] = useState<Membership[]>([]);
    const [requests, setRequests] = useState<AccessRequest[]>([]);
    const [tab, setTab] = useState<Tab>('members');
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    const isOwner = project?.myRelationship === 'OWNER';
    const isMember = project?.myRelationship === 'MEMBER';
    const isAdmin = currentUser?.role === 'ADMIN';
    const canManageMembers = isOwner || isAdmin;
    const canManageClients = isOwner || isMember || isAdmin;
    const canSeeManagement = isOwner || isMember || isAdmin;

    const loadProject = useCallback(async () => {
        if (!projectId) return;
        try {
            setIsLoading(true);
            const res = await api.fetchProject(projectId);
            setProject(res.data);
            setError('');
        } catch {
            setError('Project not found or access denied.');
        } finally {
            setIsLoading(false);
        }
    }, [projectId]);

    const loadMemberships = useCallback(async () => {
        if (!projectId || !canSeeManagement) return;
        try {
            const res = await api.fetchMemberships(projectId);
            setMemberships(Array.isArray(res.data) ? res.data : []);
        } catch {
            // silently ignore if not authorized
        }
    }, [projectId, canSeeManagement]);

    const loadRequests = useCallback(async () => {
        if (!projectId || !canSeeManagement) return;
        try {
            const res = await api.fetchProjectAccessRequests(projectId);
            setRequests(Array.isArray(res.data) ? res.data : []);
        } catch {
            // silently ignore
        }
    }, [projectId, canSeeManagement]);

    useEffect(() => { loadProject(); }, [loadProject]);
    useEffect(() => {
        if (project && canSeeManagement) {
            loadMemberships();
            loadRequests();
        }
    }, [project, canSeeManagement, loadMemberships, loadRequests]);

    const handleToggleVisibility = async (next: 'PUBLIC' | 'PRIVATE') => {
        if (!projectId) return;
        try {
            const res = await api.patchProject(projectId, { visibility: next });
            setProject(res.data);
        } catch {
            // swallow
        }
    };

    const handleAddMember = async (username: string) => {
        if (!projectId) return;
        await api.addMembership(projectId, { username, role: 'MEMBER' });
        loadMemberships();
    };

    const handleAddClient = async (username: string) => {
        if (!projectId) return;
        await api.addMembership(projectId, { username, role: 'CLIENT' });
        loadMemberships();
    };

    const handleRemoveMembership = async (userId: number) => {
        if (!projectId) return;
        await api.removeMembership(projectId, userId);
        loadMemberships();
    };

    const handleDecide = async (reqId: number, decision: 'APPROVED' | 'DENIED') => {
        if (!projectId) return;
        await api.decideAccessRequest(projectId, reqId, decision);
        loadRequests();
        if (decision === 'APPROVED') loadMemberships();
    };

    const members = memberships.filter((m) => m.role === 'MEMBER');
    const clients = memberships.filter((m) => m.role === 'CLIENT');

    const pendingCount = requests.filter((r) => r.status === 'PENDING').length;

    const tabs: { key: Tab; label: string }[] = [
        { key: 'members', label: 'Members' },
        { key: 'clients', label: 'Clients' },
        { key: 'requests', label: pendingCount > 0 ? `Requests (${pendingCount})` : 'Requests' },
        { key: 'model', label: 'Model Card' },
    ];

    if (isLoading) {
        return <div className="flex-1 flex items-center justify-center text-(--text-secondary)">Loading…</div>;
    }
    if (error || !project) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center gap-4 text-(--text-secondary)">
                <p>{error || 'Project not found.'}</p>
                <button onClick={() => navigate('/dashboard')} className="text-sm text-(--accent-primary)">← Back to Dashboard</button>
            </div>
        );
    }

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden font-sans">
            <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px) saturate(160%)' }}>
                <div className="flex flex-col gap-4">
                    <button
                        onClick={() => navigate(-1)}
                        className="inline-flex items-center gap-1.5 text-sm text-(--text-secondary) hover:text-(--text-primary) transition-colors w-fit"
                    >
                        <ArrowLeft className="w-4 h-4" />
                        Back
                    </button>
                    <div className="flex flex-wrap items-center justify-between gap-4">
                        <div>
                            <h1 className="font-display text-3xl font-semibold tracking-tight text-(--text-primary)">{project.name}</h1>
                            <p className="text-sm text-(--text-secondary) mt-1">{project.modelType} · {project.modelName} · {project.status}</p>
                        </div>
                        {project.visibility && (
                            <VisibilityToggle
                                visibility={project.visibility}
                                onToggle={handleToggleVisibility}
                                canEdit={canManageMembers}
                            />
                        )}
                    </div>
                </div>
            </div>

            {canSeeManagement && (
                <>
                    <div className="flex gap-1 px-8 pt-4 border-b" style={{ borderColor: 'var(--border-color)' }}>
                        {tabs.map((t) => (
                            <button
                                key={t.key}
                                onClick={() => setTab(t.key)}
                                className={cn(
                                    'px-4 py-2.5 text-[14px] font-medium rounded-t-xl transition-colors border-b-2',
                                    tab === t.key
                                        ? 'text-(--accent-primary) border-(--accent-primary) bg-(--background-card)'
                                        : 'text-(--text-secondary) border-transparent hover:text-(--text-primary)'
                                )}
                            >
                                {t.label}
                            </button>
                        ))}
                    </div>

                    <div className="flex-1 overflow-y-auto px-8 py-6">
                        {tab === 'members' && (
                            <motion.div key="members" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-2xl flex flex-col gap-4">
                                {canManageMembers && (
                                    <UserSearchInput label="member" onAdd={handleAddMember} />
                                )}
                                {members.length === 0 ? (
                                    <p className="text-[13px] text-(--text-secondary)">No members yet.</p>
                                ) : (
                                    members.map((m) => (
                                        <MembershipRow key={m.userId} m={m} canRemove={canManageMembers} onRemove={handleRemoveMembership} />
                                    ))
                                )}
                            </motion.div>
                        )}

                        {tab === 'clients' && (
                            <motion.div key="clients" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-2xl flex flex-col gap-4">
                                {canManageClients && (
                                    <UserSearchInput label="client" onAdd={handleAddClient} />
                                )}
                                {clients.length === 0 ? (
                                    <p className="text-[13px] text-(--text-secondary)">No clients yet.</p>
                                ) : (
                                    clients.map((m) => (
                                        <MembershipRow key={m.userId} m={m} canRemove={canManageClients} onRemove={handleRemoveMembership} />
                                    ))
                                )}
                            </motion.div>
                        )}

                        {tab === 'requests' && (
                            <motion.div key="requests" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-2xl flex flex-col gap-4">
                                {requests.length === 0 ? (
                                    <p className="text-[13px] text-(--text-secondary)">No access requests.</p>
                                ) : (
                                    requests.map((r) => (
                                        <RequestRow key={r.id} req={r} canDecide={canManageClients} onDecide={handleDecide} />
                                    ))
                                )}
                            </motion.div>
                        )}

                        {tab === 'model' && (
                            <motion.div key="model" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="max-w-2xl">
                                <div
                                    className="rounded-2xl p-6 text-center text-(--text-secondary)"
                                    style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}
                                >
                                    <p className="text-[15px] font-medium text-(--text-primary)">Model Hub</p>
                                    <p className="text-[13px] mt-1">Model publishing and inference UI coming in Plan 4.</p>
                                </div>
                            </motion.div>
                        )}
                    </div>
                </>
            )}

            {!canSeeManagement && (
                <div className="flex-1 flex flex-col items-center justify-center text-(--text-secondary) gap-2">
                    <p className="text-[15px] font-medium text-(--text-primary)">You are a client of this project.</p>
                    <p className="text-[13px]">Connect via the Electron app to participate in training.</p>
                </div>
            )}
        </div>
    );
}
