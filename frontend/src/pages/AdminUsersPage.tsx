import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Users } from 'lucide-react';
import { cn } from '../lib/utils';
import { useAuth } from '../context/AuthContext';
import * as api from '../services/apiServices';
import type { AdminUser } from '../services/apiServices';

function RoleBadge({ role }: { role: 'USER' | 'ADMIN' }) {
    if (role === 'ADMIN') {
        return (
            <span 
                className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-semibold uppercase tracking-wider"
                style={{
                    backgroundColor: 'color-mix(in srgb, var(--accent-primary) 10%, transparent)',
                    color: 'var(--accent-primary)',
                    border: '1px solid color-mix(in srgb, var(--accent-primary) 20%, transparent)'
                }}
            >
                {role}
            </span>
        );
    }
    return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-semibold uppercase tracking-wider bg-(--border-color) text-(--text-secondary) border border-(--border-color)">
            {role}
        </span>
    );
}

export default function AdminUsersPage() {
    const { currentUser } = useAuth();
    const navigate = useNavigate();
    const [users, setUsers] = useState<AdminUser[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [actionError, setActionError] = useState('');

    useEffect(() => {
        if (currentUser?.role !== 'ADMIN') {
            navigate('/dashboard', { replace: true });
        }
    }, [currentUser, navigate]);

    const load = useCallback(async () => {
        try {
            setIsLoading(true);
            const res = await api.fetchAdminUsers();
            setUsers(Array.isArray(res.data) ? res.data : []);
            setError('');
        } catch {
            setError('Failed to load users.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    const handleRoleChange = async (user: AdminUser, newRole: 'USER' | 'ADMIN') => {
        setActionError('');
        try {
            const res = await api.updateUserRole(user.id, newRole);
            setUsers((prev) => prev.map((u) => u.id === user.id ? res.data : u));
        } catch (err: any) {
            if (err?.response?.status === 409) {
                setActionError('Cannot demote the only remaining admin.');
            } else {
                setActionError('Failed to change role.');
            }
        }
    };

    const adminCount = users.filter((u) => u.role === 'ADMIN').length;

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden font-sans">
            <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px) saturate(160%)' }}>
                <div className="flex items-center gap-3">
                    <Users className="w-6 h-6 text-(--accent-primary)" />
                    <div>
                        <h1 className="font-display text-3xl font-semibold tracking-tight text-(--text-primary)">Manage Users</h1>
                        <p className="text-sm text-(--text-secondary) mt-1">{users.length} total users · {adminCount} admin{adminCount !== 1 ? 's' : ''}</p>
                    </div>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto px-8 py-8">
                {(error || actionError) && (
                    <div className="mb-6 px-5 py-3 rounded-2xl text-sm font-medium" style={{ backgroundColor: 'color-mix(in srgb, #ef4444 12%, transparent)', color: '#ef4444', border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)' }}>
                        {error || actionError}
                    </div>
                )}

                {isLoading ? (
                    <div className="flex items-center justify-center h-64 text-(--text-secondary)">Loading…</div>
                ) : (
                    <div className="max-w-4xl flex flex-col gap-2">
                        {users.map((user) => (
                            <motion.div
                                key={user.id}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="flex items-center gap-4 px-5 py-4 rounded-2xl"
                                style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}
                            >
                                <div className="w-10 h-10 rounded-full bg-(--background-secondary) flex items-center justify-center text-sm font-semibold text-(--text-primary) shrink-0">
                                    {user.username.slice(0, 2).toUpperCase()}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2">
                                        <span className="text-[15px] font-semibold text-(--text-primary) truncate">{user.username}</span>
                                        <RoleBadge role={user.role} />
                                    </div>
                                    <p className="text-[12px] text-(--text-secondary) mt-0.5">
                                        {user.email} · {user.projectsOwned} project{user.projectsOwned !== 1 ? 's' : ''} owned · {user.memberships} membership{user.memberships !== 1 ? 's' : ''}
                                    </p>
                                </div>
                                <div className="flex items-center gap-2 shrink-0">
                                    {user.role === 'USER' ? (
                                        <button
                                            onClick={() => handleRoleChange(user, 'ADMIN')}
                                            className="px-3 py-1.5 rounded-xl text-[13px] font-semibold text-white"
                                            style={{ backgroundColor: 'var(--accent-primary)' }}
                                        >
                                            Promote to Admin
                                        </button>
                                    ) : (
                                        <button
                                            onClick={() => handleRoleChange(user, 'USER')}
                                            disabled={adminCount <= 1}
                                            title={adminCount <= 1 ? 'Cannot demote the only admin' : 'Demote to User'}
                                            className={cn(
                                                'px-3 py-1.5 rounded-xl text-[13px] font-semibold border transition-all',
                                                adminCount <= 1
                                                    ? 'border-(--border-color) text-(--text-secondary) opacity-40 cursor-not-allowed'
                                                    : 'border-rose-500/30 text-rose-500 hover:bg-rose-500/10'
                                            )}
                                        >
                                            Demote to User
                                        </button>
                                    )}
                                </div>
                            </motion.div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
