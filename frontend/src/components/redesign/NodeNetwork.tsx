import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import * as api from '../../services/apiServices';
import { Plus, Trash2, Shield, Network, X, UserRound, Mail } from 'lucide-react';
import type { User, RegisterData } from '../../services/apiServices';

interface CreateClientModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: RegisterData) => void;
  isLoading?: boolean;
}

function CreateClientModal({ isOpen, onClose, onSubmit, isLoading }: CreateClientModalProps) {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  useEffect(() => {
    if (!isOpen) {
      setUsername('');
      setEmail('');
      setPassword('');
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ username, email, password });
  };

  const inputClass =
    'w-full rounded-xl px-4 py-3 text-[15px] focus:outline-none focus:ring-2 focus:ring-[color:var(--accent-primary)]/30 transition-all';

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 backdrop-blur-md"
      style={{ backgroundColor: 'rgba(0,0,0,0.35)' }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="w-full max-w-md rounded-3xl shadow-2xl overflow-hidden"
        style={{ backgroundColor: 'var(--background-card)', border: '1px solid var(--border-color)' }}
      >
        <div className="flex items-center justify-between p-6 pb-4">
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-(--accent-primary)" />
            <h2 className="text-xl font-semibold text-(--text-primary)">Create Client Node</h2>
          </div>
          <button onClick={onClose} className="w-8 h-8 rounded-full inline-flex items-center justify-center text-(--text-secondary) hover:text-(--text-primary)">
            <X className="w-[18px] h-[18px]" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="px-6 pb-6 flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold uppercase tracking-widest text-(--text-secondary)">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="node-edge-01"
              className={inputClass}
              style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}
              required
              autoFocus
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold uppercase tracking-widest text-(--text-secondary)">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="edge01@fedlearn.internal"
              className={inputClass}
              style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}
              required
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold uppercase tracking-widest text-(--text-secondary)">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className={inputClass}
              style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}
              required
            />
          </div>

          <div className="flex gap-3 mt-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isLoading}
              className="flex-1 rounded-full py-3 text-sm font-medium"
              style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading || !username || !email || !password}
              className="flex-1 rounded-full py-3 text-sm font-semibold text-white disabled:opacity-50"
              style={{ backgroundColor: 'var(--accent-primary)' }}
            >
              {isLoading ? 'Creating...' : 'Create'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export function NodeNetwork() {
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);

  const loadUsers = useCallback(async () => {
    try {
      setIsLoading(true);
      const res = await api.fetchUsers();
      setUsers(Array.isArray(res.data) ? res.data : []);
      setError('');
    } catch {
      setError('Failed to fetch clients network.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  const handleCreate = async (data: RegisterData) => {
    try {
      setIsCreating(true);
      await api.createUser(data);
      setIsModalOpen(false);
      loadUsers();
    } catch {
      setError('Failed to create client.');
    } finally {
      setIsCreating(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm('Delete this client node?')) return;
    try {
      await api.deleteUser(id);
      setUsers((prev) => prev.filter((u) => u.id !== id));
    } catch {
      setError('Failed to delete client.');
    }
  };

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden">
      <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px)' }}>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="flex items-center justify-between gap-4">
          <div>
            <h1 className="font-display text-4xl font-semibold tracking-tight text-(--text-primary)">Node Network</h1>
            <p className="text-sm text-(--text-secondary) mt-1">Manage operator accounts for edge clients.</p>
          </div>
          <button
            onClick={() => setIsModalOpen(true)}
            className="inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-semibold text-white"
            style={{ backgroundColor: 'var(--accent-primary)' }}
          >
            <Plus className="w-4 h-4" />
            Add Client
          </button>
        </motion.div>
      </div>

      <div className="flex-1 overflow-y-auto px-8 py-8">
        {error && (
          <div className="mb-6 px-5 py-3 rounded-2xl text-sm font-medium" style={{ backgroundColor: 'color-mix(in srgb, #ef4444 12%, transparent)', color: '#ef4444', border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)' }}>
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center h-64 text-(--text-secondary)">Loading network data...</div>
        ) : users.length > 0 ? (
          <motion.div
            initial="hidden"
            animate="visible"
            variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.06 } } }}
            className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6"
          >
            {users.map((user) => (
              <motion.div key={user.id} variants={{ hidden: { opacity: 0, y: 12 }, visible: { opacity: 1, y: 0 } }}>
                <div className="rounded-3xl p-5 flex flex-col gap-4" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="w-10 h-10 rounded-xl inline-flex items-center justify-center" style={{ backgroundColor: 'color-mix(in srgb, var(--accent-primary) 12%, transparent)' }}>
                        <Network className="w-5 h-5 text-(--accent-primary)" />
                      </div>
                      <div className="min-w-0">
                        <h3 className="text-lg font-semibold text-(--text-primary) truncate">{user.username}</h3>
                        <p className="text-xs text-(--text-secondary)">Client account</p>
                      </div>
                    </div>
                    <button onClick={() => handleDelete(user.id)} className="h-8 w-8 rounded-full inline-flex items-center justify-center text-(--text-secondary) hover:text-rose-500">
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>

                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2 text-(--text-secondary)">
                      <UserRound className="w-4 h-4" />
                      <span>ID #{user.id}</span>
                    </div>
                    <div className="flex items-center gap-2 text-(--text-secondary) break-all">
                      <Mail className="w-4 h-4" />
                      <span>{user.email}</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        ) : (
          <div className="flex flex-col items-center justify-center h-64 text-(--text-secondary) gap-2">
            <p className="text-lg text-(--text-primary)">No clients found.</p>
            <p className="text-sm">Create accounts to onboard edge nodes.</p>
          </div>
        )}
      </div>

      <CreateClientModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} onSubmit={handleCreate} isLoading={isCreating} />
    </div>
  );
}
