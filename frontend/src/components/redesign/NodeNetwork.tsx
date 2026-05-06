// =============================================================================
// FedLearn Frontend — Node Network (Client Management)
// =============================================================================
// Allows tracking, creating, and deleting FL clients (users).

import { useState, useEffect, useCallback } from 'react';
import * as api from '../../services/apiServices';
import { Plus, Trash2, Shield, Network, X } from 'lucide-react';
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
    "w-full bg-[#1c1c1e] border border-[rgba(255,255,255,0.1)] rounded-xl px-4 py-3 text-[15px] text-[#f5f5f7] placeholder-[#86868b] focus:outline-none focus:ring-[3px] focus:ring-[#0a84ff]/30 focus:border-[#0a84ff]/50 transition-all";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xl font-sans"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-[rgba(28,28,30,0.92)] border border-[rgba(255,255,255,0.1)] w-full max-w-md rounded-[32px] shadow-[0_20px_50px_rgba(0,0,0,0.5)] flex flex-col overflow-hidden text-[#f5f5f7]">
        <div className="flex items-center justify-between p-6 pb-4">
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-[#32d74b]" />
            <h2 className="text-[22px] font-semibold tracking-tight">Create Client Node</h2>
          </div>
          <button onClick={onClose} className="w-8 h-8 flex items-center justify-center text-[#86868b] bg-[#3a3a3c] hover:bg-[rgba(255,255,255,0.2)] rounded-full transition-colors">
            <X className="w-[18px] h-[18px]" />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="px-6 pb-6 flex flex-col gap-5">
          <div className="flex flex-col gap-2">
            <label className="text-[13px] font-medium text-[#86868b] uppercase tracking-wider">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="e.g. node-edge-01"
              className={inputClass}
              required
              autoFocus
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className="text-[13px] font-medium text-[#86868b] uppercase tracking-wider">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="e.g. edge01@fedlearn.internal"
              className={inputClass}
              required
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className="text-[13px] font-medium text-[#86868b] uppercase tracking-wider">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className={inputClass}
              required
            />
          </div>
          <div className="flex gap-3 mt-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isLoading}
              className="flex-1 bg-[#2c2c2e] hover:bg-[#3a3a3c] text-[#f5f5f7] py-3 rounded-full text-[15px] font-medium tracking-tight transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading || !username || !email || !password}
              className="flex-1 bg-[#f5f5f7] text-black hover:bg-white py-3 rounded-full text-[15px] font-medium tracking-tight transition-all duration-200 transform active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Creating...' : 'Create Client'}
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

  useEffect(() => { loadUsers(); }, [loadUsers]);

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
    if (!window.confirm("Are you sure you want to delete this client?")) return;
    try {
      await api.deleteUser(id);
      setUsers((prev) => prev.filter((u) => u.id !== id));
    } catch {
      setError('Failed to delete client.');
    }
  };

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-black text-[#f5f5f7] font-sans">
      <div className="h-24 flex items-center justify-between px-10 border-b border-[#2c2c2e] bg-[rgba(0,0,0,0.65)] backdrop-blur-3xl saturate-[1.8] sticky top-0 z-20">
        <div>
          <h1 className="text-[28px] font-semibold tracking-tight text-[#f5f5f7]">Node Network</h1>
          <p className="text-[15px] text-[#86868b] mt-0.5 tracking-tight">Manage edge devices and client credentials.</p>
        </div>
        <button
          onClick={() => setIsModalOpen(true)}
          className="flex items-center gap-2 bg-[#f5f5f7] text-black hover:bg-white px-5 py-2.5 rounded-full text-[15px] font-medium transition-all duration-200 transform active:scale-95"
        >
          <Plus className="w-[18px] h-[18px]" />
          Add Client
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-10 py-10 relative z-10 bg-black">
        {error && (
          <div className="mb-6 px-5 py-3 rounded-2xl bg-[#ff453a]/10 text-[#ff453a] text-[14px] font-medium">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center h-64 text-[#86868b]">
            Loading network data...
          </div>
        ) : users.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {users.map((user) => (
              <div key={user.id} className="bg-[#1c1c1e] rounded-[24px] p-6 flex flex-col gap-4 border border-[rgba(255,255,255,0.05)] hover:bg-[#2c2c2e]/60 transition-all">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-[#32d74b]/10 text-[#32d74b] flex items-center justify-center">
                      <Network className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-[17px] font-semibold tracking-tight">{user.username}</h3>
                      <p className="text-[13px] text-[#86868b]">{user.email}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDelete(user.id)}
                    className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-[#ff453a]/20 text-[#86868b] hover:text-[#ff453a] transition-colors"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
                <div className="bg-[#2c2c2e]/40 rounded-xl p-3 flex justify-between items-center text-[13px]">
                  <span className="text-[#86868b]">Node ID:</span>
                  <span className="font-mono text-[#f5f5f7] tracking-wider">{user.id}</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-64 text-[#86868b] gap-2">
            <p className="text-[17px]">No clients found.</p>
          </div>
        )}
      </div>

      <CreateClientModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSubmit={handleCreate}
        isLoading={isCreating}
      />
    </div>
  );
}
