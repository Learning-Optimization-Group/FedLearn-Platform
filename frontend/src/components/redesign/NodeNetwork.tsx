// =============================================================================
// FedLearn Frontend — Node Network (Client Management)
// =============================================================================
// Allows tracking, creating, and deleting FL clients (users).

import { useState, useEffect, useCallback } from 'react';
import * as api from '../../services/apiServices';
import { Plus, Trash2, Shield, Network } from 'lucide-react';
import type { User, RegisterData } from '../../services/apiServices';
import { Button, Card, Input, Modal, ConfirmDialog } from '../ui';

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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ username, email, password });
  };

  return (
    <Modal
      open={isOpen}
      onClose={onClose}
      title={
        <span className="flex items-center gap-2">
          <Shield strokeWidth={1.5} className="w-5 h-5 text-success" />
          Create Client Node
        </span>
      }
    >
      <form onSubmit={handleSubmit} className="flex flex-col gap-5">
        <div className="flex flex-col gap-2">
          <label className="text-caption font-medium text-fg-muted uppercase tracking-wide">Username</label>
          <Input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="e.g. node-edge-01"
            required
            autoFocus
          />
        </div>
        <div className="flex flex-col gap-2">
          <label className="text-caption font-medium text-fg-muted uppercase tracking-wide">Email</label>
          <Input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="e.g. edge01@fedlearn.internal"
            required
          />
        </div>
        <div className="flex flex-col gap-2">
          <label className="text-caption font-medium text-fg-muted uppercase tracking-wide">Password</label>
          <Input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
            required
          />
        </div>
        <div className="flex gap-3 mt-2">
          <Button
            type="button"
            variant="secondary"
            onClick={onClose}
            disabled={isLoading}
            className="flex-1"
          >
            Cancel
          </Button>
          <Button
            type="submit"
            disabled={isLoading || !username || !email || !password}
            className="flex-1"
          >
            {isLoading ? 'Creating...' : 'Create Client'}
          </Button>
        </div>
      </form>
    </Modal>
  );
}

export function NodeNetwork() {
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState<number | null>(null);

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

  const confirmDelete = async () => {
    if (pendingDeleteId === null) return;
    const id = pendingDeleteId;
    setPendingDeleteId(null);
    try {
      await api.deleteUser(id);
      setUsers((prev) => prev.filter((u) => u.id !== id));
    } catch {
      setError('Failed to delete client.');
    }
  };

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
      <div className="h-24 flex items-center justify-between px-10 border-b border-hairline bg-canvas/65 backdrop-blur-xl sticky top-0 z-20">
        <div>
          <h1 className="text-h2 text-fg">Node Network</h1>
          <p className="text-body text-fg-muted mt-0.5">Manage edge devices and client credentials.</p>
        </div>
        <Button onClick={() => setIsModalOpen(true)}>
          <Plus strokeWidth={1.5} className="w-[18px] h-[18px]" />
          Add Client
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto px-10 py-10 relative z-10 bg-canvas">
        {error && (
          <div className="mb-6 px-5 py-3 rounded-card bg-surface-1 border border-hairline text-danger text-body font-medium">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center h-64 text-fg-muted">
            Loading network data...
          </div>
        ) : users.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {users.map((user) => (
              <Card key={user.id} padding="lg" className="flex flex-col gap-4 hover:bg-surface-2 transition-colors duration-[160ms]">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-pill bg-surface-2 text-success flex items-center justify-center">
                      <Network strokeWidth={1.5} className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-h4 text-fg">{user.username}</h3>
                      <p className="text-label text-fg-muted">{user.email}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setPendingDeleteId(user.id)}
                    className="w-8 h-8 flex items-center justify-center rounded-pill hover:bg-surface-2 text-fg-muted hover:text-danger transition-colors duration-[120ms]"
                  >
                    <Trash2 strokeWidth={1.5} className="w-4 h-4" />
                  </button>
                </div>
                <div className="bg-surface-2 rounded-sm p-3 flex justify-between items-center text-label">
                  <span className="text-fg-muted">Node ID:</span>
                  <span className="font-mono tabular-nums text-fg">{user.id}</span>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-64 text-fg-muted gap-2">
            <p className="text-h4">No clients found.</p>
          </div>
        )}
      </div>

      <CreateClientModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSubmit={handleCreate}
        isLoading={isCreating}
      />

      <ConfirmDialog
        open={pendingDeleteId !== null}
        title="Delete client"
        message="Are you sure you want to delete this client?"
        confirmLabel="Delete"
        danger
        onConfirm={confirmDelete}
        onCancel={() => setPendingDeleteId(null)}
      />
    </div>
  );
}
