// =============================================================================
// FedLearn Frontend — Devices (Client Management, Ember design system)
// =============================================================================
// Allows tracking, creating, and deleting FL clients (devices).

import { useState, useEffect, useCallback } from 'react';
import * as api from '../../services/apiServices';
import { Plus, Trash2, AlertCircle, Smartphone } from 'lucide-react';
import type { User, RegisterData } from '../../services/apiServices';
import { Button, Card, Input, Modal, ConfirmDialog, Skeleton } from '../ui';
import { BrandMark } from '../brand';
import { PageHeader } from './PageHeader';

interface CreateClientModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: RegisterData) => void;
  isLoading?: boolean;
}

const labelClass = 'text-label font-medium text-fg';

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
          <Smartphone strokeWidth={1.5} className="w-5 h-5 text-accent" />
          Add a device
        </span>
      }
    >
      <p className="-mt-1 mb-5 text-body text-fg-muted">
        Give the device a name and sign-in so it can join your training runs.
      </p>
      <form onSubmit={handleSubmit} className="flex flex-col gap-5">
        <div className="flex flex-col gap-1.5">
          <label className={labelClass}>Device name</label>
          <Input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="e.g. laptop-anna"
            required
            autoFocus
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <label className={labelClass}>Email</label>
          <Input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="e.g. anna@example.com"
            required
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <label className={labelClass}>Password</label>
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
            {isLoading ? 'Adding…' : 'Add device'}
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
      setError('Failed to load your devices.');
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
      setError('Failed to add device.');
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
      setError('Failed to remove device.');
    }
  };

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
      <PageHeader title="Devices" subtitle="The devices that train your models, together.">
        <Button onClick={() => setIsModalOpen(true)}>
          <Plus strokeWidth={2} className="w-[18px] h-[18px]" />
          Add device
        </Button>
      </PageHeader>

      <div className="flex-1 overflow-y-auto px-6 md:px-10 py-8 relative z-10 bg-canvas">
        {error && (
          <div className="mb-6 flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
            <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[0, 1, 2].map((i) => (
              <Card key={i} padding="lg" className="flex flex-col gap-4">
                <div className="flex items-center gap-3">
                  <Skeleton className="h-11 w-11 rounded-full" />
                  <div className="flex flex-col gap-2">
                    <Skeleton className="h-4 w-28" />
                    <Skeleton className="h-3 w-40" />
                  </div>
                </div>
                <Skeleton className="h-10 w-full" />
              </Card>
            ))}
          </div>
        ) : users.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {users.map((user) => (
              <Card key={user.id} padding="lg" className="flex flex-col gap-4 transition-colors duration-[160ms] hover:bg-surface-2 hover:border-accent/25">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3 min-w-0">
                    <span className="icon-tile flex-shrink-0">
                      <Smartphone strokeWidth={1.5} className="w-5 h-5" />
                    </span>
                    <div className="min-w-0">
                      <h3 className="text-h4 font-display text-fg truncate">{user.username}</h3>
                      <p className="text-label text-fg-muted truncate">{user.email}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setPendingDeleteId(user.id)}
                    className="w-8 h-8 flex items-center justify-center rounded-pill hover:bg-surface-2 text-fg-muted hover:text-danger transition-colors duration-[120ms] flex-shrink-0"
                    aria-label="Remove device"
                  >
                    <Trash2 strokeWidth={1.5} className="w-4 h-4" />
                  </button>
                </div>
                <div className="bg-surface-2 border border-hairline rounded-lg p-3 flex justify-between items-center text-label">
                  <span className="text-fg-muted">Device #</span>
                  <span className="font-mono tabular-nums text-fg">{user.id}</span>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center text-center gap-5 mt-16 md:mt-24">
            <div className="grid h-20 w-20 place-items-center rounded-card border border-hairline bg-surface-1">
              <BrandMark size={48} />
            </div>
            <div className="max-w-sm">
              <p className="text-h4 font-display text-fg">No devices yet</p>
              <p className="text-body text-fg-muted mt-1.5">
                Add a device so it can join your training runs and help teach your models.
              </p>
            </div>
            <Button size="lg" onClick={() => setIsModalOpen(true)}>
              <Plus strokeWidth={2} className="w-[18px] h-[18px]" />
              Add your first device
            </Button>
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
        title="Remove device?"
        message="This device won't be able to join training until it's added again."
        confirmLabel="Remove"
        danger
        onConfirm={confirmDelete}
        onCancel={() => setPendingDeleteId(null)}
      />
    </div>
  );
}
