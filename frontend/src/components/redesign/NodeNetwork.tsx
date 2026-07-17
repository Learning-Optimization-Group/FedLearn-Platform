// =============================================================================
// FedLearn Frontend — Users (Platform account management, Ledger design system)
// =============================================================================
// This is the PLATFORM USER table, not a device inventory: it lists real user
// accounts and "Remove user" PERMANENTLY DELETES the account (DELETE /users/:id,
// an admin-only endpoint). Routed to PLATFORM_ADMIN only (see App.tsx / Sidebar).
//
// TODO: when the backend ships an owner-scoped device endpoint, split this into
// a real per-owner device page (list/add/remove the machines that train a
// project). Until then this stays admin-only account management.

import { useState, useEffect, useCallback } from 'react';
import * as api from '../../services/apiServices';
import { Plus, Trash2, AlertCircle, UserRound, Users } from 'lucide-react';
import type { User, RegisterData } from '../../services/apiServices';
import { Button, Card, Input, Modal, ConfirmDialog, FormField, Skeleton } from '../ui';
import { useAuth } from '../../context/AuthContext';
import { PageHeader } from './PageHeader';

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
      title="Add a user"
      footer={
        <>
          <Button type="button" variant="secondary" onClick={onClose} disabled={isLoading}>
            Cancel
          </Button>
          <Button
            type="submit"
            form="add-user-form"
            disabled={isLoading || !username || !email || !password}
          >
            {isLoading ? 'Adding…' : 'Add user'}
          </Button>
        </>
      }
    >
      <p className="-mt-1 mb-5 text-body text-fg-muted">
        Create a platform account so this person can sign in and join training runs.
      </p>
      <form id="add-user-form" onSubmit={handleSubmit} className="flex flex-col gap-5">
        <FormField label="Username">
          <Input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="e.g. anna"
            required
            autoFocus
          />
        </FormField>
        <FormField label="Email">
          <Input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="e.g. anna@example.com"
            required
          />
        </FormField>
        <FormField label="Password">
          <Input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
            required
          />
        </FormField>
      </form>
    </Modal>
  );
}

export function NodeNetwork() {
  const { currentUser } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  // Keep the whole target so the confirm dialog can name the account it deletes.
  const [pendingDelete, setPendingDelete] = useState<User | null>(null);

  // currentUser carries no id (see AuthContext), so identify "me" by username.
  const isSelf = (user: User) => currentUser?.username === user.username;

  const loadUsers = useCallback(async () => {
    try {
      setIsLoading(true);
      const res = await api.fetchUsers();
      setUsers(Array.isArray(res.data) ? res.data : []);
      setError('');
    } catch {
      setError('Failed to load users.');
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
      setError('Failed to add user.');
    } finally {
      setIsCreating(false);
    }
  };

  const confirmDelete = async () => {
    const target = pendingDelete;
    setPendingDelete(null);
    if (!target) return;
    // Hard guard: never delete your own signed-in account, even if the UI
    // somehow let the dialog open for it.
    if (isSelf(target)) {
      setError('You cannot remove your own account.');
      return;
    }
    try {
      await api.deleteUser(target.id);
      setUsers((prev) => prev.filter((u) => u.id !== target.id));
    } catch {
      setError('Failed to remove user.');
    }
  };

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
      <PageHeader title="Users" subtitle="Platform accounts that can sign in and join training runs.">
        <Button onClick={() => setIsModalOpen(true)}>
          <Plus strokeWidth={2} className="w-[18px] h-[18px]" />
          Add user
        </Button>
      </PageHeader>

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[1400px] px-6 py-6 md:px-10">
          {error && (
            <div className="mb-6 flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
              <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
              {error}
            </div>
          )}

          {isLoading ? (
            <div className="flex flex-col gap-2">
              {[0, 1, 2].map((i) => (
                <Card key={i} padding="md" className="flex items-center gap-3">
                  <Skeleton className="h-11 w-11 rounded-lg" />
                  <div className="flex flex-col gap-2">
                    <Skeleton className="h-4 w-28" />
                    <Skeleton className="h-3 w-40" />
                  </div>
                </Card>
              ))}
            </div>
          ) : users.length > 0 ? (
            <div className="flex flex-col gap-2">
              {users.map((user) => {
                const self = isSelf(user);
                return (
                  <Card key={user.id} padding="md" className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3 min-w-0">
                      <span className="icon-tile flex-shrink-0">
                        <UserRound strokeWidth={1.5} className="w-5 h-5" />
                      </span>
                      <div className="min-w-0">
                        <p className="text-body font-medium text-fg truncate">
                          {user.username}
                          {self && <span className="ml-2 text-caption text-fg-subtle">(you)</span>}
                        </p>
                        <p className="text-caption text-fg-muted truncate">{user.email}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 flex-shrink-0">
                      <span className="font-mono tabular-nums text-label text-fg-muted">#{user.id}</span>
                      <button
                        onClick={() => setPendingDelete(user)}
                        disabled={self}
                        className="flex h-8 w-8 items-center justify-center rounded-md text-fg-muted transition-colors duration-[120ms] hover:bg-surface-2 hover:text-danger focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface-1 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-fg-muted"
                        aria-label={self ? 'You cannot remove your own account' : `Remove user ${user.username}`}
                        title={self ? 'You cannot remove your own account' : undefined}
                      >
                        <Trash2 strokeWidth={1.5} className="w-4 h-4" />
                      </button>
                    </div>
                  </Card>
                );
              })}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center text-center gap-4 pt-16 md:pt-24">
              <div className="grid h-12 w-12 place-items-center rounded-pill bg-surface-2 text-fg-muted">
                <Users className="h-6 w-6" strokeWidth={1.5} />
              </div>
              <div className="max-w-sm">
                <p className="text-h4 font-semibold text-fg">No users yet</p>
                <p className="text-caption text-fg-muted mt-1">
                  Add a user so they can sign in and join training runs.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      <CreateClientModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSubmit={handleCreate}
        isLoading={isCreating}
      />

      <ConfirmDialog
        open={pendingDelete !== null}
        title="Remove user?"
        message={
          pendingDelete
            ? `This PERMANENTLY DELETES the account "${pendingDelete.username}" (${pendingDelete.email}). ` +
              'The user loses access immediately and this cannot be undone.'
            : ''
        }
        confirmLabel="Permanently delete"
        danger
        onConfirm={confirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
}
