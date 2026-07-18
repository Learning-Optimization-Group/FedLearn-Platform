// =============================================================================
// FedLearn Frontend — Users directory (role: PLATFORM_ADMIN, Ledger design system)
// =============================================================================
// Search-first paginated directory over GET /admin/users/search: debounced text
// search (username OR email), role + status filters, and a 25-per-page pager —
// all mirrored into the URL query string so a filtered view is linkable and
// survives refresh/back. Row-level "Manage" opens a drawer (Modal lg) with the
// identity block and the explicit account actions:
//   · Change role   — confirm-gated; 409 (last admin) surfaces inline.
//   · Suspend/Reactivate — confirm-gated; 409 guards (last active admin,
//     self-suspend) surface inline.
//   · Remove user   — the FE-1 permanent-delete flow, dialog copy unchanged.
// This is PLATFORM ACCOUNT management, not a device inventory: "Remove user"
// PERMANENTLY DELETES the account (DELETE /users/:id, admin-only).

import { useState, useEffect, useCallback, useRef, type ReactNode } from 'react';
import { useSearchParams } from 'react-router-dom';
import * as api from '../../services/apiServices';
import { errorMessage } from '../../services/apiServices';
import { Plus, Trash2, AlertCircle, Users } from 'lucide-react';
import type { AdminUser, Paged, RegisterData, Role, UserStatus } from '../../services/apiServices';
import {
  Button,
  Card,
  Input,
  Modal,
  ConfirmDialog,
  FormField,
  Select,
  Skeleton,
  StatusPill,
  SectionLabel,
  type StatusKind,
} from '../ui';
import { useAuth } from '../../context/AuthContext';
import { PageHeader } from './PageHeader';

const PAGE_SIZE = 25;
const SEARCH_DEBOUNCE_MS = 300;

const ROLE_OPTIONS: Role[] = ['USER', 'PROJECT_OWNER', 'PLATFORM_ADMIN'];
const ROLE_LABEL: Record<Role, string> = {
  USER: 'User',
  PROJECT_OWNER: 'Owner',
  PLATFORM_ADMIN: 'Admin',
};

const STATUS_OPTIONS: UserStatus[] = ['ACTIVE', 'SUSPENDED', 'PENDING'];
const STATUS_LABEL: Record<UserStatus, string> = {
  ACTIVE: 'Active',
  SUSPENDED: 'Suspended',
  PENDING: 'Pending',
};
const STATUS_KIND: Record<UserStatus, StatusKind> = {
  ACTIVE: 'completed',
  SUSPENDED: 'error',
  PENDING: 'pending',
};

// SectionLabel-styled table header cell (matches the admin tables).
const th = 'px-4 py-2.5 text-left text-[11px] font-semibold uppercase tracking-[0.08em] text-fg-muted';
const td = 'px-4 py-3 text-body text-fg align-middle';

/** Compact relative timestamp for the "Last active" column. */
function relativeTime(iso?: string): string {
  if (!iso) return '—';
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return '—';
  const diff = Date.now() - t;
  const minute = 60_000;
  const hour = 3_600_000;
  const day = 86_400_000;
  if (diff < minute) return 'just now';
  if (diff < hour) return `${Math.floor(diff / minute)}m ago`;
  if (diff < day) return `${Math.floor(diff / hour)}h ago`;
  if (diff < 30 * day) return `${Math.floor(diff / day)}d ago`;
  return new Date(t).toLocaleDateString();
}

/** Absolute date for the identity block ("Member since"). */
function formatDate(iso?: string): string {
  if (!iso) return '—';
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return '—';
  return new Date(t).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

/** Quiet, non-interactive role chip — deliberately NOT a live select (FE). */
function RoleChip({ role }: { role: Role }) {
  return (
    <span className="inline-flex items-center rounded-pill border border-hairline bg-surface-2 px-2.5 py-0.5 text-caption font-medium text-fg-muted">
      {ROLE_LABEL[role]}
    </span>
  );
}

/** One label/value pair in the drawer's identity block. */
function IdentityField({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0">
      <SectionLabel>{label}</SectionLabel>
      <p className="mt-0.5 truncate text-body text-fg">{value}</p>
    </div>
  );
}

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

  // The URL query string is the single source of truth for the directory
  // query, so a filtered view is shareable and survives refresh/back.
  const [searchParams, setSearchParams] = useSearchParams();
  const q = searchParams.get('q') ?? '';
  const roleFilter = searchParams.get('role') ?? '';
  const statusFilter = searchParams.get('status') ?? '';
  const page = Math.max(0, Number.parseInt(searchParams.get('page') ?? '0', 10) || 0);

  const [pageData, setPageData] = useState<Paged<AdminUser> | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  // Debounced search box: local echo of ?q, pushed to the URL after a pause.
  const [qInput, setQInput] = useState(q);

  // Add-user modal.
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);

  // Manage drawer. Only the id is stored — the row object is re-derived from
  // the current page so the drawer always reflects the latest server state
  // after a mutation.
  const [managedId, setManagedId] = useState<number | null>(null);
  const [roleDraft, setRoleDraft] = useState<Role>('USER');
  const [actionError, setActionError] = useState('');

  // Confirm-gated actions (each opens a ConfirmDialog above the drawer).
  const [pendingRole, setPendingRole] = useState<Role | null>(null);
  const [pendingStatus, setPendingStatus] = useState<UserStatus | null>(null);
  const [pendingDelete, setPendingDelete] = useState<AdminUser | null>(null);

  const items = pageData?.items ?? [];
  const managed = managedId === null ? null : (items.find((u) => u.id === managedId) ?? null);
  const total = pageData?.total ?? 0;
  const rangeStart = total === 0 ? 0 : page * PAGE_SIZE + 1;
  const rangeEnd = Math.min((page + 1) * PAGE_SIZE, total);
  const isFiltered = Boolean(q || roleFilter || statusFilter);

  // currentUser carries no id (see AuthContext), so identify "me" by username.
  const isSelf = useCallback(
    (user: Pick<AdminUser, 'username'>) => currentUser?.username === user.username,
    [currentUser?.username],
  );
  const managedSelf = managed !== null && isSelf(managed);

  /** Patch the URL query params; null/'' deletes a key. Replaces history. */
  const applyParams = useCallback(
    (patch: Record<string, string | null>) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          for (const [key, value] of Object.entries(patch)) {
            if (value === null || value === '') next.delete(key);
            else next.set(key, value);
          }
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  // Keep the search box in sync when ?q changes from outside (back/forward).
  useEffect(() => {
    setQInput(q);
  }, [q]);

  // Push the debounced search text into the URL; new query resets the pager.
  useEffect(() => {
    if (qInput === q) return;
    const timer = setTimeout(() => {
      applyParams({ q: qInput || null, page: null });
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [qInput, q, applyParams]);

  // Guards against out-of-order responses when the query changes rapidly.
  const loadSeq = useRef(0);
  const load = useCallback(async () => {
    const seq = ++loadSeq.current;
    setIsLoading(true);
    try {
      const res = await api.searchAdminUsers({
        q: q || undefined,
        role: (roleFilter || undefined) as Role | undefined,
        status: (statusFilter || undefined) as UserStatus | undefined,
        page,
        size: PAGE_SIZE,
      });
      if (seq !== loadSeq.current) return; // superseded by a newer query
      setPageData(res.data);
      setError('');
    } catch (err) {
      if (seq !== loadSeq.current) return;
      setError(errorMessage(err, 'Failed to load users.'));
    } finally {
      if (seq === loadSeq.current) setIsLoading(false);
    }
  }, [q, roleFilter, statusFilter, page]);

  useEffect(() => {
    load();
  }, [load]);

  const goToPage = (p: number) => applyParams({ page: p <= 0 ? null : String(p) });

  /** Merge a mutated user back into the current page (keeps unlisted fields). */
  const mergeUser = (updated: AdminUser) => {
    setPageData((prev) =>
      prev
        ? { ...prev, items: prev.items.map((u) => (u.id === updated.id ? { ...u, ...updated } : u)) }
        : prev,
    );
  };

  const handleCreate = async (data: RegisterData) => {
    try {
      setIsCreating(true);
      await api.createUser(data);
      setIsModalOpen(false);
      load();
    } catch (err) {
      setError(errorMessage(err, 'Failed to add user.'));
    } finally {
      setIsCreating(false);
    }
  };

  const openDrawer = (user: AdminUser) => {
    setManagedId(user.id);
    setRoleDraft(user.role);
    setActionError('');
  };

  const closeDrawer = () => {
    setManagedId(null);
    setPendingRole(null);
    setPendingStatus(null);
    setPendingDelete(null);
  };

  const confirmRoleChange = async () => {
    const target = managed;
    const nextRole = pendingRole;
    setPendingRole(null);
    if (!target || !nextRole) return;
    try {
      const res = await api.updateUserRole(target.id, nextRole);
      mergeUser(res.data);
      setRoleDraft(res.data.role);
      setActionError('');
    } catch (err) {
      // 409 = would demote the last admin — surface the backend's guard inline.
      setActionError(errorMessage(err, 'Could not change that role.'));
    }
  };

  const confirmStatusChange = async () => {
    const target = managed;
    const nextStatus = pendingStatus;
    setPendingStatus(null);
    if (!target || !nextStatus) return;
    try {
      const res = await api.updateUserStatus(target.id, nextStatus);
      mergeUser(res.data);
      setActionError('');
    } catch (err) {
      // 409 = suspending the last active admin, or an admin suspending themselves.
      setActionError(errorMessage(err, 'Could not change that account status.'));
    }
  };

  const confirmDelete = async () => {
    const target = pendingDelete;
    setPendingDelete(null);
    if (!target) return;
    // Hard guard: never delete your own signed-in account, even if the UI
    // somehow let the dialog open for it.
    if (isSelf(target)) {
      setActionError('You cannot remove your own account.');
      return;
    }
    try {
      await api.deleteUser(target.id);
      setPageData((prev) =>
        prev
          ? {
              ...prev,
              items: prev.items.filter((u) => u.id !== target.id),
              total: Math.max(0, prev.total - 1),
            }
          : prev,
      );
      setManagedId(null);
    } catch (err) {
      setActionError(errorMessage(err, 'Failed to remove user.'));
    }
  };

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
      <PageHeader title="Users" subtitle="Search, filter, and manage every platform account.">
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

          {/* Search + filters */}
          <div className="mb-4 flex flex-wrap items-center gap-3">
            <div className="w-full max-w-sm">
              <Input
                aria-label="Search users"
                placeholder="Search by username or email…"
                value={qInput}
                onChange={(e) => setQInput(e.target.value)}
              />
            </div>
            <div className="w-40">
              <Select
                aria-label="Filter by role"
                value={roleFilter}
                onChange={(e) => applyParams({ role: e.target.value || null, page: null })}
              >
                <option value="">All roles</option>
                {ROLE_OPTIONS.map((r) => (
                  <option key={r} value={r}>{ROLE_LABEL[r]}</option>
                ))}
              </Select>
            </div>
            <div className="w-40">
              <Select
                aria-label="Filter by status"
                value={statusFilter}
                onChange={(e) => applyParams({ status: e.target.value || null, page: null })}
              >
                <option value="">All statuses</option>
                {STATUS_OPTIONS.map((s) => (
                  <option key={s} value={s}>{STATUS_LABEL[s]}</option>
                ))}
              </Select>
            </div>
          </div>

          {isLoading && !pageData ? (
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
          ) : !isLoading && total === 0 && !isFiltered ? (
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
          ) : (
            <>
              <Card padding="none" className="overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead className="border-b border-hairline bg-surface-2">
                      <tr>
                        <th className={th}>User</th>
                        <th className={th}>Email</th>
                        <th className={th}>Role</th>
                        <th className={th}>Status</th>
                        <th className={th}>Owned</th>
                        <th className={th}>Last active</th>
                        <th className={th}><span className="sr-only">Actions</span></th>
                      </tr>
                    </thead>
                    <tbody>
                      {items.map((u) => {
                        const self = isSelf(u);
                        return (
                          <tr key={u.id} className="border-b border-hairline last:border-0">
                            <td className={`${td} font-medium`}>
                              <div className="min-w-0">
                                <p className="truncate">
                                  {u.username}
                                  {self && <span className="ml-2 text-caption font-normal text-fg-subtle">(you)</span>}
                                </p>
                                {u.displayName && (
                                  <p className="truncate text-caption font-normal text-fg-muted">{u.displayName}</p>
                                )}
                              </div>
                            </td>
                            <td className={`${td} text-fg-muted`}>{u.email}</td>
                            <td className={td}><RoleChip role={u.role} /></td>
                            <td className={td}>
                              {u.status ? (
                                <StatusPill status={STATUS_KIND[u.status]}>{STATUS_LABEL[u.status]}</StatusPill>
                              ) : (
                                '—'
                              )}
                            </td>
                            <td className={`${td} font-mono tabular-nums`}>{u.projectsOwned}</td>
                            <td className={`${td} text-fg-muted`}>{relativeTime(u.lastLoginAt)}</td>
                            <td className={`${td} text-right`}>
                              <Button
                                size="sm"
                                variant="ghost"
                                aria-label={`Manage ${u.username}`}
                                onClick={() => openDrawer(u)}
                              >
                                Manage
                              </Button>
                            </td>
                          </tr>
                        );
                      })}
                      {items.length === 0 && (
                        <tr>
                          <td className={`${td} text-fg-muted`} colSpan={7}>
                            No users match this search.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </Card>

              {/* Pager */}
              <div className="mt-3 flex items-center justify-between gap-3">
                <span className="font-mono tabular-nums text-label text-fg-muted">
                  {rangeStart}–{rangeEnd} of {total}
                </span>
                <div className="flex items-center gap-2">
                  <Button size="sm" variant="secondary" disabled={page === 0} onClick={() => goToPage(page - 1)}>
                    Previous
                  </Button>
                  <Button size="sm" variant="secondary" disabled={rangeEnd >= total} onClick={() => goToPage(page + 1)}>
                    Next
                  </Button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      <CreateClientModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSubmit={handleCreate}
        isLoading={isCreating}
      />

      {/* Manage drawer */}
      <Modal open={managed !== null} onClose={closeDrawer} title="Manage user" size="lg">
        {managed && (
          <div className="flex flex-col gap-6">
            {actionError && (
              <div className="flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
                <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                {actionError}
              </div>
            )}

            {/* Identity */}
            <div className="grid grid-cols-2 gap-x-6 gap-y-4">
              <IdentityField label="Display name" value={managed.displayName || '—'} />
              <IdentityField
                label="Username"
                value={
                  <>
                    {managed.username}
                    {managedSelf && <span className="ml-2 text-caption text-fg-subtle">(you)</span>}
                  </>
                }
              />
              <IdentityField label="Email" value={managed.email} />
              <IdentityField
                label="Status"
                value={
                  managed.status ? (
                    <StatusPill status={STATUS_KIND[managed.status]}>{STATUS_LABEL[managed.status]}</StatusPill>
                  ) : (
                    '—'
                  )
                }
              />
              <IdentityField label="Member since" value={formatDate(managed.createdAt)} />
              <IdentityField label="Last login" value={relativeTime(managed.lastLoginAt)} />
            </div>

            {/* Role */}
            <div className="flex flex-col gap-2">
              <SectionLabel>Role</SectionLabel>
              <div className="flex items-center gap-2">
                <div className="w-44">
                  <Select
                    aria-label={`New role for ${managed.username}`}
                    value={roleDraft}
                    onChange={(e) => setRoleDraft(e.target.value as Role)}
                  >
                    {ROLE_OPTIONS.map((r) => (
                      <option key={r} value={r}>{ROLE_LABEL[r]}</option>
                    ))}
                  </Select>
                </div>
                <Button
                  disabled={roleDraft === managed.role}
                  onClick={() => {
                    setActionError('');
                    setPendingRole(roleDraft);
                  }}
                >
                  Change role
                </Button>
              </div>
              <p className="text-caption text-fg-muted">Role changes take effect immediately.</p>
            </div>

            {/* Access */}
            <div className="flex flex-col gap-2">
              <SectionLabel>Access</SectionLabel>
              {managed.status === 'SUSPENDED' ? (
                <div className="flex flex-wrap items-center gap-3">
                  <p className="text-body text-fg-muted">
                    This account is suspended and blocked from all API access.
                  </p>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => {
                      setActionError('');
                      setPendingStatus('ACTIVE');
                    }}
                  >
                    Reactivate account
                  </Button>
                </div>
              ) : (
                <div className="flex flex-wrap items-center gap-3">
                  <p className="text-body text-fg-muted">
                    Suspending signs the user out and blocks all API access.
                  </p>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="text-danger hover:text-danger"
                    onClick={() => {
                      setActionError('');
                      setPendingStatus('SUSPENDED');
                    }}
                  >
                    Suspend account
                  </Button>
                </div>
              )}
            </div>

            {/* Danger zone */}
            <div className="flex flex-col gap-2">
              <SectionLabel>Danger zone</SectionLabel>
              <div className="flex flex-wrap items-center gap-3">
                <p className="text-body text-fg-muted">Permanently delete this account.</p>
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-danger hover:text-danger"
                  disabled={managedSelf}
                  aria-label={
                    managedSelf ? 'You cannot remove your own account' : `Remove user ${managed.username}`
                  }
                  title={managedSelf ? 'You cannot remove your own account' : undefined}
                  onClick={() => {
                    setActionError('');
                    setPendingDelete(managed);
                  }}
                >
                  <Trash2 strokeWidth={1.5} className="w-4 h-4" />
                  Remove user
                </Button>
              </div>
            </div>
          </div>
        )}
      </Modal>

      {/* Confirm: role change */}
      <ConfirmDialog
        open={pendingRole !== null && managed !== null}
        title="Change role?"
        message={
          pendingRole && managed
            ? `Change ${managed.username} from ${ROLE_LABEL[managed.role]} to ${ROLE_LABEL[pendingRole]}? ` +
              'Owners can create and run projects; Admins manage the whole platform.'
            : ''
        }
        confirmLabel="Change role"
        onConfirm={confirmRoleChange}
        onCancel={() => setPendingRole(null)}
      />

      {/* Confirm: suspend / reactivate */}
      <ConfirmDialog
        open={pendingStatus !== null && managed !== null}
        title={pendingStatus === 'SUSPENDED' ? 'Suspend account?' : 'Reactivate account?'}
        message={
          pendingStatus && managed
            ? pendingStatus === 'SUSPENDED'
              ? `${managed.username} will be signed out and blocked from all API access until reactivated.`
              : `${managed.username} will be able to sign in and use the API again.`
            : ''
        }
        confirmLabel={pendingStatus === 'SUSPENDED' ? 'Suspend' : 'Reactivate'}
        danger={pendingStatus === 'SUSPENDED'}
        onConfirm={confirmStatusChange}
        onCancel={() => setPendingStatus(null)}
      />

      {/* Confirm: permanent delete (FE-1 — copy is a tested contract) */}
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
