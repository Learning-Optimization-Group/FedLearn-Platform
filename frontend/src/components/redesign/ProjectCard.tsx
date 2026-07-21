// =============================================================================
// FedLearn Frontend — ProjectCard (Ember design system)
// =============================================================================
// Full feature parity: delete, copy ID, copy port, start/stop, results, logs.

import { useState, useRef, useEffect, type KeyboardEvent } from 'react';
import { ResponsiveContainer, LineChart, Line, YAxis } from 'recharts';
import { Trash2, Copy, Check, MoreHorizontal, Edit3, Play, Square, Settings2, Clock } from 'lucide-react';
import { Card, Button, Input, Modal, FormField, MetricTile, StatusPill, ConfirmDialog, toStatusKind } from '../ui';
import { useAuth } from '../../context/AuthContext';
import type { Project, ProjectResult } from '../../services/apiServices';

interface ProjectCardProps {
  project: Project;
  results?: ProjectResult[];
  onOpenResults: () => void;
  onOpenLogs: () => void;
  onToggleServer: () => void;
  onEditProject: () => void;
  /**
   * Hard delete (DELETE /projects/{id}). Admin-only on the backend — the card
   * only routes here for PLATFORM_ADMIN. Non-admin owners go through
   * `onRequestDeletion` instead.
   */
  onDeleteProject: () => void;
  /**
   * Non-admin owner path: submit a deletion request (POST
   * /projects/{id}/deletion-request) for a platform admin to approve. When
   * provided, non-admins see "Request deletion" (with a reason capture) instead
   * of the hard "Delete project".
   */
  onRequestDeletion?: (reason: string) => void;
  /** A deletion request is already pending admin approval for this project. */
  deletionPending?: boolean;
  /**
   * Owner-only: opens the manage panel (visibility, join requests, members,
   * request-deletion). When provided, a "Manage" item appears in the menu.
   */
  onManageProject?: () => void;
}

/** Plain-language status label shown to people. */
function statusLabel(status: Project['status']): string {
  switch (status) {
    case 'INITIALIZING':
      return 'Preparing';
    case 'RUNNING':
      return 'Training';
    case 'COMPLETED':
      return 'Done';
    case 'FAILED':
      return 'Error';
    default:
      return 'Ready';
  }
}

function CopyButton({ text, label }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <button
      onClick={handleCopy}
      className="inline-flex items-center gap-1.5 text-fg-muted hover:text-fg transition-colors group/copy"
      title={`Copy ${label || text}`}
    >
      {label && <span className="text-caption font-medium">{label}:</span>}
      <code className="text-caption font-mono tabular-nums bg-surface-2 px-2 py-0.5 rounded-sm text-fg max-w-[120px] truncate">
        {text}
      </code>
      {copied ? (
        <Check className="w-3.5 h-3.5 text-success" strokeWidth={1.5} />
      ) : (
        <Copy className="w-3.5 h-3.5 opacity-0 group-hover/copy:opacity-100 transition-opacity" strokeWidth={1.5} />
      )}
    </button>
  );
}

export function ProjectCard({
  project,
  results = [],
  onOpenResults,
  onOpenLogs,
  onToggleServer,
  onEditProject,
  onDeleteProject,
  onRequestDeletion,
  deletionPending = false,
  onManageProject,
}: ProjectCardProps) {
  const { isAdmin } = useAuth();
  const [showMenu, setShowMenu] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [requestDelete, setRequestDelete] = useState(false);
  const [deleteReason, setDeleteReason] = useState('');

  // FE-13: WAI-ARIA menu-button semantics for the project-actions kebab. On open,
  // focus the first item; arrow keys roam, Home/End jump, Escape closes and returns
  // focus to the trigger, Tab closes without trapping. Selecting an item just closes
  // (the modal/dialog it opens manages its own focus via useFocusTrap).
  const menuRef = useRef<HTMLDivElement>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!showMenu) return;
    menuRef.current?.querySelector<HTMLElement>('[role="menuitem"]')?.focus();
  }, [showMenu]);

  const closeMenu = (restoreFocus = true) => {
    setShowMenu(false);
    if (restoreFocus) menuButtonRef.current?.focus();
  };

  const onMenuKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const items = Array.from(
      menuRef.current?.querySelectorAll<HTMLElement>('[role="menuitem"]') ?? [],
    );
    if (items.length === 0) return;
    const idx = items.indexOf(document.activeElement as HTMLElement);
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        items[(idx + 1) % items.length].focus();
        break;
      case 'ArrowUp':
        e.preventDefault();
        items[(idx - 1 + items.length) % items.length].focus();
        break;
      case 'Home':
        e.preventDefault();
        items[0].focus();
        break;
      case 'End':
        e.preventDefault();
        items[items.length - 1].focus();
        break;
      case 'Escape':
        e.preventDefault();
        closeMenu();
        break;
      case 'Tab':
        closeMenu(false);
        break;
    }
  };

  // Admins hard-delete; everyone else (project owners) files a deletion request
  // for an admin to approve. The card never routes an owner to DELETE — that
  // returns 403 for non-admins on the backend.
  const canRequestDeletion = !isAdmin && !!onRequestDeletion;

  const isRunning = project.status === 'RUNNING';
  const isFailed = project.status === 'FAILED';

  // Build accuracy trend from real results
  const accuracyTrend = results.slice(-10).map((r) => ({
    round: r.serverRound,
    accuracy: r.accuracy,
  }));

  // Rounds trained so far = the highest server round we have a result for.
  // Federated training has no fixed "total": a project can always train further
  // on new data / more rounds, and accuracy keeps moving — so a "% complete"
  // dial is the wrong abstraction. We show the round COUNT and let the status
  // pill (not a fake percentage) convey state.
  const roundsTrained = results.length > 0 ? results[results.length - 1].serverRound : 0;

  const latestAccuracy =
    accuracyTrend.length > 0
      ? `${(accuracyTrend[accuracyTrend.length - 1].accuracy * 100).toFixed(1)}%`
      : '—';

  return (
    <Card padding="lg" className="group relative flex w-full flex-col gap-5 text-fg font-sans">
      {/* Header Row */}
      <div className="flex justify-between items-start gap-3">
        <div className="flex-1 min-w-0">
          <h3 className="text-h4 font-semibold tracking-tight truncate">{project.name}</h3>
          <p className="text-caption text-fg-muted mt-0.5 truncate">
            {project.modelName} · {project.modelType} · {project.optimizer}
          </p>
          <div className="flex items-center gap-2 mt-1.5 flex-wrap">
            <StatusPill status={toStatusKind(project.status)}>{statusLabel(project.status)}</StatusPill>
            {deletionPending && <StatusPill status="pending">Deletion pending</StatusPill>}
          </div>
        </div>

        {/* Actions Menu */}
        <div className="relative">
          <button
            ref={menuButtonRef}
            onClick={() => setShowMenu(!showMenu)}
            className="w-8 h-8 flex items-center justify-center rounded-pill hover:bg-surface-2 text-fg-muted hover:text-fg transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-surface-1"
            aria-label="Project actions"
            aria-haspopup="menu"
            aria-expanded={showMenu}
          >
            <MoreHorizontal className="w-4 h-4" strokeWidth={1.5} />
          </button>
          {showMenu && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setShowMenu(false)} />
              <div
                ref={menuRef}
                role="menu"
                aria-label="Project actions"
                onKeyDown={onMenuKeyDown}
                className="absolute right-0 top-10 z-20 bg-surface-1 border border-line rounded-lg py-1 w-48 shadow-overlay"
              >
                <button
                  role="menuitem"
                  tabIndex={-1}
                  onClick={() => {
                    onEditProject();
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left text-body font-medium transition-colors flex items-center gap-2 text-fg hover:bg-surface-2"
                >
                  <Edit3 className="w-4 h-4" strokeWidth={1.5} />
                  Edit project
                </button>
                {onManageProject && (
                  <button
                    role="menuitem"
                    tabIndex={-1}
                    onClick={() => {
                      onManageProject();
                      setShowMenu(false);
                    }}
                    className="w-full px-4 py-2 text-left text-body font-medium transition-colors flex items-center gap-2 text-fg hover:bg-surface-2"
                  >
                    <Settings2 className="w-4 h-4" strokeWidth={1.5} />
                    Manage access
                  </button>
                )}
                <div className="h-px bg-hairline my-1" />
                {deletionPending ? (
                  <div
                    role="menuitem"
                    aria-disabled="true"
                    tabIndex={-1}
                    className="w-full px-4 py-2 text-left text-body font-medium flex items-center gap-2 text-warning"
                  >
                    <Clock className="w-4 h-4" strokeWidth={1.5} />
                    Deletion pending
                  </div>
                ) : canRequestDeletion ? (
                  <button
                    role="menuitem"
                    tabIndex={-1}
                    onClick={() => {
                      setRequestDelete(true);
                      setShowMenu(false);
                    }}
                    className="w-full px-4 py-2 text-left text-body font-medium transition-colors flex items-center gap-2 text-danger hover:bg-surface-2"
                  >
                    <Trash2 className="w-4 h-4" strokeWidth={1.5} />
                    Request deletion
                  </button>
                ) : (
                  <button
                    role="menuitem"
                    tabIndex={-1}
                    onClick={() => {
                      setConfirmDelete(true);
                      setShowMenu(false);
                    }}
                    className="w-full px-4 py-2 text-left text-body font-medium transition-colors flex items-center gap-2 text-danger hover:bg-surface-2"
                  >
                    <Trash2 className="w-4 h-4" strokeWidth={1.5} />
                    Delete project
                  </button>
                )}
              </div>
            </>
          )}
        </div>

      </div>

      {/* Project ID & Port — Copyable */}
      <div className="flex flex-wrap items-center gap-3">
        <CopyButton text={project.id} label="ID" />
        {isRunning && project.serverPort && (
          <CopyButton text={String(project.serverPort)} label="Port" />
        )}
      </div>

      {/* Metrics — one strip, divided tiles */}
      <div className="grid grid-cols-2 rounded-lg border border-hairline">
        <div className="px-4 py-3">
          <MetricTile label="Rounds" value={roundsTrained} />
        </div>
        <div className="px-4 py-3 border-l border-hairline">
          <MetricTile
            label="Accuracy"
            value={latestAccuracy}
            sparkline={
              accuracyTrend.length > 1 ? (
                <div className="h-8 w-full">
                  <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
                    <LineChart data={accuracyTrend}>
                      <YAxis domain={['auto', 'auto']} hide />
                      <Line type="monotone" dataKey="accuracy" stroke="var(--color-accent)" strokeWidth={2} dot={false} isAnimationActive={true} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : undefined
            }
          />
        </div>
      </div>

      {/* Action Row — Results/Logs quiet on the left, the lifecycle action on the right */}
      <div className="flex items-center gap-1 mt-1">
        <Button variant="ghost" size="sm" onClick={onOpenResults}>
          Results
        </Button>
        <Button variant="ghost" size="sm" onClick={onOpenLogs}>
          Logs
        </Button>
        <Button
          variant={isRunning ? 'danger' : 'primary'}
          size="sm"
          onClick={onToggleServer}
          disabled={isFailed}
          className="ml-auto"
        >
          {isRunning ? (
            <>
              <Square className="w-3.5 h-3.5" strokeWidth={2} /> Stop
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5" strokeWidth={2} /> Start
            </>
          )}
        </Button>
      </div>

      {/* Admin hard delete (DELETE /projects/{id}). */}
      <ConfirmDialog
        open={confirmDelete}
        title="Delete project?"
        message={`This permanently deletes "${project.name}" and its results. This can't be undone.`}
        confirmLabel="Delete"
        cancelLabel="Cancel"
        danger
        onConfirm={() => {
          onDeleteProject();
          setConfirmDelete(false);
        }}
        onCancel={() => setConfirmDelete(false)}
      />

      {/* Owner deletion request (POST /projects/{id}/deletion-request) — a
          platform admin approves the actual delete. */}
      <Modal
        open={requestDelete}
        onClose={() => setRequestDelete(false)}
        title="Request project deletion"
        size="sm"
        showClose={false}
        footer={
          <>
            <Button variant="secondary" onClick={() => setRequestDelete(false)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={() => {
                onRequestDeletion?.(deleteReason.trim());
                setRequestDelete(false);
                setDeleteReason('');
              }}
            >
              Request deletion
            </Button>
          </>
        }
      >
        <div className="flex flex-col gap-3">
          <p className="text-body text-fg-muted">
            Deleting “{project.name}” is permanent and must be approved by a platform admin.
          </p>
          <FormField label="Reason (optional)" help="Shown to the platform admin reviewing the request.">
            <Input
              value={deleteReason}
              onChange={(e) => setDeleteReason(e.target.value)}
              placeholder="Why should this project be deleted?"
            />
          </FormField>
        </div>
      </Modal>
    </Card>
  );
}
