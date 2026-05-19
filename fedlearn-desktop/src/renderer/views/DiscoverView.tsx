import React, { useState } from 'react';
import { usePolling } from '../hooks/usePolling';
import { fetchDiscoverProjects, requestAccess } from '../lib/api';
import type { DiscoverProject } from '../lib/types';

// ─── Status badge helper ──────────────────────────────────────────────────────

function RequestStatusBadge({ status }: { status: string }) {
  const styles: Record<string, React.CSSProperties> = {
    PENDING: {
      background: 'rgba(251, 191, 36, 0.12)',
      color: '#fbbf24',
      border: '1px solid rgba(251, 191, 36, 0.28)',
    },
    APPROVED: {
      background: 'rgba(52, 211, 153, 0.12)',
      color: '#34d399',
      border: '1px solid rgba(52, 211, 153, 0.28)',
    },
    DENIED: {
      background: 'rgba(248, 113, 113, 0.12)',
      color: '#f87171',
      border: '1px solid rgba(248, 113, 113, 0.28)',
    },
  };

  const labels: Record<string, string> = {
    PENDING: 'Pending',
    APPROVED: 'Joined',
    DENIED: 'Denied',
  };

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '3px 12px',
        borderRadius: 999,
        fontSize: 12,
        fontWeight: 600,
        letterSpacing: '0.02em',
        ...styles[status],
      }}
    >
      {labels[status] ?? status}
    </span>
  );
}

// ─── Visibility badge ─────────────────────────────────────────────────────────

function VisibilityBadge({ visibility }: { visibility: 'PUBLIC' | 'PRIVATE' }) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 4,
        padding: '2px 10px',
        borderRadius: 999,
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: '0.06em',
        textTransform: 'uppercase',
        background:
          visibility === 'PUBLIC'
            ? 'rgba(52, 211, 153, 0.1)'
            : 'rgba(251, 191, 36, 0.1)',
        color: visibility === 'PUBLIC' ? '#34d399' : '#fbbf24',
        border:
          visibility === 'PUBLIC'
            ? '1px solid rgba(52, 211, 153, 0.25)'
            : '1px solid rgba(251, 191, 36, 0.25)',
      }}
    >
      {visibility === 'PUBLIC' ? '◎' : '⊘'} {visibility}
    </span>
  );
}

// ─── Project Card ─────────────────────────────────────────────────────────────

interface DiscoverCardProps {
  project: DiscoverProject;
  submitting: boolean;
  onJoin: (p: DiscoverProject) => void;
  onRequestAccess: (p: DiscoverProject) => void;
}

function DiscoverCard({ project: p, submitting, onJoin, onRequestAccess }: DiscoverCardProps) {
  const renderCta = () => {
    if (p.myRequestStatus === 'PENDING') return <RequestStatusBadge status="PENDING" />;
    if (p.myRequestStatus === 'APPROVED') return <RequestStatusBadge status="APPROVED" />;
    if (p.myRequestStatus === 'DENIED') return <RequestStatusBadge status="DENIED" />;

    if (p.visibility === 'PUBLIC') {
      return (
        <button
          disabled={submitting}
          onClick={() => onJoin(p)}
          style={{
            padding: '6px 18px',
            borderRadius: 8,
            border: 'none',
            background: 'linear-gradient(135deg, var(--color-accent), #6246ea)',
            color: '#fff',
            fontSize: 13,
            fontWeight: 600,
            cursor: submitting ? 'not-allowed' : 'pointer',
            opacity: submitting ? 0.6 : 1,
            transition: 'opacity 0.2s',
          }}
        >
          {submitting ? 'Joining…' : 'Join'}
        </button>
      );
    }

    return (
      <button
        disabled={submitting}
        onClick={() => onRequestAccess(p)}
        style={{
          padding: '6px 18px',
          borderRadius: 8,
          border: '1px solid var(--color-border-strong)',
          background: 'var(--color-bg-elevated)',
          color: 'var(--color-text-primary)',
          fontSize: 13,
          fontWeight: 600,
          cursor: submitting ? 'not-allowed' : 'pointer',
          opacity: submitting ? 0.6 : 1,
          transition: 'all 0.2s',
        }}
      >
        Request Access
      </button>
    );
  };

  return (
    <div
      style={{
        background: 'var(--glass-bg)',
        border: '1px solid var(--color-border)',
        borderRadius: 16,
        padding: '20px 24px',
        display: 'flex',
        flexDirection: 'column',
        gap: 14,
        backdropFilter: 'blur(12px)',
        WebkitBackdropFilter: 'blur(12px)',
        transition: 'border-color 0.2s, box-shadow 0.2s',
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLDivElement).style.borderColor = 'var(--color-border-strong)';
        (e.currentTarget as HTMLDivElement).style.boxShadow = '0 8px 32px rgba(0,0,0,0.3)';
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLDivElement).style.borderColor = 'var(--color-border)';
        (e.currentTarget as HTMLDivElement).style.boxShadow = 'none';
      }}
    >
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontSize: 16,
              fontWeight: 600,
              color: 'var(--color-text-primary)',
              letterSpacing: '-0.01em',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            {p.name}
          </div>
          <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginTop: 3 }}>
            by {p.ownerUsername} · {p.modelType}
          </div>
        </div>
        <VisibilityBadge visibility={p.visibility} />
      </div>

      {/* Description */}
      {p.description && (
        <p
          style={{
            fontSize: 13,
            color: 'var(--color-text-secondary)',
            lineHeight: 1.55,
            margin: 0,
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
          }}
        >
          {p.description}
        </p>
      )}

      {/* Accuracy */}
      {p.lastAccuracy != null && (
        <div style={{ fontSize: 12, color: 'var(--color-text-secondary)' }}>
          Latest accuracy:{' '}
          <span style={{ fontWeight: 600, color: 'var(--color-text-primary)' }}>
            {(p.lastAccuracy * 100).toFixed(1)}%
          </span>
        </div>
      )}

      {/* CTA */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 4 }}>{renderCta()}</div>
    </div>
  );
}

// ─── Request message dialog ───────────────────────────────────────────────────

interface RequestDialogProps {
  project: DiscoverProject;
  submitting: boolean;
  onCancel: () => void;
  onSubmit: (message: string) => void;
}

function RequestDialog({ project, submitting, onCancel, onSubmit }: RequestDialogProps) {
  const [message, setMessage] = useState('');

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(0,0,0,0.55)',
        backdropFilter: 'blur(6px)',
        WebkitBackdropFilter: 'blur(6px)',
      }}
    >
      <div
        style={{
          width: 420,
          background: 'var(--glass-bg)',
          border: '1px solid var(--color-border-strong)',
          borderRadius: 20,
          padding: 32,
          display: 'flex',
          flexDirection: 'column',
          gap: 20,
          boxShadow: '0 32px 80px rgba(0,0,0,0.5)',
        }}
      >
        <div>
          <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--color-text-primary)' }}>
            Request Access
          </div>
          <div style={{ fontSize: 13, color: 'var(--color-text-secondary)', marginTop: 6 }}>
            Send a request to join <strong style={{ color: 'var(--color-text-primary)' }}>{project.name}</strong>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <label style={{ fontSize: 12, fontWeight: 500, color: 'var(--color-text-secondary)', letterSpacing: '0.02em' }}>
            Message to owner <span style={{ color: 'var(--color-text-muted)' }}>(optional)</span>
          </label>
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Explain why you'd like to join..."
            rows={3}
            maxLength={500}
            style={{
              width: '100%',
              padding: '10px 14px',
              background: 'var(--color-bg-tertiary)',
              border: '1px solid var(--color-border)',
              borderRadius: 10,
              color: 'var(--color-text-primary)',
              fontFamily: 'var(--font-sans)',
              fontSize: 13,
              resize: 'vertical',
              outline: 'none',
              boxSizing: 'border-box',
            }}
          />
        </div>

        <div style={{ display: 'flex', gap: 10 }}>
          <button
            disabled={submitting}
            onClick={() => onSubmit(message)}
            style={{
              flex: 1,
              padding: '10px 0',
              borderRadius: 10,
              border: 'none',
              background: 'linear-gradient(135deg, var(--color-accent), #6246ea)',
              color: '#fff',
              fontSize: 14,
              fontWeight: 600,
              cursor: submitting ? 'not-allowed' : 'pointer',
              opacity: submitting ? 0.6 : 1,
            }}
          >
            {submitting ? 'Sending…' : 'Send Request'}
          </button>
          <button
            onClick={onCancel}
            style={{
              flex: 1,
              padding: '10px 0',
              borderRadius: 10,
              border: '1px solid var(--color-border-strong)',
              background: 'var(--color-bg-elevated)',
              color: 'var(--color-text-secondary)',
              fontSize: 14,
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Main view ────────────────────────────────────────────────────────────────

const DiscoverView: React.FC = () => {
  const { data, loading, error, refresh } = usePolling<DiscoverProject[]>(
    fetchDiscoverProjects,
    [],
    30_000,
  );
  const [requesting, setRequesting] = useState<DiscoverProject | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [feedback, setFeedback] = useState<{ text: string; ok: boolean } | null>(null);

  const projects = data ?? [];

  const handleJoin = async (p: DiscoverProject) => {
    setSubmitting(true);
    const r = await requestAccess(p.id);
    setSubmitting(false);
    if (r.success && r.status === 'JOINED') {
      setFeedback({ text: `Joined ${p.name}. It now appears in My Projects.`, ok: true });
      refresh();
    } else if (r.success) {
      setFeedback({ text: `Request sent for ${p.name}.`, ok: true });
      refresh();
    } else {
      setFeedback({ text: `Failed: ${r.error ?? 'unknown error'}`, ok: false });
    }
  };

  const handleSubmitRequest = async (message: string) => {
    if (!requesting) return;
    const p = requesting;
    setSubmitting(true);
    const r = await requestAccess(p.id, message);
    setSubmitting(false);
    setRequesting(null);
    if (r.success) {
      setFeedback({ text: `Request sent for ${p.name}.`, ok: true });
      refresh();
    } else {
      setFeedback({ text: `Failed: ${r.error ?? 'unknown error'}`, ok: false });
    }
  };

  return (
    <>
      {/* Header */}
      <div
        style={{
          padding: '20px 28px 18px',
          borderBottom: '1px solid var(--color-border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 16,
          background: 'var(--glass-bg)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
        }}
      >
        <div>
          <div
            style={{
              fontSize: 22,
              fontWeight: 700,
              color: 'var(--color-text-primary)',
              letterSpacing: '-0.02em',
            }}
          >
            Discover Projects
          </div>
          <div style={{ fontSize: 13, color: 'var(--color-text-secondary)', marginTop: 4 }}>
            Browse public projects to join or request access to private ones.
          </div>
        </div>
        <button
          onClick={refresh}
          style={{
            padding: '7px 18px',
            borderRadius: 8,
            border: '1px solid var(--color-border-strong)',
            background: 'var(--color-bg-elevated)',
            color: 'var(--color-text-secondary)',
            fontSize: 13,
            fontWeight: 500,
            cursor: 'pointer',
            transition: 'all 0.15s',
            flexShrink: 0,
          }}
        >
          ↻ Refresh
        </button>
      </div>

      {/* Scrollable body */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '24px 28px' }}>
        {/* Feedback toast */}
        {feedback && (
          <div
            style={{
              marginBottom: 20,
              padding: '10px 16px',
              borderRadius: 10,
              fontSize: 13,
              fontWeight: 500,
              background: feedback.ok
                ? 'rgba(52, 211, 153, 0.1)'
                : 'rgba(248, 113, 113, 0.1)',
              color: feedback.ok ? '#34d399' : '#f87171',
              border: feedback.ok
                ? '1px solid rgba(52, 211, 153, 0.25)'
                : '1px solid rgba(248, 113, 113, 0.25)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <span>{feedback.text}</span>
            <button
              onClick={() => setFeedback(null)}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'inherit',
                cursor: 'pointer',
                fontSize: 16,
                lineHeight: 1,
                padding: '0 4px',
              }}
            >
              ×
            </button>
          </div>
        )}

        {/* Error */}
        {error && (
          <div
            style={{
              padding: '10px 16px',
              borderRadius: 10,
              background: 'rgba(248, 113, 113, 0.1)',
              color: '#f87171',
              border: '1px solid rgba(248, 113, 113, 0.2)',
              fontSize: 13,
              marginBottom: 20,
            }}
          >
            Failed to load: {error}
          </div>
        )}

        {/* Loading */}
        {loading && projects.length === 0 && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: 200,
              color: 'var(--color-text-muted)',
              fontSize: 14,
            }}
          >
            Loading…
          </div>
        )}

        {/* Empty state */}
        {!loading && projects.length === 0 && !error && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: 220,
              gap: 10,
            }}
          >
            <span style={{ fontSize: 36, opacity: 0.3 }}>◎</span>
            <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--color-text-primary)' }}>
              Nothing to discover yet
            </div>
            <div
              style={{
                fontSize: 13,
                color: 'var(--color-text-secondary)',
                textAlign: 'center',
                maxWidth: 300,
              }}
            >
              No public projects are open right now, and you have access to every private one
              you can see.
            </div>
          </div>
        )}

        {/* Project grid */}
        {projects.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
              gap: 16,
            }}
          >
            {projects.map((p) => (
              <DiscoverCard
                key={p.id}
                project={p}
                submitting={submitting}
                onJoin={handleJoin}
                onRequestAccess={setRequesting}
              />
            ))}
          </div>
        )}
      </div>

      {/* Request dialog */}
      {requesting && (
        <RequestDialog
          project={requesting}
          submitting={submitting}
          onCancel={() => setRequesting(null)}
          onSubmit={handleSubmitRequest}
        />
      )}
    </>
  );
};

export default DiscoverView;
