import React, { useState } from 'react';
import { usePolling } from '../hooks/usePolling';
import { fetchDiscoverProjects, requestAccess } from '../lib/api';
import RequestAccessDialog from '../components/RequestAccessDialog';
import type { DiscoverProject } from '../lib/types';

const DiscoverView: React.FC = () => {
  const { data, loading, error, refresh } = usePolling<DiscoverProject[]>(fetchDiscoverProjects, [], 30_000);
  const [requesting, setRequesting] = useState<DiscoverProject | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  const projects = data ?? [];

  const handleJoin = async (p: DiscoverProject) => {
    setSubmitting(true);
    const r = await requestAccess(p.id);
    setSubmitting(false);
    if (r.success && r.status === 'JOINED') {
      setFeedback(`Joined ${p.name}. It now appears in My Projects.`);
      refresh();
    } else if (r.success) {
      setFeedback(`Request sent for ${p.name}.`);
      refresh();
    } else {
      setFeedback(`Failed: ${r.error ?? 'unknown error'}`);
    }
  };

  const handleSubmitRequest = async (message: string) => {
    if (!requesting) return;
    setSubmitting(true);
    const r = await requestAccess(requesting.id, message);
    setSubmitting(false);
    setRequesting(null);
    if (r.success) {
      setFeedback(`Request sent for ${requesting.name}.`);
      refresh();
    } else {
      setFeedback(`Failed: ${r.error ?? 'unknown error'}`);
    }
  };

  const ctaFor = (p: DiscoverProject) => {
    if (p.myRequestStatus === 'PENDING')  return <span className="badge badge--idle">Pending</span>;
    if (p.myRequestStatus === 'APPROVED') return <span className="badge badge--completed">Approved</span>;
    if (p.myRequestStatus === 'DENIED')   return <span className="badge badge--failed">Denied</span>;
    if (p.visibility === 'PUBLIC') {
      return <button className="btn-primary" disabled={submitting} onClick={() => handleJoin(p)}>Join</button>;
    }
    return <button className="btn-primary" disabled={submitting} onClick={() => setRequesting(p)}>Request Access</button>;
  };

  return (
    <>
      <div className="view-header">
        <div>
          <div className="view-header__title">Discover</div>
          <div className="view-header__subtitle">Public projects to join and private projects to request access to</div>
        </div>
        <button className="btn-secondary" onClick={refresh}>Refresh</button>
      </div>
      {feedback && (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)',
                      borderRadius: 'var(--radius-md)', padding: '10px 14px', fontSize: 13,
                      color: 'var(--text-secondary)', marginBottom: 16 }}>
          {feedback}
        </div>
      )}
      {error && <div className="empty-state"><div className="empty-state__desc">Failed to load: {error}</div></div>}
      {loading && projects.length === 0 && (
        <div className="empty-state"><div className="empty-state__desc">Loading…</div></div>
      )}
      {!loading && projects.length === 0 && !error && (
        <div className="empty-state">
          <div className="empty-state__title">Nothing to discover yet</div>
          <div className="empty-state__desc">No public projects are open right now, and you have access to every private one you can see.</div>
        </div>
      )}
      <div className="card-grid">
        {projects.map((p) => (
          <div key={p.id} className="project-card">
            <div className="project-card__row">
              <div className="project-card__name">{p.name}</div>
              <span className={p.visibility === 'PUBLIC' ? 'badge badge--public' : 'badge badge--private'}>
                {p.visibility}
              </span>
            </div>
            <div className="project-card__meta">
              <span><strong>Owner:</strong> {p.ownerUsername}</span>
              <span><strong>Model:</strong> {p.modelType}</span>
              {p.description && <span style={{ marginTop: 4 }}>{p.description}</span>}
            </div>
            <div className="project-card__row">
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                {p.lastAccuracy != null ? `Last accuracy: ${(p.lastAccuracy * 100).toFixed(1)}%` : '—'}
              </span>
              {ctaFor(p)}
            </div>
          </div>
        ))}
      </div>
      {requesting && (
        <RequestAccessDialog
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
