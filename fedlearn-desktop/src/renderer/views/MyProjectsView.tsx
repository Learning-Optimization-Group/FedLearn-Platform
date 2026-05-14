import React from 'react';
import { usePolling } from '../hooks/usePolling';
import { fetchMyProjects } from '../lib/api';
import type { ClientProject, ProjectStatus } from '../lib/types';

interface MyProjectsViewProps {
  onTrain: (project: ClientProject) => void;
  trainingProjectId: string | null;
}

function statusBadge(status: ProjectStatus): string {
  switch (status) {
    case 'RUNNING':   return 'badge badge--running';
    case 'COMPLETED': return 'badge badge--completed';
    case 'FAILED':    return 'badge badge--failed';
    default:          return 'badge badge--idle';
  }
}

const MyProjectsView: React.FC<MyProjectsViewProps> = ({ onTrain, trainingProjectId }) => {
  const { data, loading, error, refresh } = usePolling<ClientProject[]>(fetchMyProjects, [], 30_000);
  const projects = data ?? [];

  return (
    <>
      <div className="view-header">
        <div>
          <div className="view-header__title">My Projects</div>
          <div className="view-header__subtitle">
            {projects.length} project{projects.length === 1 ? '' : 's'} you can train on
          </div>
        </div>
        <button className="btn-secondary" onClick={refresh}>Refresh</button>
      </div>
      {error && <div className="empty-state"><div className="empty-state__desc">Failed to load: {error}</div></div>}
      {loading && projects.length === 0 && (
        <div className="empty-state"><div className="empty-state__desc">Loading projects…</div></div>
      )}
      {!loading && projects.length === 0 && !error && (
        <div className="empty-state">
          <div className="empty-state__title">No projects yet</div>
          <div className="empty-state__desc">
            Head to Discover to find public projects you can join, or ask an owner to add you as a client.
          </div>
        </div>
      )}
      <div className="card-grid">
        {projects.map((p) => {
          const trainable = p.status === 'RUNNING';
          const busy = trainingProjectId === p.projectId;
          return (
            <div key={p.projectId} className="project-card">
              <div className="project-card__row">
                <div className="project-card__name">{p.name}</div>
                <span className={statusBadge(p.status)}>{p.status}</span>
              </div>
              <div className="project-card__meta">
                <span><strong>Model:</strong> {p.modelType} / {p.modelName}</span>
                <span><strong>Visibility:</strong> {p.visibility}</span>
              </div>
              <div className="project-card__row">
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  {trainable ? 'Ready to connect' : 'Waiting for owner to start training'}
                </span>
                <button
                  className="btn-primary"
                  disabled={!trainable || busy}
                  onClick={() => onTrain(p)}
                >
                  {busy ? 'Starting…' : 'Train'}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
};

export default MyProjectsView;
