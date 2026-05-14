import React, { useEffect, useState } from 'react';
import { getLastDatasetPath, setLastDatasetPath } from '../lib/api';
import type { ClientProject } from '../lib/types';

interface DatasetConfirmDialogProps {
  project: ClientProject;
  onCancel: () => void;
  onConfirm: (datasetPath: string) => void;
}

const DatasetConfirmDialog: React.FC<DatasetConfirmDialogProps> = ({ project, onCancel, onConfirm }) => {
  const [path, setPath] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const stored = await getLastDatasetPath(project.projectId);
      if (!cancelled) {
        setPath(stored);
        setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [project.projectId]);

  const handleBrowse = async () => {
    const r = await window.fedLearnAPI.selectDatasetPath();
    if (r.success && r.path) setPath(r.path);
  };

  const handleConfirm = async () => {
    await setLastDatasetPath(project.projectId, path);
    onConfirm(path);
  };

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className="modal">
        <div className="modal__title">Connect to {project.name}</div>
        <div className="modal__subtitle">
          The backend will assign your partition automatically. Confirm your dataset folder below — leave blank to use the container's bundled dataset.
        </div>
        <div className="modal__field">
          <label className="modal__label">Dataset folder (optional)</label>
          <input
            className="modal__input"
            type="text"
            value={path}
            placeholder="/path/to/dataset"
            onChange={(e) => setPath(e.target.value)}
            disabled={loading}
          />
          <span className="modal__hint">Last used path is remembered per project.</span>
        </div>
        <div className="modal__actions">
          <button className="btn-secondary" onClick={handleBrowse} disabled={loading}>Browse…</button>
          <button className="btn-secondary" onClick={onCancel}>Cancel</button>
          <button className="btn-primary" onClick={handleConfirm} disabled={loading}>Start Training</button>
        </div>
      </div>
    </div>
  );
};

export default DatasetConfirmDialog;
