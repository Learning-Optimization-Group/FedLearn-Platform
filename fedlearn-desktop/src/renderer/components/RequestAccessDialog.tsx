import React, { useState } from 'react';
import type { DiscoverProject } from '../lib/types';

interface RequestAccessDialogProps {
  project: DiscoverProject;
  onCancel: () => void;
  onSubmit: (message: string) => void;
  submitting: boolean;
}

const RequestAccessDialog: React.FC<RequestAccessDialogProps> = ({ project, onCancel, onSubmit, submitting }) => {
  const [message, setMessage] = useState<string>('');
  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className="modal">
        <div className="modal__title">Request access to {project.name}</div>
        <div className="modal__subtitle">
          Owner <strong>{project.ownerUsername}</strong> will be notified. You can add a short note.
        </div>
        <div className="modal__field">
          <label className="modal__label">Note (optional)</label>
          <textarea
            className="modal__input"
            rows={3}
            maxLength={500}
            value={message}
            placeholder="e.g. I have CIFAR-10 data on a Jetson Orin and would like to contribute."
            onChange={(e) => setMessage(e.target.value)}
          />
        </div>
        <div className="modal__actions">
          <button className="btn-secondary" onClick={onCancel} disabled={submitting}>Cancel</button>
          <button className="btn-primary" onClick={() => onSubmit(message)} disabled={submitting}>
            {submitting ? 'Sending…' : 'Send Request'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default RequestAccessDialog;
