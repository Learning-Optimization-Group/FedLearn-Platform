import React from 'react';
import { usePolling } from '../hooks/usePolling';
import { fetchMyRequests } from '../lib/api';
import type { MyAccessRequest, RequestStatus } from '../lib/types';

function badgeFor(status: RequestStatus): string {
  if (status === 'APPROVED') return 'badge badge--completed';
  if (status === 'DENIED')   return 'badge badge--failed';
  return 'badge badge--idle';
}

function fmtDate(iso?: string | null): string {
  if (!iso) return '—';
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

const MyRequestsView: React.FC = () => {
  const { data, loading, error, refresh } = usePolling<MyAccessRequest[]>(fetchMyRequests, [], 30_000);
  const requests = data ?? [];

  return (
    <>
      <div className="view-header">
        <div>
          <div className="view-header__title">My Requests</div>
          <div className="view-header__subtitle">Access requests you have sent</div>
        </div>
        <button className="btn-secondary" onClick={refresh}>Refresh</button>
      </div>
      {error && <div className="empty-state"><div className="empty-state__desc">Failed to load: {error}</div></div>}
      {loading && requests.length === 0 && (
        <div className="empty-state"><div className="empty-state__desc">Loading…</div></div>
      )}
      {!loading && requests.length === 0 && !error && (
        <div className="empty-state">
          <div className="empty-state__title">No requests yet</div>
          <div className="empty-state__desc">Requests you send from Discover will appear here with their status.</div>
        </div>
      )}
      {requests.map((r) => (
        <div key={r.id} className="request-row">
          <div>
            <div className="request-row__name">{r.projectName}</div>
            <div className="request-row__meta">
              Requested {fmtDate(r.requestedAt)}
              {r.decidedAt && ` · decided ${fmtDate(r.decidedAt)}${r.decidedByUsername ? ` by ${r.decidedByUsername}` : ''}`}
            </div>
          </div>
          <span className={badgeFor(r.status)}>{r.status}</span>
          {r.message && <div className="request-row__msg">"{r.message}"</div>}
        </div>
      ))}
    </>
  );
};

export default MyRequestsView;
