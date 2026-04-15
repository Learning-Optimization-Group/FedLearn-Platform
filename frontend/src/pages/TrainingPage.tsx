import React, { useEffect, useMemo, useState } from 'react';
import * as api from '../services/apiServices';
import type { Project } from '../services/apiServices';
import DiskLoader from '../components/DiskLoader';
import '../styles/Dashboard.css';
import '../styles/ClientsPage.css';

interface StatusBucket {
    label: string;
    key: Project['status'];
    color: string;
    count: number;
}

const TrainingPage: React.FC = () => {
    const [projects, setProjects] = useState<Project[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        (async () => {
            try {
                setIsLoading(true);
                const res = await api.fetchProjects();
                setProjects(Array.isArray(res.data) ? res.data : []);
            } catch (err) {
                setError('Failed to load training activity.');
                console.error(err);
            } finally {
                setIsLoading(false);
            }
        })();
    }, []);

    const buckets = useMemo<StatusBucket[]>(() => [
        { label: 'Running', key: 'RUNNING', color: '#10b981', count: projects.filter(p => p.status === 'RUNNING').length },
        { label: 'Stopped', key: 'STOPPED', color: '#6b7280', count: projects.filter(p => p.status === 'STOPPED').length },
        { label: 'Completed', key: 'COMPLETED', color: '#2563eb', count: projects.filter(p => p.status === 'COMPLETED').length },
        { label: 'Failed', key: 'FAILED', color: '#dc2626', count: projects.filter(p => p.status === 'FAILED').length },
    ], [projects]);

    if (isLoading) return <DiskLoader message="Loading Training Activity…" />;

    return (
        <div className="dashboard">
            <header className="dashboard-header">
                <div>
                    <h1>Training Activity</h1>
                    <p className="section-subtitle">
                        Aggregate health of every federated project on this server.
                    </p>
                </div>
            </header>

            {error && (
                <div className="error-message" role="alert">
                    {error}
                </div>
            )}

            <div className="training-stats">
                {buckets.map((b) => (
                    <div className="stat-card" key={b.key}>
                        <span className="stat-dot" style={{ background: b.color }} aria-hidden />
                        <div className="stat-body">
                            <span className="stat-number">{b.count}</span>
                            <span className="stat-label">{b.label}</span>
                        </div>
                    </div>
                ))}
            </div>

            <h2 className="section-title">All Projects</h2>
            {projects.length === 0 ? (
                <div className="empty-state">
                    <p>No projects yet.</p>
                </div>
            ) : (
                <div className="training-table-wrapper">
                    <table className="training-table">
                        <thead>
                            <tr>
                                <th>Project</th>
                                <th>Model</th>
                                <th>Optimizer</th>
                                <th>Status</th>
                                <th>Server Port</th>
                            </tr>
                        </thead>
                        <tbody>
                            {projects.map((p) => (
                                <tr key={p.id}>
                                    <td>{p.name}</td>
                                    <td>
                                        {p.modelName}
                                        <span className="model-type"> · {p.modelType}</span>
                                    </td>
                                    <td>{p.optimizer || '—'}</td>
                                    <td>
                                        <span className={`pill pill-${p.status.toLowerCase()}`}>
                                            {p.status}
                                        </span>
                                    </td>
                                    <td>{p.serverPort ?? '—'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default TrainingPage;
