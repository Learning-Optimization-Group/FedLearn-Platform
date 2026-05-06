import React, { useEffect, useMemo, useState } from 'react';
import * as api from '../services/apiServices';
import type { Project } from '../services/apiServices';
import DiskLoader from '../components/DiskLoader';
import { createLogger } from '../lib/logger';
import '../styles/Dashboard.css';
import '../styles/ClientsPage.css';

const log = createLogger('ModelsPage');

interface ModelSummary {
    modelName: string;
    modelType: string;
    projectCount: number;
    runningCount: number;
    completedCount: number;
    optimizers: string[];
}

function summarize(projects: Project[]): ModelSummary[] {
    const byModel = new Map<string, ModelSummary>();
    for (const p of projects) {
        const key = `${p.modelType}::${p.modelName}`;
        const existing = byModel.get(key);
        if (existing) {
            existing.projectCount += 1;
            if (p.status === 'RUNNING') existing.runningCount += 1;
            if (p.status === 'COMPLETED') existing.completedCount += 1;
            if (p.optimizer && !existing.optimizers.includes(p.optimizer)) {
                existing.optimizers.push(p.optimizer);
            }
        } else {
            byModel.set(key, {
                modelName: p.modelName,
                modelType: p.modelType,
                projectCount: 1,
                runningCount: p.status === 'RUNNING' ? 1 : 0,
                completedCount: p.status === 'COMPLETED' ? 1 : 0,
                optimizers: p.optimizer ? [p.optimizer] : [],
            });
        }
    }
    return Array.from(byModel.values()).sort((a, b) => b.projectCount - a.projectCount);
}

const ModelsPage: React.FC = () => {
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
                setError('Failed to load models.');
                log.error('fetchProjects failed', err);
            } finally {
                setIsLoading(false);
            }
        })();
    }, []);

    const summaries = useMemo(() => summarize(projects), [projects]);

    if (isLoading) return <DiskLoader message="Loading Models…" />;

    return (
        <div className="dashboard">
            <header className="dashboard-header">
                <div>
                    <h1>Models</h1>
                    <p className="section-subtitle">
                        Model architectures currently in rotation across projects.
                    </p>
                </div>
            </header>

            {error && (
                <div className="error-message" role="alert">
                    {error}
                </div>
            )}

            {summaries.length === 0 ? (
                <div className="empty-state">
                    <p>No model has been attached to a project yet.</p>
                    <p className="empty-sub">Create a project to start tracking a model here.</p>
                </div>
            ) : (
                <div className="client-grid">
                    {summaries.map((m) => (
                        <div className="client-card" key={`${m.modelType}-${m.modelName}`}>
                            <div className="client-card-header">
                                <div className="client-avatar" aria-hidden>
                                    {m.modelName.slice(0, 2).toUpperCase()}
                                </div>
                                <div className="client-identity">
                                    <h3>{m.modelName}</h3>
                                    <span className="client-email">{m.modelType}</span>
                                </div>
                            </div>
                            <dl className="client-meta">
                                <div>
                                    <dt>Projects</dt>
                                    <dd>{m.projectCount}</dd>
                                </div>
                                <div>
                                    <dt>Running</dt>
                                    <dd>{m.runningCount}</dd>
                                </div>
                                <div>
                                    <dt>Completed</dt>
                                    <dd>{m.completedCount}</dd>
                                </div>
                                <div>
                                    <dt>Optimizers</dt>
                                    <dd>{m.optimizers.join(', ') || '—'}</dd>
                                </div>
                            </dl>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default ModelsPage;
