import React, { useState, useEffect } from 'react';
import '../styles/ProjectCard.css';
import * as api from '../services/apiServices';
import ResultsModal from './ResultsModal';
import CopyIcon from './CopyIcon';
import { createLogger } from '../lib/logger';

const log = createLogger('ProjectCard');

const EditIcon: React.FC = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
    </svg>
);

const TrashIcon: React.FC = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="3 6 5 6 21 6"></polyline>
        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
    </svg>
);

interface Project {
    id: string;
    name: string;
    modelType: string;
    modelName: string;
    optimizer: string;
    status: 'RUNNING' | 'STOPPED' | 'COMPLETED' | 'FAILED';
    serverPort?: number;
}

interface ProjectCardProps {
    project: Project;
    onToggleServer: (project: Project, isCurrentlyRunning: boolean, startData: any) => void;
    onUpdateOptimizer: (projectId: string, newOptimizer: string) => void;
    onShowLogs: (projectId: string) => void;
    onDeleteProject: (projectId: string) => void;
}

const ProjectCard: React.FC<ProjectCardProps> = ({
    project,
    onToggleServer,
    onUpdateOptimizer,
    onShowLogs,
    onDeleteProject
}) => {
    const isServerRunning = project.status === "RUNNING";
    const statusText = project.status;
    const statusClassName = project.status?.toLowerCase();

    const [isEditingOptimizer, setIsEditingOptimizer] = useState(false);
    const [optimizer, setOptimizer] = useState(project.optimizer);
    const [strategy, setStrategy] = useState('FedAvg');
    const [numRounds, setNumRounds] = useState(5);
    const [minClients, setMinClients] = useState(2);

    const [isResultsModalOpen, setIsResultsModalOpen] = useState(false);
    const [results, setResults] = useState<any[]>([]);
    const [isLoadingResults, setIsLoadingResults] = useState(false);
    const [error, setError] = useState('');

    // Sync local state with prop changes
    useEffect(() => {
        setOptimizer(project.optimizer);
    }, [project.optimizer]);

    const handleOptimizerSave = () => {
        onUpdateOptimizer(project.id, optimizer);
        setIsEditingOptimizer(false);
    };

    const handleToggleClick = () => {
        if (isServerRunning) {
            onToggleServer(project, true, null);
        } else {
            const startData = {
                strategy,
                numRounds: Number(numRounds),
                minClients: Number(minClients)
            };
            onToggleServer(project, false, startData);
        }
    };

    const handleViewResultsClick = async () => {
        setIsLoadingResults(true);
        setError('');
        try {
            const response = await api.fetchProjectResults(project.id);
            setResults(response.data);
            setIsResultsModalOpen(true);
        } catch (err) {
            setError('Could not fetch results.');
            log.error('fetchProjectResults failed', err);
        } finally {
            setIsLoadingResults(false);
        }
    };

    const optimizerOptions = ["Adam", "AdamW", "SGD", "RMSprop"];

    return (
        <>
            <div className={`project-card ${statusClassName}`}>
                <div className="card-header">
                    <h3>{project.name}</h3>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div className={`status-indicator ${statusClassName}`}>
                            {statusText}
                        </div>
                        <button
                            className="icon-btn delete-btn"
                            onClick={() => onDeleteProject(project.id)}
                            title="Delete Project"
                            aria-label="Delete Project"
                            style={{ color: '#ff4d4f' }}
                        >
                            <TrashIcon />
                        </button>
                    </div>
                </div>

                <div className="card-body">
                    <div className="project-id-display">
                        <strong>Project ID:</strong>
                        <code>{project.id}</code>
                        <CopyIcon textToCopy={project.id} />
                    </div>
                    <div className="details-grid">
                        <p><strong>Type:</strong> {project.modelType}</p>
                        <p><strong>Model:</strong> {project.modelName}</p>
                        <div className="optimizer-section">
                            <strong>Optimizer:</strong>
                            {isEditingOptimizer ? (
                                <div className="edit-optimizer">
                                    <select
                                        value={optimizer}
                                        onChange={(e) => setOptimizer(e.target.value)}
                                        aria-label="Select optimizer"
                                    >
                                        {optimizerOptions.map(opt => (
                                            <option key={opt} value={opt}>{opt}</option>
                                        ))}
                                    </select>
                                    <button
                                        className="save-btn"
                                        onClick={handleOptimizerSave}
                                        aria-label="Save optimizer"
                                    >
                                        ✓
                                    </button>
                                    <button
                                        className="cancel-btn"
                                        onClick={() => setIsEditingOptimizer(false)}
                                        aria-label="Cancel editing"
                                    >
                                        ✗
                                    </button>
                                </div>
                            ) : (
                                <div className="display-optimizer">
                                    <span className="optimizer-value">{project.optimizer}</span>
                                    <button
                                        className="edit-btn icon-btn"
                                        onClick={() => setIsEditingOptimizer(true)}
                                        aria-label="Edit optimizer"
                                    >
                                        <EditIcon />
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>

                    {isServerRunning && project.serverPort && (
                        <div className="port-info">
                            <span>Listening on Port: <strong>{project.serverPort}</strong></span>
                            <CopyIcon textToCopy={String(project.serverPort)} />
                        </div>
                    )}

                    {!isServerRunning && (
                        <div className="start-config-section">
                            <h4>Run Configuration</h4>
                            <div className="config-grid">
                                <label htmlFor={`strategy-${project.id}`}>Strategy:</label>
                                <select
                                    id={`strategy-${project.id}`}
                                    value={strategy}
                                    onChange={(e) => setStrategy(e.target.value)}
                                >
                                    <option value="FedAvg">FedAvg</option>
                                    <option value="FedAdam">FedAdam</option>
                                    <option value="FedAdagrad">FedAdagrad</option>
                                </select>
                                <label htmlFor={`rounds-${project.id}`}>Rounds:</label>
                                <input
                                    id={`rounds-${project.id}`}
                                    type="number"
                                    min="1"
                                    value={numRounds}
                                    onChange={(e) => setNumRounds(Number(e.target.value))}
                                />
                                <label htmlFor={`min-clients-${project.id}`}>Minimum clients:</label>
                                <input
                                    id={`min-clients-${project.id}`}
                                    type="number"
                                    min="2"
                                    value={minClients}
                                    onChange={(e) => setMinClients(Number(e.target.value))}
                                />
                            </div>
                        </div>
                    )}
                </div>

                <div className="card-actions">
                    <button
                        className="action-btn results-btn"
                        onClick={handleViewResultsClick}
                        disabled={isLoadingResults}
                    >
                        {isLoadingResults ? 'Loading...' : 'View Results'}
                    </button>
                    <button
                        className="action-btn logs-btn"
                        onClick={() => onShowLogs(project.id)}
                        title="Show server logs"
                    >
                        View Logs
                    </button>
                </div>

                <div className="card-footer">
                    <span>Toggle Server</span>
                    <label className="toggle-switch">
                        <input
                            type="checkbox"
                            checked={isServerRunning}
                            onChange={handleToggleClick}
                            disabled={project.status === 'FAILED'}
                            aria-label="Toggle server"
                        />
                        <span className="slider round"></span>
                    </label>
                </div>

                {error && <p className="error-message">{error}</p>}
            </div>

            {isResultsModalOpen && (
                <ResultsModal
                    results={results}
                    projectName={project.name}
                    onCancel={() => setIsResultsModalOpen(false)}
                />
            )}
        </>
    );
};

export default ProjectCard;
