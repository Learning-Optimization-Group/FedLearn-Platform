import React, { useState } from 'react';
import '../styles/ProjectCard.css'; // Assuming styles are in this file
import * as api from '../services/apiServices';
import ResultsModal from './ResultsModal';
import CopyIcon from './CopyIcon';

const EditIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
    </svg>
);

// The 'onShowLogs' prop is now added to the function's arguments
const ProjectCard = ({ project, onToggleServer, onUpdateOptimizer, onShowLogs }) => {
    // Determine server status from the 'status' field for reliability
    const isServerRunning = project.status === "RUNNING";
    const statusText = project.status;
    const statusClassName = project.status?.toLowerCase();

    // State for local UI interactions
    const [isEditingOptimizer, setIsEditingOptimizer] = useState(false);
    const [optimizer, setOptimizer] = useState(project.optimizer);
    const [strategy, setStrategy] = useState('FedAvg');
    const [numRounds, setNumRounds] = useState(5);

    // State for the results modal, managed within this card
    const [isResultsModalOpen, setIsResultsModalOpen] = useState(false);
    const [results, setResults] = useState([]);
    const [isLoadingResults, setIsLoadingResults] = useState(false);
    const [error, setError] = useState('');

    const handleOptimizerSave = () => {
        onUpdateOptimizer(project.id, optimizer);
        setIsEditingOptimizer(false);
    };

    const handleToggleClick = () => {
        if (isServerRunning) {
            onToggleServer(project, true, null);
        } else {
            const startData = { strategy, numRounds: Number(numRounds) };
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
            console.error(err);
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
                    <div className={`status-indicator ${statusClassName}`}>
                        {statusText}
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
                        {/* Optimizer editing section */}
                        <div className="optimizer-section">
                            <strong>Optimizer:</strong>
                            {isEditingOptimizer ? (
                                <div className="edit-optimizer">
                                    <select value={optimizer} onChange={(e) => setOptimizer(e.target.value)}>
                                        {optimizerOptions.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                                    </select>
                                    <button className="save-btn" onClick={handleOptimizerSave}>✓</button>
                                    <button className="cancel-btn" onClick={() => setIsEditingOptimizer(false)}>✗</button>
                                </div>
                            ) : (
                                <div className="display-optimizer">
                                    <span className="optimizer-value">{project.optimizer}</span>
                                    <button className="edit-btn icon-btn" onClick={() => setIsEditingOptimizer(true)}>
                                        <EditIcon />
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>

                    {isServerRunning && (
                        <div className="port-info">
                            <span>Listening on Port: <strong>{project.serverPort}</strong></span>
                            <CopyIcon textToCopy={project.serverPort} />
                        </div>
                    )}

                    {!isServerRunning && (
                        <div className="start-config-section">
                            <h4>Run Configuration</h4>
                            <div className="config-grid">
                                <label>Strategy:</label>
                                <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                                    <option value="FedAvg">FedAvg</option>
                                    <option value="FedAdam">FedAdam</option>
                                    <option value="FedAdagrad">FedAdagrad</option>
                                </select>
                                <label>Rounds:</label>
                                <input type="number" min="1" value={numRounds} onChange={(e) => setNumRounds(e.target.value)} />
                            </div>
                        </div>
                    )}
                </div>

                {/* --- THIS IS THE UPDATED ACTIONS SECTION --- */}
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
                        disabled={!isServerRunning}
                        title={isServerRunning ? "Show live server logs" : "Start server to view logs"}
                    >
                        Show Live Logs
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
                        />
                        <span className="slider round"></span>
                    </label>
                </div>
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