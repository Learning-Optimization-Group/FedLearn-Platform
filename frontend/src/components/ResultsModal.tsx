import React from 'react';
import '../styles/ResultsModal.css';

interface ProjectResult {
    id: string;
    serverRound: number;
    loss: number;
    accuracy: number;
    gpuUtilization?: number;
}

interface ResultsModalProps {
    results: ProjectResult[];
    projectName: string;
    onCancel: () => void;
}

const ResultsModal: React.FC<ResultsModalProps> = ({ results, projectName, onCancel }) => {
    const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (e.target === e.currentTarget) {
            onCancel();
        }
    };

    const handleEscapeKey = (e: React.KeyboardEvent) => {
        if (e.key === 'Escape') {
            onCancel();
        }
    };

    return (
        <div
            className="modal-backdrop"
            onClick={handleBackdropClick}
            onKeyDown={handleEscapeKey}
            role="dialog"
            aria-labelledby="results-modal-title"
            aria-modal="true"
        >
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <h2 id="results-modal-title">Training Results for "{projectName}"</h2>

                {(!results || results.length === 0) ? (
                    <p>No training results are available for this project yet.</p>
                ) : (
                    <div className="results-table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Round</th>
                                    <th>Loss</th>
                                    <th>Accuracy</th>
                                    <th>GPU Usage (MB)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {results.map(result => (
                                    <tr key={result.id}>
                                        <td>{result.serverRound}</td>
                                        <td>{result.loss ? result.loss.toFixed(4) : 'N/A'}</td>
                                        <td>{result.accuracy ? (result.accuracy * 100).toFixed(2) + '%' : 'N/A'}</td>
                                        <td>{result.gpuUtilization || 'N/A'}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}

                <div className="modal-actions">
                    <button className="btn-close" onClick={onCancel}>Close</button>
                </div>
            </div>
        </div>
    );
};

export default ResultsModal;
