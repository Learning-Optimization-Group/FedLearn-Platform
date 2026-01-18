import React from 'react';
import '../styles/ResultsModal.css'; // We'll create this CSS file

const ResultsModal = ({ results, projectName, onCancel }) => {
    return (
        // The backdrop allows us to close the modal by clicking outside of it
        <div className="modal-backdrop" onClick={onCancel}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <h2>Training Results for "{projectName}"</h2>

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