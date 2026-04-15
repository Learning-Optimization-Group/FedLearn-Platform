import React from 'react';
import '../styles/ModelCard.css';

interface Model {
    name: string;
    type: string;
    version: string;
    status: string;
}

interface ModelCardProps {
    model: Model;
}

const ModelCard: React.FC<ModelCardProps> = ({ model }) => {
    const statusClass = model.status === 'Training' ? 'status-training' : 'status-available';

    return (
        <div className="model-card">
            <div className="card-content">
                <h4 className="model-name">{model.name}</h4>
                <p className="model-type">{model.type}</p>
                <p className="model-version">{model.version}</p>
            </div>
            <div className="card-footer">
                <span className={`status-badge ${statusClass}`}>{model.status}</span>
            </div>
        </div>
    );
};

export default ModelCard;
