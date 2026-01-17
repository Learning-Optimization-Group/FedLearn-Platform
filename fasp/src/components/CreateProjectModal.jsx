import React, { useState, useEffect } from 'react';
import '../styles/CreateProjectModal.css'; // Styles for the modal

// This data could eventually come from a GET /api/config endpoint
const modelOptions = {
    CNN: {
        models: ["net", "ResNet", "VGGNet", "AlexNet"],
        optimizers: ["Adam", "SGD", "RMSprop", "AdamW"]
    },
    Transformer: {
        models: ["opt-125m", "bert-tiny"],
        optimizers: ["AdamW", "Adam"]
    }
};

const CreateProjectModal = ({ onSubmit, onCancel, isLoading }) => {
    const [name, setName] = useState('');
    const [modelType, setModelType] = useState('CNN');
    const [modelName, setModelName] = useState(modelOptions.CNN.models[0]);
    const [optimizer, setOptimizer] = useState(modelOptions.CNN.optimizers[0]);
    const [pretrainEpochs, setPretrainEpochs] = useState(0);

    // Effect to update model and optimizer when modelType changes
    useEffect(() => {
        setModelName(modelOptions[modelType].models[0]);
        setOptimizer(modelOptions[modelType].optimizers[0]);
    }, [modelType]);

    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit({ name, modelType, modelName, optimizer, pretrainEpochs: Number(pretrainEpochs) });
    };

    return (
        <div className="modal-backdrop" onClick={onCancel}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <h2>Create New Project</h2>
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label>Project Name</label>
                        <input type="text" value={name} onChange={(e) => setName(e.target.value)} required />
                    </div>
                    <div className="form-group">
                        <label>Model Architecture</label>
                        <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
                            {Object.keys(modelOptions).map(type => <option key={type} value={type}>{type}</option>)}
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Model Name</label>
                        <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                            {modelOptions[modelType].models.map(name => <option key={name} value={name}>{name}</option>)}
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Optimizer</label>
                        <select value={optimizer} onChange={(e) => setOptimizer(e.target.value)}>
                            {modelOptions[modelType].optimizers.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                        </select>
                    </div>
                    <div className="form-group">
                        <label>Pre-train Epochs</label>
                        <input type="number" value={pretrainEpochs} onChange={(e) => setPretrainEpochs(e.target.value)} min="0" />
                    </div>
                    <div className="modal-actions">
                        <button type="button" className="btn-cancel" onClick={onCancel} disabled={isLoading}>Cancel</button>
                        <button type="submit" className="btn-submit" disabled={isLoading}>{isLoading ? 'Creating...' : 'Create Project'}</button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default CreateProjectModal;