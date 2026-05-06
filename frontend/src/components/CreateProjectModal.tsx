import React, { useState, useEffect } from 'react';
import '../styles/CreateProjectModal.css';

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

type ModelType = keyof typeof modelOptions;

interface CreateProjectModalProps {
    onSubmit: (projectData: {
        name: string;
        modelType: string;
        modelName: string;
        optimizer: string;
        pretrainEpochs: number;
    }) => void;
    onCancel: () => void;
    isLoading?: boolean;
}

const CreateProjectModal: React.FC<CreateProjectModalProps> = ({ onSubmit, onCancel, isLoading = false }) => {
    const [name, setName] = useState('');
    const [modelType, setModelType] = useState<ModelType>('CNN');
    const [modelName, setModelName] = useState(modelOptions.CNN.models[0]);
    const [optimizer, setOptimizer] = useState(modelOptions.CNN.optimizers[0]);
    const [pretrainEpochs, setPretrainEpochs] = useState(0);

    useEffect(() => {
        setModelName(modelOptions[modelType].models[0]);
        setOptimizer(modelOptions[modelType].optimizers[0]);
    }, [modelType]);

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        onSubmit({
            name,
            modelType,
            modelName,
            optimizer,
            pretrainEpochs: Number(pretrainEpochs)
        });
    };

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
            aria-labelledby="modal-title"
            aria-modal="true"
        >
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <h2 id="modal-title">Create New Project</h2>
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label htmlFor="project-name">Project Name</label>
                        <input
                            id="project-name"
                            type="text"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label htmlFor="model-type">Model Architecture</label>
                        <select
                            id="model-type"
                            value={modelType}
                            onChange={(e) => setModelType(e.target.value as ModelType)}
                        >
                            {Object.keys(modelOptions).map(type => (
                                <option key={type} value={type}>{type}</option>
                            ))}
                        </select>
                    </div>
                    <div className="form-group">
                        <label htmlFor="model-name">Model Name</label>
                        <select
                            id="model-name"
                            value={modelName}
                            onChange={(e) => setModelName(e.target.value)}
                        >
                            {modelOptions[modelType].models.map(name => (
                                <option key={name} value={name}>{name}</option>
                            ))}
                        </select>
                    </div>
                    <div className="form-group">
                        <label htmlFor="optimizer">Optimizer</label>
                        <select
                            id="optimizer"
                            value={optimizer}
                            onChange={(e) => setOptimizer(e.target.value)}
                        >
                            {modelOptions[modelType].optimizers.map(opt => (
                                <option key={opt} value={opt}>{opt}</option>
                            ))}
                        </select>
                    </div>
                    <div className="form-group">
                        <label htmlFor="pretrain-epochs">Pre-train Epochs</label>
                        <input
                            id="pretrain-epochs"
                            type="number"
                            value={pretrainEpochs}
                            onChange={(e) => setPretrainEpochs(Number(e.target.value))}
                            min="0"
                        />
                    </div>
                    <div className="modal-actions">
                        <button
                            type="button"
                            className="btn-cancel"
                            onClick={onCancel}
                            disabled={isLoading}
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            className="btn-submit"
                            disabled={isLoading}
                        >
                            {isLoading ? 'Creating...' : 'Create Project'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default CreateProjectModal;
