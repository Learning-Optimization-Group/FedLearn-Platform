// =============================================================================
// FedLearn Desktop — Model Playground ("Use a model")
// =============================================================================
// Pick a trained model and run inference on it. Image models take an uploaded
// image; tabular models take a numeric vector. Inference runs server-side via
// the Main-process InferenceService (reached over IPC).
// =============================================================================

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FlaskConical, Upload, Sparkles, AlertTriangle } from 'lucide-react';
import type { InferableModel, InferenceResult } from '../inference.types';

// Reject oversized files before FileReader pulls them fully into renderer memory.
// The preload/IPC/backend layers also bound the encoded size, but this stops a
// multi-hundred-MB pick from freezing the UI before those checks ever run.
const MAX_IMAGE_FILE_BYTES = 10 * 1024 * 1024;

function parseVector(raw: string): number[] {
  return raw
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number)
    .filter((n) => Number.isFinite(n));
}

const ModelPlayground: React.FC = () => {
  const [models, setModels] = useState<InferableModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState('');
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [imageName, setImageName] = useState('');
  const [vectorText, setVectorText] = useState('');
  const [textInput, setTextInput] = useState('');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await window.fedLearnAPI.listModels();
        if (res.success && res.models) {
          setModels(res.models);
          if (res.models.length > 0) setSelectedId(res.models[0].projectId);
        } else {
          setError(res.error || 'Failed to load models.');
        }
      } catch {
        setError('Failed to load models.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const selected = useMemo(
    () => models.find((m) => m.projectId === selectedId) ?? null,
    [models, selectedId],
  );

  useEffect(() => {
    setResult(null);
    setError('');
    setImageDataUrl(null);
    setImageName('');
    setVectorText('');
    setTextInput('');
  }, [selectedId]);

  const handleFile = useCallback((file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please choose an image file.');
      return;
    }
    if (file.size > MAX_IMAGE_FILE_BYTES) {
      setError(`Image is too large (max ${MAX_IMAGE_FILE_BYTES / (1024 * 1024)} MB).`);
      return;
    }
    setError('');
    const reader = new FileReader();
    reader.onload = () => {
      setImageDataUrl(reader.result as string);
      setImageName(file.name);
      setResult(null);
    };
    reader.readAsDataURL(file);
  }, []);

  const parsedVector = useMemo(() => parseVector(vectorText), [vectorText]);

  const canRun = useMemo(() => {
    if (!selected || !selected.supported || running) return false;
    if (selected.inputKind === 'image') return !!imageDataUrl;
    if (selected.inputKind === 'vector') return parsedVector.length > 0;
    if (selected.inputKind === 'text') return textInput.trim().length > 0;
    return false;
  }, [selected, running, imageDataUrl, parsedVector, textInput]);

  const handleRun = useCallback(async () => {
    if (!selected) return;
    setRunning(true);
    setError('');
    setResult(null);
    try {
      const payload =
        selected.inputKind === 'image'
          ? { imageBase64: imageDataUrl as string }
          : selected.inputKind === 'vector'
            ? { values: parsedVector }
            : { text: textInput };
      const res = await window.fedLearnAPI.runInference(selected.projectId, payload);
      if (res.success && res.result) {
        setResult(res.result);
      } else {
        setError(res.error || 'Inference failed.');
      }
    } catch {
      setError('Inference failed.');
    } finally {
      setRunning(false);
    }
  }, [selected, imageDataUrl, parsedVector, textInput]);

  return (
    <div className="main-grid">
      {/* ── Input panel ── */}
      <section className="panel config-panel">
        <div className="panel-header">
          <h2 className="panel-title">Use a model</h2>
          <span className="panel-badge">Server inference</span>
        </div>

        {error && (
          <div className="docker-warning" role="alert">
            <span className="error-icon"><AlertTriangle strokeWidth={1.5} size={16} /></span>
            <span>{error}</span>
          </div>
        )}

        {loading ? (
          <p className="pg-muted">Loading your models…</p>
        ) : models.length === 0 ? (
          <div className="pg-empty">
            <FlaskConical size={28} strokeWidth={1.25} />
            <p>No trained models yet. Finish a training run and it’ll appear here.</p>
          </div>
        ) : (
          <div className="config-inputs">
            <div className="form-group">
              <label className="form-label" htmlFor="pg-model">Model</label>
              <select
                id="pg-model"
                className="form-input"
                value={selectedId}
                onChange={(e) => setSelectedId(e.target.value)}
              >
                {models.map((m) => (
                  <option key={m.projectId} value={m.projectId}>
                    {m.name} — {m.modelType}/{m.modelName}
                    {m.supported ? '' : ' (not runnable yet)'}
                  </option>
                ))}
              </select>
            </div>

            {selected && !selected.supported && (
              <p className="pg-muted">
                Interactive inference for {selected.modelType} models isn’t supported yet.
              </p>
            )}

            {selected?.inputKind === 'image' && (
              <div className="form-group">
                <label className="form-label">Image</label>
                <div
                  className="pg-dropzone"
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    handleFile(e.dataTransfer.files?.[0]);
                  }}
                >
                  {imageDataUrl ? (
                    <>
                      <img className="pg-preview" src={imageDataUrl} alt="input preview" />
                      <span className="pg-muted">{imageName}</span>
                    </>
                  ) : (
                    <>
                      <Upload size={24} strokeWidth={1.5} />
                      <span>Click or drop an image here</span>
                      <span className="pg-hint">Resized to 32×32 for the model</span>
                    </>
                  )}
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={(e) => handleFile(e.target.files?.[0])}
                />
              </div>
            )}

            {selected?.inputKind === 'vector' && (
              <div className="form-group">
                <label className="form-label" htmlFor="pg-vector">Feature vector</label>
                <textarea
                  id="pg-vector"
                  className="form-input"
                  rows={5}
                  value={vectorText}
                  onChange={(e) => setVectorText(e.target.value)}
                  placeholder="Paste numbers separated by commas, spaces, or newlines…"
                />
                <span className="pg-hint">{parsedVector.length} value(s) parsed</span>
              </div>
            )}

            {selected?.inputKind === 'text' && (
              <div className="form-group">
                <label className="form-label" htmlFor="pg-text">Text</label>
                <textarea
                  id="pg-text"
                  className="form-input"
                  rows={5}
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Enter text to classify…"
                />
              </div>
            )}

            <button className="btn btn-primary btn-full" onClick={handleRun} disabled={!canRun}>
              {running ? 'Running…' : (<><Sparkles size={16} strokeWidth={1.5} /> Run inference</>)}
            </button>
          </div>
        )}
      </section>

      {/* ── Result panel ── */}
      <section className="panel">
        <div className="panel-header">
          <h2 className="panel-title">Prediction</h2>
        </div>
        {!result ? (
          <div className="pg-empty">
            <FlaskConical size={28} strokeWidth={1.25} />
            <p>Run a model to see its prediction.</p>
          </div>
        ) : (
          <div className="pg-result">
            <div className="pg-top">
              <span className="pg-top-label">Top prediction</span>
              <span className="pg-top-value">{result.predictedLabel}</span>
              <span className="pg-top-conf">
                {(result.probabilities[result.predictedIndex] * 100).toFixed(1)}% confidence
              </span>
            </div>
            <div className="pg-bars">
              {result.classes.map((cls, i) => {
                const p = result.probabilities[i] ?? 0;
                const isTop = i === result.predictedIndex;
                return (
                  <div className="pg-bar-row" key={cls}>
                    <div className="pg-bar-head">
                      <span className={isTop ? 'pg-bar-label-top' : 'pg-bar-label'}>{cls}</span>
                      <span className="pg-bar-pct">{(p * 100).toFixed(1)}%</span>
                    </div>
                    <div className="pg-bar-track">
                      <div
                        className={isTop ? 'pg-bar-fill-top' : 'pg-bar-fill'}
                        style={{ width: `${Math.max(p * 100, 0.5)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </section>
    </div>
  );
};

export default ModelPlayground;
