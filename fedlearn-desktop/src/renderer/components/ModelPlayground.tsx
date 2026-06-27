// =============================================================================
// FedLearn Desktop — Model Playground ("Use a model")
// =============================================================================
// Pick a trained model and run inference on it. Image models take an uploaded
// image; tabular models take a numeric vector; generation models use a chat
// thread. Inference runs server-side via the Main-process InferenceService
// (reached over IPC).
// =============================================================================

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FlaskConical, Upload, Sparkles, AlertTriangle } from 'lucide-react';
import type { InferableModel, InferenceResult, GenerationResult } from '../inference.types';

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
  const [prompt, setPrompt] = useState('');
  const [maxNewTokens, setMaxNewTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [streamingText, setStreamingText] = useState('');
  const [stopped, setStopped] = useState(false);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState('');
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

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
    setPrompt('');
    setStreamingText('');
    setStopped(false);
    setMessages([]);
  }, [selectedId]);

  useEffect(() => {
    window.fedLearnAPI.onInferenceToken((token: string) => setStreamingText((p) => p + token));
    return () => window.fedLearnAPI.removeInferenceTokenListener();
  }, []);

  // Auto-scroll chat to bottom when messages or streaming text change
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingText]);

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
    if (selected.inputKind === 'generation') return prompt.trim().length > 0;
    return false;
  }, [selected, running, imageDataUrl, parsedVector, textInput, prompt]);

  const handleSend = useCallback(async () => {
    if (!selected || !prompt.trim()) return;
    const userMsg = prompt.trim();
    const history = messages;
    setMessages((m) => [...m, { role: 'user', content: userMsg }]);
    setPrompt('');
    setStopped(false);
    setRunning(true);
    setError('');
    setStreamingText('');
    try {
      const res = await window.fedLearnAPI.runGeneration(selected.projectId, {
        prompt: userMsg,
        history,
        maxNewTokens,
        temperature,
      });
      setStreamingText((finalText) => {
        setMessages((m) => [...m, { role: 'assistant', content: finalText }]);
        return '';
      });
      if (res.success && res.result && (res.result as GenerationResult).finishReason === 'stopped') {
        setStopped(true);
      } else if (!res.success) {
        setError(res.error || 'Generation failed.');
      }
    } catch {
      setError('Generation failed.');
    } finally {
      setRunning(false);
    }
  }, [selected, prompt, maxNewTokens, temperature, messages]);

  const handleStop = useCallback(async () => {
    if (!selected) return;
    setStopped(true);
    try { await window.fedLearnAPI.stopGeneration(selected.projectId); } catch { /* best-effort */ }
  }, [selected]);

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
            <p>No trained models yet. Finish a training run and it'll appear here.</p>
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
                Interactive inference for {selected.modelType} models isn't supported yet.
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

            {selected?.inputKind === 'generation' && (
              <div className="form-group">
                <textarea
                  id="pg-prompt"
                  className="form-input"
                  rows={4}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Message the model…"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey && canRun && !running) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                />
                <label className="form-label">Max new tokens: {maxNewTokens}</label>
                <input type="range" min={1} max={2048} step={1} value={maxNewTokens}
                  onChange={(e) => setMaxNewTokens(Number(e.target.value))} />
                <label className="form-label">Temperature: {temperature.toFixed(1)}</label>
                <input type="range" min={0} max={2} step={0.1} value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))} />
                <div className="pg-chat-actions">
                  {running ? (
                    <button className="btn btn-primary btn-full" onClick={handleStop} disabled={stopped}>Stop</button>
                  ) : (
                    <>
                      <button className="btn btn-primary btn-full" onClick={handleSend} disabled={!canRun}>
                        <Sparkles size={16} strokeWidth={1.5} /> Send
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={() => setMessages([])}
                        disabled={running || messages.length === 0}
                      >
                        Clear
                      </button>
                    </>
                  )}
                </div>
              </div>
            )}

            {selected?.inputKind !== 'generation' && (
              <button className="btn btn-primary btn-full"
                onClick={handleRun}
                disabled={!canRun}>
                {running ? 'Running…' : (<><Sparkles size={16} strokeWidth={1.5} /> Run inference</>)}
              </button>
            )}
          </div>
        )}
      </section>

      {/* ── Result panel ── */}
      <section className="panel">
        <div className="panel-header">
          <h2 className="panel-title">
            {selected?.inputKind === 'generation' ? 'Chat' : 'Prediction'}
          </h2>
        </div>

        {/* Classification / vector / text result */}
        {selected?.inputKind !== 'generation' && (!result ? (
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
        ))}

        {/* Generation chat thread */}
        {selected?.inputKind === 'generation' && (
          messages.length === 0 && !streamingText ? (
            <div className="pg-empty">
              <FlaskConical size={28} strokeWidth={1.25} />
              <p>Send a message to start the conversation.</p>
            </div>
          ) : (
            <div className="pg-chat">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={msg.role === 'user' ? 'pg-bubble-user' : 'pg-bubble-assistant'}
                >
                  {msg.content}
                </div>
              ))}
              {running && streamingText && (
                <div className="pg-bubble-assistant">{streamingText}</div>
              )}
              <div ref={chatEndRef} />
            </div>
          )
        )}
      </section>
    </div>
  );
};

export default ModelPlayground;
