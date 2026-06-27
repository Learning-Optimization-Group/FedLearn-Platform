// =============================================================================
// FedLearn Frontend — Playground / "Use a model" (Ember design system)
// =============================================================================
// Pick one of your trained federated models and run inference on an input.
// Image models (CNN) take an uploaded image; tabular models (MLP) take a
// numeric feature vector. Inference runs server-side (real PyTorch).

import { useEffect, useMemo, useRef, useState } from 'react';
import { FlaskConical, AlertCircle, Upload, Sparkles, ImageIcon, Hash, Loader2 } from 'lucide-react';
import { Client as StompClient } from '@stomp/stompjs';
import * as api from '../../services/apiServices';
import type { InferableModel, InferenceResult } from '../../services/apiServices';
import { Card, Button, Select, Skeleton } from '../ui';
import { BrandMark } from '../brand';
import { PageHeader } from './PageHeader';

const SERVER_ROOT_URL = import.meta.env.VITE_SERVER_ROOT_URL || `http://${window.location.hostname}:8081`;
const WS_BROKER_URL = `${SERVER_ROOT_URL.replace(/^http/, 'ws')}/ws-logs`;

// Reject oversized files before FileReader pulls them fully into browser memory.
// The backend also bounds the encoded body, but this gives immediate feedback and
// avoids a large wasted upload from a multi-hundred-MB pick.
const MAX_IMAGE_FILE_BYTES = 10 * 1024 * 1024;

/** Parse a free-form textarea (commas / spaces / newlines) into finite numbers. */
function parseVector(raw: string): number[] {
    return raw
        .split(/[\s,]+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0)
        .map(Number)
        .filter((n) => Number.isFinite(n));
}

export function PlaygroundView() {
    const [models, setModels] = useState<InferableModel[]>([]);
    const [loadingModels, setLoadingModels] = useState(true);
    const [selectedId, setSelectedId] = useState<string>('');

    const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
    const [imageName, setImageName] = useState<string>('');
    const [vectorText, setVectorText] = useState<string>('');
    const [textInput, setTextInput] = useState<string>('');

    const [prompt, setPrompt] = useState('');
    const [maxNewTokens, setMaxNewTokens] = useState(256);
    const [temperature, setTemperature] = useState(0.7);
    const [streamingText, setStreamingText] = useState('');
    const [stopped, setStopped] = useState(false);
    const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([]);
    const chatEndRef = useRef<HTMLDivElement>(null);

    const [running, setRunning] = useState(false);
    const [result, setResult] = useState<InferenceResult | null>(null);
    const [error, setError] = useState('');
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        (async () => {
            try {
                const res = await api.fetchInferableModels();
                const list = Array.isArray(res.data) ? res.data : [];
                setModels(list);
                if (list.length > 0) setSelectedId(list[0].projectId);
            } catch {
                setError('Failed to load your models.');
            } finally {
                setLoadingModels(false);
            }
        })();
    }, []);

    const selected = useMemo(
        () => models.find((m) => m.projectId === selectedId) ?? null,
        [models, selectedId],
    );

    // Clear input + result whenever the chosen model changes.
    useEffect(() => {
        setResult(null);
        setError('');
        setImageDataUrl(null);
        setImageName('');
        setVectorText('');
        setTextInput('');
        setPrompt('');
        setStreamingText('');
        setMessages([]);
        setStopped(false);
    }, [selectedId]);

    // Scroll chat to bottom whenever messages or streamingText change.
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, streamingText]);

    const handleFile = (file: File | undefined) => {
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
    };

    const parsedVector = useMemo(() => parseVector(vectorText), [vectorText]);

    const canRun = useMemo(() => {
        if (!selected || !selected.supported || running) return false;
        if (selected.inputKind === 'image') return !!imageDataUrl;
        if (selected.inputKind === 'vector') return parsedVector.length > 0;
        if (selected.inputKind === 'text') return textInput.trim().length > 0;
        if (selected.inputKind === 'generation') return prompt.trim().length > 0;
        return false;
    }, [selected, running, imageDataUrl, parsedVector, textInput, prompt]);

    const handleRun = async () => {
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
            const res = await api.runInference(selected.projectId, payload);
            setResult(res.data);
        } catch (e: unknown) {
            const msg =
                (e as { response?: { data?: { message?: string } } })?.response?.data?.message ||
                'Inference failed. Please try again.';
            setError(msg);
        } finally {
            setRunning(false);
        }
    };

    const handleSend = async () => {
        if (!selected || !prompt.trim()) return;
        const userMsg = prompt.trim();
        const history = messages;
        setMessages((m) => [...m, { role: 'user', content: userMsg }]);
        setPrompt('');
        setRunning(true);
        setError('');
        setStreamingText('');
        setStopped(false);
        const client = new StompClient({ brokerURL: WS_BROKER_URL, reconnectDelay: 5000 });
        const cleanup = () => { if (client.active) client.deactivate(); };
        await new Promise<void>((resolve) => {
            const t = setTimeout(resolve, 8000);
            client.onConnect = () => {
                clearTimeout(t);
                client.subscribe(`/topic/inference/${selected.projectId}`, (msg) => {
                    try {
                        const { token } = JSON.parse(msg.body);
                        if (typeof token === 'string') setStreamingText((p) => p + token);
                    } catch { /* ignore non-token frames */ }
                });
                resolve(); // only POST once the subscription is live (avoids the first-token race)
            };
            client.onStompError = () => resolve();
            client.onWebSocketError = () => resolve();
            client.activate();
        });
        try {
            const res = await api.runGeneration(selected.projectId, {
                prompt: userMsg, history, maxNewTokens, temperature,
            });
            setStreamingText((finalText) => {
                setMessages((m) => [...m, { role: 'assistant', content: finalText }]);
                return '';
            });
            if (res.data.finishReason === 'stopped') setStopped(true);
        } catch (e: unknown) {
            const msg = (e as { response?: { data?: { message?: string } } })?.response?.data?.message
                || 'Generation failed. Please try again.';
            setError(msg);
        } finally {
            cleanup();
            setRunning(false);
        }
    };

    const handleStop = async () => {
        if (!selected) return;
        setStopped(true);                         // instant UI feedback
        try { await api.stopGeneration(selected.projectId); } catch { /* best-effort */ }
    };

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
            <PageHeader title="Use a model" subtitle="Run one of your trained models on a new input." />

            <div className="flex-1 overflow-y-auto px-6 md:px-10 py-8 bg-canvas">
                {error && (
                    <div className="mb-6 flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium">
                        <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                        {error}
                    </div>
                )}

                {loadingModels ? (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <Card padding="lg" className="flex flex-col gap-4">
                            <Skeleton className="h-5 w-40" />
                            <Skeleton className="h-40 w-full" />
                        </Card>
                        <Card padding="lg" className="flex flex-col gap-4">
                            <Skeleton className="h-5 w-32" />
                            <Skeleton className="h-40 w-full" />
                        </Card>
                    </div>
                ) : models.length === 0 ? (
                    <div className="flex flex-col items-center justify-center text-center gap-5 mt-16 md:mt-24">
                        <div className="grid h-20 w-20 place-items-center rounded-card border border-hairline bg-surface-1">
                            <BrandMark size={48} />
                        </div>
                        <div className="max-w-sm">
                            <p className="text-h4 font-display text-fg">No trained models yet</p>
                            <p className="text-body text-fg-muted mt-1.5">
                                Once a project finishes training, its model shows up here ready to use.
                            </p>
                        </div>
                    </div>
                ) : selected?.inputKind === 'generation' ? (
                        /* ── Chat thread (generation models) ── */
                        <Card padding="lg" className="flex flex-col gap-4">
                            <div className="flex items-center gap-3">
                                <span className="icon-tile flex-shrink-0">
                                    <Sparkles strokeWidth={1.5} className="w-5 h-5" />
                                </span>
                                <div className="flex-1 min-w-0">
                                    <h3 className="text-h4 font-display text-fg">Chat</h3>
                                    <p className="text-label text-fg-muted truncate">
                                        {selected.name} — {selected.modelType}/{selected.modelName}
                                    </p>
                                </div>
                                <label className="flex flex-col gap-0.5 text-caption text-fg-muted shrink-0">
                                    <span className="text-caption uppercase tracking-wide font-semibold text-fg-muted">Model</span>
                                    <Select value={selectedId} onChange={(e) => setSelectedId(e.target.value)}>
                                        {models.map((m) => (
                                            <option key={m.projectId} value={m.projectId}>
                                                {m.name} — {m.modelType}/{m.modelName}
                                                {m.supported ? '' : ' (not runnable yet)'}
                                            </option>
                                        ))}
                                    </Select>
                                </label>
                            </div>

                            {/* Scrollable bubble list */}
                            <div className="flex flex-col gap-3 overflow-y-auto max-h-[420px] min-h-[200px] rounded-card border border-hairline bg-surface-1 px-4 py-4">
                                {messages.length === 0 && !streamingText ? (
                                    <div className="flex flex-1 flex-col items-center justify-center text-center gap-2 py-10 text-fg-subtle">
                                        <Sparkles className="w-8 h-8" strokeWidth={1.25} />
                                        <p className="text-body text-fg-muted">Start the conversation…</p>
                                    </div>
                                ) : (
                                    messages.map((msg, i) => (
                                        <div
                                            key={i}
                                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                        >
                                            <div
                                                className={`max-w-[80%] rounded-card px-3 py-2 whitespace-pre-wrap font-mono text-label ${
                                                    msg.role === 'user'
                                                        ? 'bg-accent/10 text-fg border border-accent/25'
                                                        : 'bg-surface-2 text-fg border border-hairline'
                                                }`}
                                            >
                                                {msg.content}
                                            </div>
                                        </div>
                                    ))
                                )}
                                {/* Live in-flight assistant bubble */}
                                {running && streamingText && (
                                    <div className="flex justify-start">
                                        <div className="max-w-[80%] rounded-card px-3 py-2 whitespace-pre-wrap font-mono text-label bg-surface-2 text-fg border border-hairline">
                                            {streamingText}<span className="animate-pulse">▌</span>
                                        </div>
                                    </div>
                                )}
                                {running && !streamingText && (
                                    <div className="flex justify-start">
                                        <div className="rounded-card px-3 py-2 bg-surface-2 border border-hairline text-fg-muted text-label flex items-center gap-2">
                                            <Loader2 className="w-3.5 h-3.5 animate-spin" strokeWidth={2} /> Generating…
                                        </div>
                                    </div>
                                )}
                                <div ref={chatEndRef} />
                            </div>

                            {/* Prompt textarea */}
                            <textarea
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && canRun) {
                                        e.preventDefault();
                                        handleSend();
                                    }
                                }}
                                rows={3}
                                placeholder="Message the model…"
                                className="w-full rounded-md border border-hairline bg-surface-1 px-3 py-2 text-label text-fg placeholder:text-fg-subtle focus:border-accent/50 focus:outline-none resize-y"
                            />

                            {/* Compact sliders */}
                            <div className="flex flex-wrap gap-6">
                                <label className="flex flex-col gap-1 text-label text-fg-muted flex-1 min-w-[160px]">
                                    Max new tokens: <span className="font-mono tabular-nums text-fg">{maxNewTokens}</span>
                                    <input type="range" min={1} max={2048} step={1} value={maxNewTokens}
                                        onChange={(e) => setMaxNewTokens(Number(e.target.value))} />
                                </label>
                                <label className="flex flex-col gap-1 text-label text-fg-muted flex-1 min-w-[160px]">
                                    Temperature: <span className="font-mono tabular-nums text-fg">{temperature.toFixed(1)}</span>
                                    <input type="range" min={0} max={2} step={0.1} value={temperature}
                                        onChange={(e) => setTemperature(Number(e.target.value))} />
                                </label>
                            </div>

                            {/* Button row */}
                            <div className="flex items-center gap-2">
                                {running ? (
                                    <Button
                                        variant="primary"
                                        onClick={handleStop}
                                        disabled={stopped}
                                        className="inline-flex items-center justify-center gap-2"
                                    >
                                        Stop
                                    </Button>
                                ) : (
                                    <Button
                                        variant="primary"
                                        onClick={handleSend}
                                        disabled={!canRun}
                                        className="inline-flex items-center justify-center gap-2"
                                    >
                                        <Sparkles className="w-4 h-4" strokeWidth={1.5} /> Send
                                    </Button>
                                )}
                                <Button
                                    variant="secondary"
                                    onClick={() => setMessages([])}
                                    disabled={running || messages.length === 0}
                                    className="inline-flex items-center justify-center gap-2"
                                >
                                    Clear
                                </Button>
                            </div>
                        </Card>
                    ) : (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* ── Input panel ── */}
                        <Card padding="lg" className="flex flex-col gap-5">
                            <div className="flex items-center gap-3">
                                <span className="icon-tile flex-shrink-0">
                                    <FlaskConical strokeWidth={1.5} className="w-5 h-5" />
                                </span>
                                <div>
                                    <h3 className="text-h4 font-display text-fg">Input</h3>
                                    <p className="text-label text-fg-muted">Choose a model and give it something to predict.</p>
                                </div>
                            </div>

                            <label className="flex flex-col gap-1.5">
                                <span className="text-caption uppercase tracking-wide font-semibold text-fg-muted">Model</span>
                                <Select value={selectedId} onChange={(e) => setSelectedId(e.target.value)}>
                                    {models.map((m) => (
                                        <option key={m.projectId} value={m.projectId}>
                                            {m.name} — {m.modelType}/{m.modelName}
                                            {m.supported ? '' : ' (not runnable yet)'}
                                        </option>
                                    ))}
                                </Select>
                            </label>

                            {selected && !selected.supported && (
                                <div className="flex items-center gap-2 px-3 py-2.5 rounded-md border border-hairline bg-surface-1 text-fg-muted text-label">
                                    <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
                                    Interactive inference for {selected.modelType} models isn't supported yet.
                                </div>
                            )}

                            {/* Image input */}
                            {selected?.inputKind === 'image' && (
                                <div className="flex flex-col gap-3">
                                    <span className="text-caption uppercase tracking-wide font-semibold text-fg-muted flex items-center gap-1.5">
                                        <ImageIcon className="w-3.5 h-3.5" strokeWidth={1.5} /> Image
                                    </span>
                                    <div
                                        onClick={() => fileInputRef.current?.click()}
                                        onDragOver={(e) => e.preventDefault()}
                                        onDrop={(e) => {
                                            e.preventDefault();
                                            handleFile(e.dataTransfer.files?.[0]);
                                        }}
                                        className="cursor-pointer rounded-card border border-dashed border-hairline bg-surface-1 hover:border-accent/40 hover:bg-surface-2 transition-colors p-6 flex flex-col items-center justify-center gap-3 text-center min-h-[180px]"
                                    >
                                        {imageDataUrl ? (
                                            <>
                                                <img
                                                    src={imageDataUrl}
                                                    alt="input preview"
                                                    className="max-h-32 rounded-md border border-hairline object-contain"
                                                />
                                                <span className="text-label text-fg-muted truncate max-w-full">{imageName}</span>
                                            </>
                                        ) : (
                                            <>
                                                <Upload className="w-7 h-7 text-fg-subtle" strokeWidth={1.5} />
                                                <span className="text-body text-fg-muted">
                                                    Click or drop an image here
                                                </span>
                                                <span className="text-caption text-fg-subtle">
                                                    Resized to 32×32 for the model
                                                </span>
                                            </>
                                        )}
                                    </div>
                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept="image/*"
                                        className="hidden"
                                        onChange={(e) => handleFile(e.target.files?.[0])}
                                    />
                                </div>
                            )}

                            {/* Vector input */}
                            {selected?.inputKind === 'vector' && (
                                <div className="flex flex-col gap-2">
                                    <span className="text-caption uppercase tracking-wide font-semibold text-fg-muted flex items-center gap-1.5">
                                        <Hash className="w-3.5 h-3.5" strokeWidth={1.5} /> Feature vector
                                    </span>
                                    <textarea
                                        value={vectorText}
                                        onChange={(e) => setVectorText(e.target.value)}
                                        rows={6}
                                        placeholder="Paste numbers separated by commas, spaces, or newlines…"
                                        className="w-full rounded-md border border-hairline bg-surface-1 px-3 py-2 text-label font-mono text-fg placeholder:text-fg-subtle focus:border-accent/50 focus:outline-none resize-y"
                                    />
                                    <span className="text-caption text-fg-subtle">
                                        {parsedVector.length} value{parsedVector.length === 1 ? '' : 's'} parsed
                                    </span>
                                </div>
                            )}

                            {/* Text input */}
                            {selected?.inputKind === 'text' && (
                                <div className="flex flex-col gap-2">
                                    <span className="text-caption uppercase tracking-wide font-semibold text-fg-muted flex items-center gap-1.5">
                                        <Sparkles className="w-3.5 h-3.5" strokeWidth={1.5} /> Text
                                    </span>
                                    <textarea
                                        value={textInput}
                                        onChange={(e) => setTextInput(e.target.value)}
                                        rows={5}
                                        placeholder="Enter text to classify…"
                                        className="w-full rounded-md border border-hairline bg-surface-1 px-3 py-2 text-label text-fg placeholder:text-fg-subtle focus:border-accent/50 focus:outline-none resize-y"
                                    />
                                </div>
                            )}

                            <Button
                                variant="primary"
                                onClick={handleRun}
                                disabled={!canRun}
                                className="mt-1 inline-flex items-center justify-center gap-2"
                            >
                                {running ? (
                                    <span className="flex items-center gap-2">
                                        <Loader2 className="w-4 h-4 animate-spin" strokeWidth={2} /> Running…
                                    </span>
                                ) : (
                                    <span className="flex items-center gap-2">
                                        <Sparkles className="w-4 h-4" strokeWidth={1.5} /> Run inference
                                    </span>
                                )}
                            </Button>
                        </Card>

                        {/* ── Result panel ── */}
                        <Card padding="lg" className="flex flex-col gap-5">
                            <div className="flex items-center gap-3">
                                <span className="icon-tile flex-shrink-0">
                                    <Sparkles strokeWidth={1.5} className="w-5 h-5" />
                                </span>
                                <div>
                                    <h3 className="text-h4 font-display text-fg">Prediction</h3>
                                    <p className="text-label text-fg-muted">What the model thinks.</p>
                                </div>
                            </div>

                            {!result ? (
                                <div className="flex flex-1 flex-col items-center justify-center text-center gap-2 py-10 text-fg-subtle">
                                    <FlaskConical className="w-8 h-8" strokeWidth={1.25} />
                                    <p className="text-body text-fg-muted">Run a model to see its prediction.</p>
                                </div>
                            ) : (
                                <div className="flex flex-col gap-4">
                                    <div className="rounded-card border border-accent/25 bg-accent/10 px-4 py-4">
                                        <p className="text-caption uppercase tracking-wide font-semibold text-fg-muted">
                                            Top prediction
                                        </p>
                                        <p className="text-h2 font-display text-fg capitalize mt-1">
                                            {result.predictedLabel}
                                        </p>
                                        <p className="text-label text-fg-muted mt-0.5 font-mono tabular-nums">
                                            {(result.probabilities[result.predictedIndex] * 100).toFixed(1)}% confidence
                                        </p>
                                    </div>

                                    <div className="flex flex-col gap-2">
                                        {result.classes.map((cls, i) => {
                                            const p = result.probabilities[i] ?? 0;
                                            const isTop = i === result.predictedIndex;
                                            return (
                                                <div key={cls} className="flex flex-col gap-1">
                                                    <div className="flex items-center justify-between text-label">
                                                        <span className={isTop ? 'text-fg font-medium capitalize' : 'text-fg-muted capitalize'}>
                                                            {cls}
                                                        </span>
                                                        <span className="font-mono tabular-nums text-fg-muted">
                                                            {(p * 100).toFixed(1)}%
                                                        </span>
                                                    </div>
                                                    <div className="h-2 rounded-pill bg-surface-2 overflow-hidden">
                                                        <div
                                                            className={isTop ? 'h-full rounded-pill bg-accent' : 'h-full rounded-pill bg-fg-subtle/40'}
                                                            style={{ width: `${Math.max(p * 100, 0.5)}%` }}
                                                        />
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}
                        </Card>
                    </div>
                )}
            </div>
        </div>
    );
}
