import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { Play, Send, Square } from 'lucide-react-native';

import {
  listInferableModels,
  runInference,
  startGeneration,
  stopGeneration,
  type ChatTurn,
  type InferableModel,
  type InferenceResult,
} from '../lib/inferenceApi';
import { connectStomp, type StompHandle } from '../lib/stompClient';
import { readError } from '../lib/errors';
import { ErrorBanner } from '../components/ErrorBanner';
import { useThemeTokens } from '../theme/useThemeTokens';

type Mode = 'classify' | 'chat';
interface ChatMessage extends ChatTurn {
  streaming?: boolean;
}

const isChatModel = (m: InferableModel) =>
  m.inputKind == null || /llm|transformer|gpt|lora/i.test(m.modelType);
const isClassifyModel = (m: InferableModel) =>
  m.supported && (m.inputKind === 'vector' || m.inputKind === 'text');

// Server-side model playground — parity with the desktop "Use a Model" tab: pick a trained model and
// either classify an input (vector/text) or hold a streaming chat with a generative model. Tokens
// stream over STOMP (/topic/inference/{projectId}); the on-device native-inference path lives on the
// ModelTesting push. Pushed over the tabs from the Models hub — the native-stack header owns the
// title and back affordance, so no h2 self-title and no manual top inset here.
export function PlaygroundScreen() {
  const { colors } = useThemeTokens();
  const [mode, setMode] = useState<Mode>('classify');
  const [models, setModels] = useState<InferableModel[]>([]);
  const [selected, setSelected] = useState<InferableModel | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setModels(await listInferableModels());
    } catch (e: unknown) {
      setError(readError(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load]),
  );

  const pool = models.filter(mode === 'chat' ? isChatModel : isClassifyModel);

  // Keep a valid selection for the active mode.
  useEffect(() => {
    if (!selected || !pool.some((m) => m.projectId === selected.projectId)) {
      setSelected(pool[0] ?? null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, models]);

  return (
    <ScrollView className="flex-1 bg-canvas" keyboardShouldPersistTaps="handled">
      <View className="px-4 pt-4 pb-2">
        <Text className="text-caption font-sans text-fg-muted">Run a trained model on the server.</Text>
      </View>

      {/* Mode switch */}
      <View className="mx-4 flex-row rounded-pill bg-surface-2 p-1" accessibilityRole="tablist">
        {(['classify', 'chat'] as Mode[]).map((m) => (
          <Pressable
            key={m}
            accessibilityRole="tab"
            accessibilityLabel={m === 'classify' ? 'Classify' : 'Chat'}
            accessibilityState={{ selected: mode === m }}
            className={`flex-1 items-center py-2 rounded-pill active:opacity-80 ${mode === m ? 'bg-accent' : ''}`}
            onPress={() => setMode(m)}>
            <Text
              className={`text-label font-sans ${mode === m ? 'text-accent-fg' : 'text-fg-muted'}`}>
              {m === 'classify' ? 'Classify' : 'Chat'}
            </Text>
          </Pressable>
        ))}
      </View>

      {/* Model picker */}
      {loading && models.length === 0 ? (
        <View className="items-center mt-8">
          <ActivityIndicator color={colors.accent} />
        </View>
      ) : pool.length === 0 ? (
        <View className="mx-4 mt-4 p-4 rounded-card bg-surface-1 border border-hairline">
          <Text className="text-body font-sans text-fg-muted">
            No {mode === 'chat' ? 'generative' : 'classification'} models available to you yet.
          </Text>
        </View>
      ) : (
        <>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} className="mt-3">
            <View className="flex-row px-4">
              {pool.map((m) => {
                const on = selected?.projectId === m.projectId;
                return (
                  <Pressable
                    key={m.projectId}
                    accessibilityRole="button"
                    accessibilityLabel={m.name}
                    accessibilityState={{ selected: on }}
                    className={`mr-2 px-3 py-2 rounded-pill border active:opacity-80 ${
                      on ? 'bg-accent border-accent' : 'bg-surface-1 border-hairline'
                    }`}
                    onPress={() => setSelected(m)}>
                    <Text
                      className={`text-caption font-sans ${on ? 'text-accent-fg' : 'text-fg'}`}>
                      {m.name}
                    </Text>
                  </Pressable>
                );
              })}
            </View>
          </ScrollView>

          {selected ? (
            mode === 'classify' ? (
              <ClassifyPanel model={selected} />
            ) : (
              <ChatPanel model={selected} />
            )
          ) : null}
        </>
      )}

      {error ? <ErrorBanner message={error} className="mx-4 my-3" /> : null}
      <View className="h-8" />
    </ScrollView>
  );
}

// ─── Classify (vector / text) ────────────────────────────────────────────────
function ClassifyPanel({ model }: { model: InferableModel }) {
  const { colors } = useThemeTokens();
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const isVector = model.inputKind === 'vector';

  useEffect(() => {
    setInput('');
    setResult(null);
    setError(null);
  }, [model.projectId]);

  const onRun = useCallback(async () => {
    setError(null);
    setBusy(true);
    try {
      const payload = isVector
        ? { values: parseVector(input) }
        : { text: input };
      if (isVector && (payload.values as number[]).length === 0) {
        throw new Error('Enter some numbers (comma, space, or newline separated).');
      }
      setResult(await runInference(model.projectId, payload));
    } catch (e: unknown) {
      setError(readError(e));
    } finally {
      setBusy(false);
    }
  }, [input, isVector, model.projectId]);

  return (
    <View className="mx-4 mt-3">
      <View className="p-4 rounded-card bg-surface-1 border border-hairline">
        <Text className="text-caption font-sans text-fg-muted mb-2">
          {isVector
            ? 'Feature vector — comma, space, or newline separated numbers'
            : 'Text input'}
        </Text>
        <TextInput
          className="rounded-md bg-surface-2 border border-hairline px-3 py-2 text-body font-mono text-fg"
          accessibilityLabel={isVector ? 'Feature vector' : 'Text input'}
          value={input}
          onChangeText={setInput}
          multiline
          placeholder={isVector ? '0.12, -0.4, 1.03, …' : 'Type an input to classify…'}
          placeholderTextColor={colors['fg-subtle']}
          style={{ minHeight: 72, textAlignVertical: 'top' }}
        />
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Run inference"
          accessibilityState={{ disabled: busy }}
          className={`mt-3 flex-row items-center justify-center bg-accent rounded-md py-3 active:opacity-80 ${
            busy ? 'opacity-50' : ''
          }`}
          disabled={busy}
          onPress={onRun}>
          {busy ? (
            <ActivityIndicator color={colors['accent-fg']} />
          ) : (
            <>
              <Play color={colors['accent-fg']} size={18} strokeWidth={1.5} />
              <Text className="text-accent-fg text-label font-sans ml-2">Run inference</Text>
            </>
          )}
        </Pressable>
      </View>

      {result ? (
        <View className="mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
          <Text className="text-fg text-body font-sans mb-2">
            Prediction: {result.predictedLabel}
          </Text>
          {result.probabilities.map((p, i) => (
            <View key={i} className="flex-row items-center mb-1">
              <Text
                className="w-24 text-caption text-fg-muted font-sans"
                numberOfLines={1}>
                {result.classes[i] ?? i}
              </Text>
              <View className="flex-1 h-3 rounded-pill bg-surface-2 overflow-hidden mr-2">
                <View
                  className={i === result.predictedIndex ? 'h-3 bg-accent' : 'h-3 bg-series-1'}
                  style={{ width: `${clamp01(p) * 100}%` }}
                />
              </View>
              <Text
                className="w-12 text-right text-caption text-fg font-mono"
                style={{ fontVariant: ['tabular-nums'] }}>
                {(p * 100).toFixed(1)}%
              </Text>
            </View>
          ))}
        </View>
      ) : null}

      {error ? <ErrorBanner message={error} className="mt-3" /> : null}
    </View>
  );
}

// ─── Chat (streaming generation) ─────────────────────────────────────────────
function ChatPanel({ model }: { model: InferableModel }) {
  const { colors } = useThemeTokens();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [prompt, setPrompt] = useState('');
  const [maxTokens, setMaxTokens] = useState('256');
  const [temperature, setTemperature] = useState('0.7');
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stompRef = useRef<StompHandle | null>(null);
  const unsubRef = useRef<(() => void) | null>(null);
  const streamRef = useRef<string>(''); // accumulated tokens for the in-flight assistant turn

  // (Re)connect STOMP for this model's inference topic; tear down on model change / unmount.
  useEffect(() => {
    let alive = true;
    setMessages([]);
    setError(null);
    (async () => {
      try {
        const handle = await connectStomp((msg) => alive && setError(msg));
        if (!alive) {
          handle.deactivate();
          return;
        }
        stompRef.current = handle;
        unsubRef.current = handle.subscribe(`/topic/inference/${model.projectId}`, (body) => {
          const token = parseToken(body);
          if (!token) return;
          streamRef.current += token;
          setMessages((prev) => {
            const next = prev.slice();
            const last = next[next.length - 1];
            if (last && last.role === 'assistant' && last.streaming) {
              next[next.length - 1] = { ...last, content: streamRef.current };
            }
            return next;
          });
        });
      } catch (e: unknown) {
        if (alive) setError(readError(e));
      }
    })();
    return () => {
      alive = false;
      unsubRef.current?.();
      unsubRef.current = null;
      stompRef.current?.deactivate();
      stompRef.current = null;
    };
  }, [model.projectId]);

  const onSend = useCallback(async () => {
    const text = prompt.trim();
    if (!text || generating) return;
    setError(null);
    setPrompt('');
    streamRef.current = '';

    const history: ChatTurn[] = messages
      .filter((m) => !m.streaming)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [
      ...prev,
      { role: 'user', content: text },
      { role: 'assistant', content: '', streaming: true },
    ]);
    setGenerating(true);
    try {
      const res = await startGeneration(model.projectId, {
        prompt: text,
        maxNewTokens: clampInt(maxTokens, 1, 2048, 256),
        temperature: clampFloat(temperature, 0, 2, 0.7),
        history,
      });
      // Finalize with the authoritative accumulated text (fall back to the streamed buffer).
      const finalText = res.generatedText || streamRef.current;
      setMessages((prev) => {
        const next = prev.slice();
        const last = next[next.length - 1];
        if (last && last.role === 'assistant') {
          next[next.length - 1] = { role: 'assistant', content: finalText };
        }
        return next;
      });
    } catch (e: unknown) {
      setError(readError(e));
      setMessages((prev) => prev.filter((m) => !m.streaming)); // drop the empty placeholder
    } finally {
      setGenerating(false);
      streamRef.current = '';
    }
  }, [prompt, generating, messages, maxTokens, temperature, model.projectId]);

  const onStop = useCallback(async () => {
    try {
      await stopGeneration(model.projectId);
    } catch {
      /* best-effort */
    }
  }, [model.projectId]);

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      className="mx-4 mt-3">
      {/* Generation controls */}
      <View className="flex-row mb-2">
        <View className="flex-1 mr-2">
          <Text className="text-caption font-sans text-fg-muted mb-1">Max tokens</Text>
          <TextInput
            className="rounded-md bg-surface-2 border border-hairline px-3 py-2 text-body font-mono text-fg"
            accessibilityLabel="Max tokens"
            value={maxTokens}
            onChangeText={setMaxTokens}
            keyboardType="number-pad"
          />
        </View>
        <View className="flex-1 ml-2">
          <Text className="text-caption font-sans text-fg-muted mb-1">Temperature</Text>
          <TextInput
            className="rounded-md bg-surface-2 border border-hairline px-3 py-2 text-body font-mono text-fg"
            accessibilityLabel="Temperature"
            value={temperature}
            onChangeText={setTemperature}
            keyboardType="decimal-pad"
          />
        </View>
      </View>

      {/* Thread */}
      <View className="p-4 rounded-card bg-surface-1 border border-hairline" style={{ minHeight: 160 }}>
        {messages.length === 0 ? (
          <Text className="text-caption font-sans text-fg-subtle">
            Ask {model.name} something to start the conversation.
          </Text>
        ) : (
          messages.map((m, i) => (
            <View
              key={i}
              className={`mb-2 px-3 py-2 rounded-lg ${
                m.role === 'user' ? 'bg-surface-2 self-end' : 'bg-code-well self-start'
              }`}
              style={{ maxWidth: '88%' }}>
              <Text className="text-caption font-sans text-fg-subtle mb-0.5">
                {m.role === 'user' ? 'You' : model.name}
              </Text>
              <Text className="text-body font-sans text-fg">
                {m.content || (m.streaming ? 'Generating…' : '')}
              </Text>
            </View>
          ))
        )}
      </View>

      {/* Composer */}
      <View className="flex-row items-end mt-2">
        <TextInput
          className="flex-1 rounded-md bg-surface-2 border border-hairline px-3 py-2 text-body font-sans text-fg mr-2"
          accessibilityLabel="Message"
          value={prompt}
          onChangeText={setPrompt}
          multiline
          placeholder="Message…"
          placeholderTextColor={colors['fg-subtle']}
          style={{ maxHeight: 120 }}
        />
        {generating ? (
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Stop generating"
            className="items-center justify-center bg-danger rounded-md px-4 py-3 active:opacity-80"
            onPress={onStop}>
            <Square color={colors['accent-fg']} size={18} strokeWidth={1.5} />
          </Pressable>
        ) : (
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Send message"
            accessibilityState={{ disabled: !prompt.trim() }}
            className={`items-center justify-center bg-accent rounded-md px-4 py-3 active:opacity-80 ${
              !prompt.trim() ? 'opacity-50' : ''
            }`}
            disabled={!prompt.trim()}
            onPress={onSend}>
            <Send color={colors['accent-fg']} size={18} strokeWidth={1.5} />
          </Pressable>
        )}
      </View>

      {error ? <ErrorBanner message={error} className="mt-3" /> : null}
    </KeyboardAvoidingView>
  );
}

// ─── helpers ─────────────────────────────────────────────────────────────────
function parseVector(s: string): number[] {
  return s
    .split(/[\s,]+/)
    .map((t) => t.trim())
    .filter(Boolean)
    .map(Number)
    .filter((n) => Number.isFinite(n));
}
function parseToken(body: string): string {
  try {
    const j = JSON.parse(body);
    return typeof j?.token === 'string' ? j.token : '';
  } catch {
    return body; // tolerate a raw-string frame
  }
}
const clamp01 = (n: number) => Math.max(0, Math.min(1, n));
function clampInt(s: string, lo: number, hi: number, dflt: number): number {
  const n = parseInt(s, 10);
  return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : dflt;
}
function clampFloat(s: string, lo: number, hi: number, dflt: number): number {
  const n = parseFloat(s);
  return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : dflt;
}
