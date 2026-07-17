import React, { useCallback, useMemo, useState } from 'react';
import { Pressable, ScrollView, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useFocusEffect } from '@react-navigation/native';
import { Eraser, Play } from 'lucide-react-native';

import nativeCore, { type InferResult } from '../lib/nativeCore';
import { listModels, type SavedModel } from '../lib/modelStore';
import { ErrorBanner } from '../components/ErrorBanner';
import { useThemeTokens } from '../theme/useThemeTokens';

const GRID = 8; // 8x8 input grid; the RN layer flattens it for the model (input shape is model-specific)
const N = GRID * GRID;

export function ModelTestingScreen() {
  const { colors } = useThemeTokens();
  const insets = useSafeAreaInsets(); // top safe area on the root; bottom belongs to the tab bar
  const [active, setActive] = useState<SavedModel | null>(null);
  const [cells, setCells] = useState<boolean[]>(() => new Array(N).fill(false));
  const [result, setResult] = useState<InferResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useFocusEffect(
    useCallback(() => {
      (async () => {
        try {
          const models = await listModels();
          const first = models[0] ?? null;
          setActive(first);
          if (first) await nativeCore.loadModel(first.path, first.sha256); // integrity-checked
        } catch (e) {
          setError(String(e));
        }
      })();
    }, []),
  );

  const toggle = useCallback((i: number) => {
    setCells((prev) => {
      const next = prev.slice();
      next[i] = !next[i];
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setCells(new Array(N).fill(false));
    setResult(null);
  }, []);

  const onInfer = useCallback(async () => {
    setError(null);
    try {
      const inputJson = JSON.stringify(cells.map((c) => (c ? 1 : 0)));
      setResult(await nativeCore.infer(inputJson)); // REAL softmax from the model (C5 §3)
    } catch (e) {
      setError(String(e));
    }
  }, [cells]);

  const rows = useMemo(() => Array.from({ length: GRID }, (_, r) => r), []);

  return (
    <ScrollView className="flex-1 bg-canvas" style={{ paddingTop: insets.top }}>
      <View className="px-4 pt-4 pb-2">
        <Text className="text-h2 font-sans text-fg">Model Testing</Text>
        <Text className="text-caption font-sans text-fg-muted mt-1">
          {active ? `Active: ${active.name}` : 'No saved model — train one first'}
        </Text>
      </View>

      <View className="mx-4 p-4 rounded-card bg-surface-1 border border-hairline">
        {rows.map((r) => (
          <View key={r} className="flex-row">
            {rows.map((c) => {
              const i = r * GRID + c;
              return (
                <Pressable
                  key={c}
                  accessibilityRole="button"
                  accessibilityLabel={`Grid cell row ${r + 1}, column ${c + 1}`}
                  accessibilityState={{ selected: cells[i] }}
                  onPress={() => toggle(i)}
                  className={`flex-1 aspect-square m-0.5 rounded-sm active:opacity-80 ${cells[i] ? 'bg-accent' : 'bg-surface-2'}`}
                />
              );
            })}
          </View>
        ))}
      </View>

      <View className="flex-row mx-4 mt-2">
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Clear"
          className="flex-1 flex-row items-center justify-center bg-surface-1 border border-hairline rounded-md py-3 mr-1 active:opacity-80"
          onPress={clear}>
          <Eraser color={colors.fg} size={18} strokeWidth={1.5} />
          <Text className="text-fg text-label font-sans ml-2">Clear</Text>
        </Pressable>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Run inference"
          accessibilityState={{ disabled: !active }}
          className={`flex-1 flex-row items-center justify-center bg-accent rounded-md py-3 ml-1 active:opacity-80 ${
            !active ? 'opacity-50' : ''
          }`}
          disabled={!active}
          onPress={onInfer}>
          <Play color={colors['accent-fg']} size={18} strokeWidth={1.5} />
          <Text className="text-accent-fg text-label font-sans ml-2">Run inference</Text>
        </Pressable>
      </View>

      {result ? (
        <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
          <Text className="text-fg text-body font-sans mb-2">
            Prediction: class {result.argmax}
          </Text>
          {result.probabilities.map((p, i) => (
            <View key={i} className="flex-row items-center mb-1">
              {/* Class index + probability: mono + tabular figures. */}
              <Text
                className="w-6 text-caption text-fg-muted font-mono"
                style={{ fontVariant: ['tabular-nums'] }}>
                {i}
              </Text>
              <View className="flex-1 h-3 rounded-pill bg-surface-2 overflow-hidden mr-2">
                <View
                  className={i === result.argmax ? 'h-3 bg-accent' : 'h-3 bg-series-1'}
                  style={{ width: `${Math.max(0, Math.min(1, p)) * 100}%` }}
                />
              </View>
              <Text
                className="w-12 text-right text-caption text-fg font-mono"
                style={{ fontVariant: ['tabular-nums'] }}>
                {(p * 100).toFixed(1)}%
              </Text>
            </View>
          ))}
          <Text className="text-caption font-sans text-fg-subtle mt-1">Real softmax over model logits.</Text>
        </View>
      ) : null}

      {error ? <ErrorBanner message={error} className="mx-4 my-3" /> : null}
      <View className="h-8" />
    </ScrollView>
  );
}
