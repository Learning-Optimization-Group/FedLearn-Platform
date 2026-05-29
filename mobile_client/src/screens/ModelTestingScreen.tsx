import React, { useCallback, useMemo, useState } from 'react';
import { Pressable, ScrollView, Text, View } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { Eraser, Sparkles } from 'lucide-react-native';

import nativeCore, { type InferResult } from '../lib/nativeCore';
import { listModels, type SavedModel } from '../lib/modelStore';

const GRID = 8; // 8x8 input grid; the RN layer flattens it for the model (input shape is model-specific)
const N = GRID * GRID;

export function ModelTestingScreen() {
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
    <ScrollView className="flex-1 bg-background">
      <View className="px-4 pt-4 pb-2">
        <Text className="text-2xl font-extrabold text-foreground">Model Testing</Text>
        <Text className="text-xs text-muted mt-1">
          {active ? `Active: ${active.name}` : 'No saved model — train one first'}
        </Text>
      </View>

      <View className="mx-4 p-2 rounded-2xl bg-surface border border-border">
        {rows.map((r) => (
          <View key={r} className="flex-row">
            {rows.map((c) => {
              const i = r * GRID + c;
              return (
                <Pressable
                  key={c}
                  onPress={() => toggle(i)}
                  className={`flex-1 aspect-square m-0.5 rounded ${cells[i] ? 'bg-primary' : 'bg-surface-muted'}`}
                />
              );
            })}
          </View>
        ))}
      </View>

      <View className="flex-row mx-4 mt-2">
        <Pressable
          className="flex-1 flex-row items-center justify-center bg-surface-muted rounded-xl py-3 mr-1"
          onPress={clear}>
          <Eraser color="oklch(0.22 0.02 250)" size={18} />
          <Text className="text-foreground font-semibold ml-2">Clear</Text>
        </Pressable>
        <Pressable
          className="flex-1 flex-row items-center justify-center bg-primary rounded-xl py-3 ml-1"
          disabled={!active}
          onPress={onInfer}>
          <Sparkles color="white" size={18} />
          <Text className="text-primary-foreground font-semibold ml-2">Run inference</Text>
        </Pressable>
      </View>

      {result ? (
        <View className="mx-4 mt-3 p-4 rounded-2xl bg-surface border border-border">
          <Text className="text-foreground font-semibold mb-2">
            Prediction: class {result.argmax}
          </Text>
          {result.probabilities.map((p, i) => (
            <View key={i} className="flex-row items-center mb-1">
              <Text className="w-6 text-xs text-muted">{i}</Text>
              <View className="flex-1 h-3 rounded-full bg-surface-muted overflow-hidden mr-2">
                <View
                  className={i === result.argmax ? 'h-3 bg-primary' : 'h-3 bg-accent'}
                  style={{ width: `${Math.max(0, Math.min(1, p)) * 100}%` }}
                />
              </View>
              <Text className="w-12 text-right text-xs text-foreground">{(p * 100).toFixed(1)}%</Text>
            </View>
          ))}
          <Text className="text-[10px] text-muted mt-1">Real softmax over model logits.</Text>
        </View>
      ) : null}

      {error ? (
        <View className="mx-4 my-3 p-3 rounded-xl bg-danger">
          <Text className="text-primary-foreground text-sm">{error}</Text>
        </View>
      ) : null}
      <View className="h-8" />
    </ScrollView>
  );
}
