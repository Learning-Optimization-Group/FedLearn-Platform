import React from 'react';
import { ActivityIndicator, Pressable, Text, View } from 'react-native';
import { Check, FileUp } from 'lucide-react-native';

import type { DataShape, DatasetSnapshot } from '../lib/datasetService';
import { datasetFit, type RunDataPlan } from '../lib/runDataPlan';
import { useThemeTokens } from '../theme/useThemeTokens';
import { ErrorBanner } from './ErrorBanner';

export interface TrainingDataCardProps {
  /** The joined run's data plan; the card shows only for a run on the device's own data. */
  plan: RunDataPlan;
  datasets: DatasetSnapshot[];
  selectedId: string | null;
  onSelect: (snapshotId: string) => void;
  onImport: () => void;
  importing: boolean;
  importError: string | null;
}

const shortId = (snapshotId: string) => snapshotId.slice(0, 8);

/**
 * Choosing the data a run trains on this phone (Stage 3 slice B2). Lists the datasets imported on the device, each
 * marked usable for this run or with the reasons it is not, and imports a file against the run's classes. The data
 * never leaves the phone; only the model update does.
 */
/** One example, in words: "140 numbers", or "a 32×32 color image" packaged with its raw pixels. */
function describeExample(shape: DataShape): string {
  if (shape.kind === 'vector') return `${shape.width} numbers`;
  return `a ${shape.height}×${shape.width} ${shape.channels === 1 ? 'grayscale' : 'color'} image (a .zip image package)`;
}

export function TrainingDataCard({
  plan, datasets, selectedId, onSelect, onImport, importing, importError,
}: TrainingDataCardProps) {
  const { colors } = useThemeTokens();
  if (plan.source !== 'LOCAL_SNAPSHOT') return null;
  const rows = datasets.map((d) => ({ d, reasons: datasetFit(d, plan) }));
  const anyFits = rows.some((r) => r.reasons.length === 0);

  return (
    <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
      <Text className="text-label font-sans font-semibold text-fg">This run trains on your own data</Text>
      <Text className="mt-1 text-caption font-sans text-fg-muted">
        {`Each example is ${describeExample(plan.shape)} with one label: ${plan.classNames.join(', ')}. `
          + (plan.maxExamples === null
            ? `Any number of examples, trained in batches of ${plan.batchSize}. `
            : `At most ${plan.maxExamples} examples. `)
          + 'Your data stays on this phone.'}
      </Text>

      {rows.map(({ d, reasons }) => {
        const selected = d.snapshotId === selectedId;
        const usable = reasons.length === 0;
        return (
          <Pressable
            key={d.snapshotId}
            accessibilityRole="button"
            accessibilityLabel={`Use dataset ${shortId(d.snapshotId)}`}
            accessibilityState={{ disabled: !usable, selected }}
            disabled={!usable}
            className={`mt-2 flex-row items-center p-3 rounded-md border ${
              selected ? 'border-accent' : 'border-hairline'
            } ${usable ? '' : 'opacity-60'}`}
            onPress={() => onSelect(d.snapshotId)}>
            <View className="flex-1">
              <Text className="text-caption font-mono text-fg">
                {`${d.recordCount} examples · ${shortId(d.snapshotId)}`}
              </Text>
              {!usable && (
                <Text className="mt-0.5 text-caption font-sans text-fg-subtle">{reasons.join(', ')}</Text>
              )}
            </View>
            {selected && <Check color={colors.accent} size={16} strokeWidth={1.5} />}
          </Pressable>
        );
      })}
      {!anyFits && (
        <Text className="mt-2 text-caption font-sans text-fg-subtle">
          No dataset on this phone fits this run yet. Import a .csv file, or a zipped dataset package.
        </Text>
      )}

      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Import a dataset file"
        accessibilityState={{ disabled: importing }}
        disabled={importing}
        className={`mt-3 flex-row items-center justify-center bg-surface-1 border border-hairline rounded-md py-3 active:opacity-80 ${
          importing ? 'opacity-50' : ''
        }`}
        onPress={onImport}>
        {importing ? (
          <ActivityIndicator color={colors.accent} />
        ) : (
          <>
            <FileUp color={colors.fg} size={16} strokeWidth={1.5} />
            <Text className="text-fg text-label font-sans ml-2">Import a dataset file</Text>
          </>
        )}
      </Pressable>
      {importError && <ErrorBanner message={importError} className="mt-3" />}
    </View>
  );
}

export default TrainingDataCard;
