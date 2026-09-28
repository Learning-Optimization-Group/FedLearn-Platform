import React, { useCallback, useState } from 'react';
import { Pressable, Text, View } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';

import { deleteDataset, listDatasets, type DatasetSnapshot } from '../lib/datasetService';
import { readError } from '../lib/errors';
import { useTraining } from '../state/TrainingContext';
import { ErrorBanner } from './ErrorBanner';

const shortId = (snapshotId: string) => snapshotId.slice(0, 8);

/**
 * The datasets imported on this phone (Stage 3 slice B2), each deletable except the one a run is training. The data
 * lives only in the app's private storage; deleting it here removes it from the phone.
 */
export function DeviceDatasetsCard() {
  const { state } = useTraining();
  const inUse = state.datasetInUse;
  const [datasets, setDatasets] = useState<DatasetSnapshot[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      setDatasets(await listDatasets());
    } catch (e) {
      setError(readError(e));
    }
  }, []);
  useFocusEffect(
    useCallback(() => {
      void reload();
    }, [reload]),
  );

  const remove = useCallback(async (snapshotId: string) => {
    setError(null);
    try {
      await deleteDataset(snapshotId, inUse ? [inUse] : []);
      await reload();
    } catch (e) {
      setError(readError(e));
    }
  }, [inUse, reload]);

  return (
    <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
      <Text className="text-label font-sans font-semibold text-fg mb-1">Datasets on this phone</Text>
      <Text className="text-caption font-sans text-fg-muted">
        Imported for runs that train on your own data. They never leave this phone.
      </Text>
      {datasets !== null && datasets.length === 0 && (
        <Text className="mt-2 text-caption font-sans text-fg-subtle">
          No datasets on this phone. Import one from Home when a run trains on your own data.
        </Text>
      )}
      {(datasets ?? []).map((d) => {
        const pinned = d.snapshotId === inUse;
        return (
          <View key={d.snapshotId} className="mt-2 flex-row items-center p-3 rounded-md border border-hairline">
            <View className="flex-1">
              <Text className="text-caption font-mono text-fg">{`${d.recordCount} examples · ${shortId(d.snapshotId)}`}</Text>
              <Text className="mt-0.5 text-caption font-sans text-fg-muted">
                {pinned ? 'In use by training' : d.classNames.join(', ')}
              </Text>
            </View>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel={`Delete dataset ${shortId(d.snapshotId)}`}
              accessibilityState={{ disabled: pinned }}
              disabled={pinned}
              className={`px-3 py-2 rounded-md border border-hairline active:opacity-80 ${pinned ? 'opacity-50' : ''}`}
              onPress={() => {
                void remove(d.snapshotId);
              }}>
              <Text className="text-danger text-caption font-sans">Delete</Text>
            </Pressable>
          </View>
        );
      })}
      {error && <ErrorBanner message={error} className="mt-3" />}
    </View>
  );
}

export default DeviceDatasetsCard;
