// On-device saved-model registry (metadata only; the .pt files live in app-private encrypted
// storage). The round loop auto-saves an entry after a run (15-LLD: "Model library auto-saves
// metadata after each FL round"); the Library + Testing screens read it.
import EncryptedStorage from 'react-native-encrypted-storage';

export interface SavedModel {
  name: string;
  path: string; // app-private path of the .pt
  sha256: string; // integrity hash (verified before load)
  tier: string; // 1M | 10M | 100M
  round: number; // last FL round this snapshot reflects
  savedAt: string; // ISO timestamp
}

const KEY = 'fedlearn.models';

export async function listModels(): Promise<SavedModel[]> {
  const raw = await EncryptedStorage.getItem(KEY);
  return raw ? (JSON.parse(raw) as SavedModel[]) : [];
}

export async function saveModelMeta(model: SavedModel): Promise<void> {
  const all = await listModels();
  const next = [model, ...all.filter((m) => m.path !== model.path)];
  await EncryptedStorage.setItem(KEY, JSON.stringify(next));
}
