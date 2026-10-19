import { SubmittedRoundStore } from '../lib/submittedRoundStore';

function memoryStore() {
  const data = new Map<string, string>();
  return {
    getItem: async (key: string) => data.get(key) ?? null,
    setItem: async (key: string, value: string) => { data.set(key, value); },
    data,
  };
}

test('remembers the last uploaded round for each run across store instances', async () => {
  const storage = memoryStore();
  const first = new SubmittedRoundStore(storage);
  await first.save('run-1', 1);
  await first.save('run-1', 2);
  await first.save('run-2', 3);
  const reopened = new SubmittedRoundStore(storage);
  expect(await reopened.load('run-1')).toBe(2);
  expect(await reopened.load('run-2')).toBe(3);
  expect(await reopened.load('unknown')).toBeNull();
});

test('does not lose a newer checkpoint to a late older write', async () => {
  const store = new SubmittedRoundStore(memoryStore());
  await Promise.all([store.save('run-1', 3), store.save('run-1', 2)]);
  expect(await store.load('run-1')).toBe(3);
});

test('refuses corrupt checkpoint storage rather than assuming the phone has not uploaded', async () => {
  const storage = memoryStore();
  storage.data.set('fedlearn.submittedRounds', '{');
  await expect(new SubmittedRoundStore(storage).load('run-1')).rejects.toThrow(/checkpoint/i);
});
