import { DiagnosticJournal } from '../lib/diagnosticJournal';

function memoryStorage() {
  const data = new Map<string, string>();
  return {
    getItem: async (key: string) => data.get(key) ?? null,
    setItem: async (key: string, value: string) => { data.set(key, value); },
  };
}

test('keeps bounded events across app restarts with useful round and phase details', async () => {
  const storage = memoryStorage();
  const journal = new DiagnosticJournal(storage, 2);
  await journal.append('join', 'registered');
  await journal.append('round', 'uploaded round 1');
  await journal.append('round', 'uploaded round 2');
  const exported = await new DiagnosticJournal(storage, 2).exportText();
  expect(exported).toContain('uploaded round 1');
  expect(exported).toContain('uploaded round 2');
  expect(exported).not.toContain('registered');
});

test('redacts credentials, URLs, and device-local paths before an event is stored or shared', async () => {
  const journal = new DiagnosticJournal(memoryStorage());
  await journal.append('error', 'Bearer secret123 at https://host.local/x?token=abc /data/user/0/com.app/files/model.pte');
  const text = await journal.exportText();
  expect(text).not.toMatch(/secret123|abc|host.local|\/data\/user/);
  expect(text).toContain('error');
});

test('serializes concurrent appends without losing a failure', async () => {
  const journal = new DiagnosticJournal(memoryStorage());
  await Promise.all([journal.append('round', '1'), journal.append('error', 'failure')]);
  expect(await journal.exportText()).toMatch(/round.*1[\s\S]*error.*failure/);
});
