import { DiagnosticJournal } from '../lib/diagnosticJournal';

function memoryStorage() {
  const data = new Map<string, string>();
  return {
    getItem: async (key: string) => data.get(key) ?? null,
    setItem: async (key: string, value: string) => { data.set(key, value); },
    removeItem: async (key: string) => { data.delete(key); },
    data,
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

test('redacts a JWT and a multi-line private key before persistence', async () => {
  const storage = memoryStorage();
  const journal = new DiagnosticJournal(storage);
  await journal.append('error',
    'bad eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c ' +
    '-----BEGIN PRIVATE KEY-----\nSECRET_KEY_BODY\n-----END PRIVATE KEY-----');
  const persisted = storage.data.get('fedlearn.diagnostics.v1') ?? '';
  expect(persisted).not.toMatch(/eyJhbGci|SECRET_KEY_BODY|BEGIN PRIVATE KEY/);
});

test('keeps the newest 500 entries and bounds one error message to 1000 characters', async () => {
  const journal = new DiagnosticJournal(memoryStorage());
  for (let i = 0; i <= 500; i += 1) await journal.append('round', `event-${i}`);
  await journal.append('error', 'x'.repeat(1200));
  const lines = (await journal.exportText()).split('\n');
  expect(lines).toHaveLength(501);
  expect(lines.join('\n')).not.toContain('event-0');
  expect(lines.at(-1)?.match(/x/g)).toHaveLength(1000);
});

test('clears persisted diagnostics even after concurrent appends', async () => {
  const storage = memoryStorage();
  const journal = new DiagnosticJournal(storage);
  await Promise.all([journal.append('round', '1'), journal.append('error', 'failure')]);
  await journal.clear();
  expect(await new DiagnosticJournal(storage).exportText()).toBe('FedLearn on-device diagnostics');
});
