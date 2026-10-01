import EncryptedStorage from 'react-native-encrypted-storage';

const KEY = 'fedlearn.diagnostics.v1';
const DEFAULT_MAX_EVENTS = 500;
const MAX_DETAIL_CHARS = 1000;

interface Storage {
  getItem(key: string): Promise<string | null>;
  setItem(key: string, value: string): Promise<void>;
  removeItem(key: string): Promise<void>;
}

interface Event {
  at: string;
  phase: string;
  detail: string;
}

function scrub(value: string): string {
  return value
    .replace(/-----BEGIN ([A-Z ]*(?:PRIVATE KEY|CERTIFICATE))-----[\s\S]*?-----END \1-----/g, '[credential]')
    .replace(/\b[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b/g, '[credential]')
    .replace(/Bearer\s+\S+/gi, '[credential]')
    .replace(/(?:https?|grpc):\/\/[^\s]+/gi, '[endpoint]')
    .replace(/\/(?:data|storage|Users|home|private|var)\/[^\s]+/gi, '[local path]')
    .replace(/\b(?:token|password|secret|api[_-]?key)\s*[:=]\s*[^\s,;]+/gi, '[credential]')
    .replace(/[\r\n]+/g, ' ')
    .slice(0, MAX_DETAIL_CHARS);
}

export class DiagnosticJournal {
  private tail: Promise<void> = Promise.resolve();

  constructor(private readonly storage: Storage, private readonly maxEvents = DEFAULT_MAX_EVENTS) {}

  private async read(): Promise<Event[]> {
    const raw = await this.storage.getItem(KEY);
    if (raw === null) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) throw new Error('Invalid diagnostic journal');
    return parsed as Event[];
  }

  append(phase: string, detail: string): Promise<void> {
    const event: Event = { at: new Date().toISOString(), phase: scrub(phase), detail: scrub(detail) };
    const task = this.tail.then(async () => {
      const events = await this.read();
      await this.storage.setItem(KEY, JSON.stringify([...events, event].slice(-this.maxEvents)));
    });
    this.tail = task.catch(() => undefined);
    return task;
  }

  async exportText(): Promise<string> {
    await this.tail;
    const events = await this.read();
    return ['FedLearn on-device diagnostics', ...events.map((e) => `${e.at} ${e.phase}: ${e.detail}`)].join('\n');
  }

  clear(): Promise<void> {
    const task = this.tail.then(() => this.storage.removeItem(KEY));
    this.tail = task.catch(() => undefined);
    return task;
  }
}

export const diagnosticJournal = new DiagnosticJournal(EncryptedStorage);
