// MO-16: readError turns an unknown thrown value into a human-readable string. The live phone test
// surfaced "[object Object]" on the join screen — TrainingScreen rendered an axios error via String(e).
// readError must prefer the backend-supplied message, then Error.message, and NEVER render "[object Object]".
import { readError } from '../lib/errors';

describe('readError (MO-16)', () => {
  test('returns a plain string as-is', () => {
    expect(readError('boom')).toBe('boom');
  });

  test('prefers the backend-supplied message on an axios-style error', () => {
    const axiosErr = {
      message: 'Request failed with status code 409',
      response: { data: { message: 'Run is not currently running (status=COMPLETED)' } },
    };
    expect(readError(axiosErr)).toBe('Run is not currently running (status=COMPLETED)');
  });

  test('falls back to Error.message when there is no backend message', () => {
    expect(readError(new Error('No active run for this project yet — the owner needs to start one.')))
      .toBe('No active run for this project yet — the owner needs to start one.');
  });

  test('never returns "[object Object]" for a bare object', () => {
    const r = readError({ some: 'object' });
    expect(r).not.toBe('[object Object]');
    expect(r.length).toBeGreaterThan(0);
  });

  test('handles null/undefined without throwing', () => {
    expect(typeof readError(null)).toBe('string');
    expect(typeof readError(undefined)).toBe('string');
    expect(readError(null)).not.toBe('[object Object]');
  });

  test('ignores a blank/whitespace backend message and Error.message, using the fallback', () => {
    const r = readError({ response: { data: { message: '   ' } }, message: '' });
    expect(r).not.toBe('[object Object]');
    expect(r.trim().length).toBeGreaterThan(0);
  });
});
