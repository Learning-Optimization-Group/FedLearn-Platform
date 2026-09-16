// MO-6: pins the register-flow validation contract (rules extracted from RegisterScreen into a pure fn).
import {
  validateRegistration,
  MIN_USERNAME_LEN,
  MIN_PASSWORD_LEN,
} from '../lib/registerValidation';

const good = { username: 'alice', email: 'alice@example.com', password: 'hunter2' };

describe('validateRegistration (MO-6)', () => {
  test('accepts a well-formed registration and returns trimmed username/email', () => {
    const r = validateRegistration({ ...good, username: '  alice  ', email: '  alice@example.com ' });
    expect(r).toEqual({ ok: true, username: 'alice', email: 'alice@example.com', password: 'hunter2' });
  });

  test('does NOT trim the password — leading/trailing spaces are legal password characters', () => {
    const r = validateRegistration({ ...good, password: '  spaced  ' });
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.password).toBe('  spaced  ');
  });

  test('rejects a username shorter than the minimum (after trimming)', () => {
    expect(validateRegistration({ ...good, username: 'ab' })).toEqual({
      ok: false,
      error: `Username must be at least ${MIN_USERNAME_LEN} characters.`,
    });
    // whitespace-padded but too short once trimmed
    expect(validateRegistration({ ...good, username: '  a  ' }).ok).toBe(false);
  });

  test('accepts a username exactly at the minimum length (boundary)', () => {
    expect(validateRegistration({ ...good, username: 'abc' }).ok).toBe(true);
  });

  test.each([
    ['no @', 'aliceexample.com'],
    ['no dot domain', 'alice@examplecom'],
    ['embedded space', 'alice @example.com'],
    ['empty', ''],
    ['trailing text only', '@.'],
  ])('rejects an invalid email (%s)', (_label, email) => {
    expect(validateRegistration({ ...good, email })).toEqual({
      ok: false,
      error: 'Enter a valid email address.',
    });
  });

  test('rejects a password shorter than the minimum', () => {
    expect(validateRegistration({ ...good, password: '12345' })).toEqual({
      ok: false,
      error: `Password must be at least ${MIN_PASSWORD_LEN} characters.`,
    });
  });

  test('accepts a password exactly at the minimum length (boundary)', () => {
    expect(validateRegistration({ ...good, password: '123456' }).ok).toBe(true);
  });

  test('checks fields in order: username first, then email, then password', () => {
    // all three invalid -> username error wins (matches the screen short-circuit order)
    const r = validateRegistration({ username: 'x', email: 'bad', password: '1' });
    expect(r).toEqual({ ok: false, error: `Username must be at least ${MIN_USERNAME_LEN} characters.` });
  });
});
