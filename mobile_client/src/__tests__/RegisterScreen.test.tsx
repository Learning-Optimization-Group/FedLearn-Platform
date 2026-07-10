// TE-12: behavior tests for the mobile RegisterScreen (submit / validation / error / navigation).
//
// The repo has no renderer installed (no react-test-renderer, react-dom, or
// @testing-library/react-native) and the task forbids adding deps, so we drive the real component
// with a tiny in-file harness: React's `useState` is swapped for an index-keyed cell store, and each
// setter re-invokes the component to produce a fresh element tree. RegisterScreen uses only `useState`
// (5 calls, verified) plus useAuth()/useNavigation() — both mocked — so the fixed hook order is stable.
// We keep registerValidation REAL (the pure logic under test at the screen's seam) and assert against
// the actual rendered element tree, not snapshots.

import React from 'react';

// --- useState harness -------------------------------------------------------
// mock-prefixed so jest.mock('react')'s factory may reference it (jest-hoist rule).
const mockHooks: { cells: unknown[]; idx: number; tree: React.ReactNode; render: () => void } = {
  cells: [],
  idx: 0,
  tree: null,
  render: () => {},
};
function mockUseState<S>(init: S | (() => S)): [S, (v: S | ((p: S) => S)) => void] {
  const i = mockHooks.idx++;
  if (i >= mockHooks.cells.length) {
    mockHooks.cells[i] = typeof init === 'function' ? (init as () => S)() : init;
  }
  const set = (v: S | ((p: S) => S)): void => {
    mockHooks.cells[i] = typeof v === 'function' ? (v as (p: S) => S)(mockHooks.cells[i] as S) : v;
    mockHooks.render(); // re-render synchronously, mirroring React's setState → re-render.
  };
  return [mockHooks.cells[i] as S, set];
}

const mockRegister = jest.fn<Promise<void>, [string, string, string]>();
const mockGoBack = jest.fn();
const mockNavigate = jest.fn();

jest.mock('react', () => ({ ...jest.requireActual('react'), useState: mockUseState }));
jest.mock('../context/AuthContext', () => ({ useAuth: () => ({ register: mockRegister }) }));
jest.mock('@react-navigation/native', () => ({
  useNavigation: () => ({ goBack: mockGoBack, navigate: mockNavigate }),
}));

import RegisterScreen from '../screens/RegisterScreen';

// --- element-tree helpers ---------------------------------------------------
type El = React.ReactElement<Record<string, unknown>>;

function props(el: El): Record<string, unknown> {
  return el.props;
}
function allElements(): El[] {
  const out: El[] = [];
  const walk = (node: React.ReactNode): void => {
    if (Array.isArray(node)) {
      node.forEach(walk);
      return;
    }
    if (React.isValidElement(node)) {
      const el = node as El;
      out.push(el);
      walk(props(el).children as React.ReactNode);
    }
  };
  walk(mockHooks.tree);
  return out;
}
function textOf(node: React.ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(textOf).join('');
  if (React.isValidElement(node)) return textOf((node as El).props.children as React.ReactNode);
  return '';
}
function screenText(): string {
  return textOf(mockHooks.tree);
}

function inputs(): El[] {
  return allElements().filter((e) => typeof props(e).onChangeText === 'function');
}
function pressables(): El[] {
  return allElements().filter((e) => typeof props(e).onPress === 'function');
}
function submitButton(): El {
  // The submit control is the pressable that is not the "Sign in" link (robust while it shows a spinner).
  const found = pressables().find((e) => !textOf(props(e).children as React.ReactNode).includes('Already have an account'));
  if (!found) throw new Error('submit button not found');
  return found;
}
function signInLink(): El {
  const found = pressables().find((e) => textOf(props(e).children as React.ReactNode).includes('Already have an account'));
  if (!found) throw new Error('sign-in link not found');
  return found;
}

// --- drivers ----------------------------------------------------------------
function renderScreen(): void {
  mockHooks.cells = [];
  mockHooks.idx = 0;
  mockHooks.render = () => {
    mockHooks.idx = 0;
    mockHooks.tree = (RegisterScreen as () => React.ReactNode)();
  };
  mockHooks.render();
}
function typeInput(index: number, value: string): void {
  const el = inputs()[index];
  if (!el) throw new Error(`no input at index ${index}`);
  (props(el).onChangeText as (t: string) => void)(value);
}
function fill(username: string, email: string, password: string): void {
  typeInput(0, username); // order per RegisterScreen JSX: username, email, password
  typeInput(1, email);
  typeInput(2, password);
}
function flush(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}
async function pressSubmit(): Promise<void> {
  await (props(submitButton()).onPress as () => unknown)();
  await flush();
}

beforeEach(() => {
  jest.resetAllMocks();
});

describe('RegisterScreen behavior (TE-12)', () => {
  test('renders the sign-up form: three inputs, a submit button, and a sign-in link', () => {
    renderScreen();
    expect(inputs()).toHaveLength(3);
    // email input carries keyboardType, password input is secured — confirms the field order.
    expect(props(inputs()[1] as El).keyboardType).toBe('email-address');
    expect(props(inputs()[2] as El).secureTextEntry).toBe(true);
    expect(textOf(props(submitButton()).children as React.ReactNode)).toContain('Create account');
    expect(textOf(props(signInLink()).children as React.ReactNode)).toContain('Already have an account');
  });

  test('short username shows the validation error and does NOT call the register API', async () => {
    renderScreen();
    fill('ab', 'bob@example.com', 'sekret1');
    await pressSubmit();
    expect(screenText()).toContain('Username must be at least 3 characters.');
    expect(mockRegister).not.toHaveBeenCalled();
  });

  test('invalid email shows the validation error and does NOT call the register API', async () => {
    renderScreen();
    fill('alice', 'not-an-email', 'sekret1');
    await pressSubmit();
    expect(screenText()).toContain('Enter a valid email address.');
    expect(mockRegister).not.toHaveBeenCalled();
  });

  test('short password shows the validation error and does NOT call the register API', async () => {
    renderScreen();
    fill('alice', 'alice@example.com', '12345');
    await pressSubmit();
    expect(screenText()).toContain('Password must be at least 6 characters.');
    expect(mockRegister).not.toHaveBeenCalled();
  });

  test('a valid submit calls register with cleaned values (username/email trimmed, password verbatim)', async () => {
    renderScreen();
    mockRegister.mockResolvedValueOnce(undefined);
    fill('  bob  ', '  bob@example.com  ', ' sekret1 '); // password keeps its surrounding spaces
    await pressSubmit();
    expect(mockRegister).toHaveBeenCalledTimes(1);
    expect(mockRegister).toHaveBeenCalledWith('bob', 'bob@example.com', ' sekret1 ');
    // no validation/error banner on the happy path
    expect(screenText()).not.toContain('must be at least');
    expect(screenText()).not.toContain('Sign-up failed');
  });

  test('a backend failure surfaces the server message (not "[object Object]" / no crash)', async () => {
    renderScreen();
    mockRegister.mockRejectedValueOnce({
      response: { data: { message: 'That username or email is already taken.' } },
    });
    fill('bob', 'bob@example.com', 'sekret1');
    await pressSubmit();
    expect(screenText()).toContain('That username or email is already taken.');
    expect(screenText()).not.toContain('[object Object]');
    expect(mockRegister).toHaveBeenCalledTimes(1);
  });

  test('a non-axios rejection surfaces the readable fallback (not "[object Object]" / no crash)', async () => {
    renderScreen();
    mockRegister.mockRejectedValueOnce(new Error('socket hang up'));
    fill('bob', 'bob@example.com', 'sekret1');
    await pressSubmit();
    expect(screenText()).toContain('Sign-up failed. That username or email may already be taken.');
    expect(screenText()).not.toContain('[object Object]');
  });

  test('while the register call is in flight the button is disabled and shows a spinner, then restores', async () => {
    renderScreen();
    let resolveReg: () => void = () => {};
    mockRegister.mockReturnValueOnce(new Promise<void>((res) => { resolveReg = res; }));
    fill('bob', 'bob@example.com', 'sekret1');
    const pending = (props(submitButton()).onPress as () => unknown)(); // don't await yet
    expect(props(submitButton()).disabled).toBe(true);
    expect(textOf(props(submitButton()).children as React.ReactNode)).not.toContain('Create account');
    resolveReg();
    await pending;
    await flush();
    expect(props(submitButton()).disabled).toBe(false);
    expect(textOf(props(submitButton()).children as React.ReactNode)).toContain('Create account');
  });

  test('the "Sign in" link navigates back to the login screen', () => {
    renderScreen();
    (props(signInLink()).onPress as () => void)();
    expect(mockGoBack).toHaveBeenCalledTimes(1);
  });
});
