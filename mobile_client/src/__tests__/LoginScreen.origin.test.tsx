import type * as ReactTypes from 'react';

const mockEffects: Array<() => unknown> = [];
const mockLogin = jest.fn();
const mockNavigate = jest.fn();
type MockUseRef = (initial: boolean) => { current: boolean };
type MockCookieCallback = (removed: boolean) => void;

jest.mock('react', () => {
  const mock = jest.requireActual<typeof import('../testUtils/componentHarness')>('../testUtils/componentHarness')
    .createReactMock();
  return { ...mock, useEffect: (effect: () => unknown) => {
    const seen = (mock.useRef as MockUseRef)(false);
    if (!seen.current) { seen.current = true; mockEffects.push(effect); }
  } };
});
jest.mock('@react-navigation/native', () => ({ useNavigation: () => ({ navigate: mockNavigate }) }));
jest.mock('react-native-safe-area-context', () => ({ SafeAreaView: 'SafeAreaView' }));
jest.mock('lucide-react-native', () => new Proxy({}, { get: () => () => null }));
jest.mock('../context/AuthContext', () => ({ useAuth: () => ({ login: mockLogin }) }));
jest.mock('../theme/useThemeTokens', () => ({
  useThemeTokens: () => ({ colors: new Proxy({}, { get: () => '#000000' }) }),
}));
jest.mock('react-native/Libraries/Network/RCTNetworking', () => ({
  __esModule: true, default: { clearCookies: (callback: MockCookieCallback) => callback(false) },
}));
jest.mock('react-native-encrypted-storage', () => {
  const values = new Map<string, string>();
  return {
    getItem: jest.fn(async (key: string) => values.get(key) ?? null),
    setItem: jest.fn(async (key: string, value: string) => { values.set(key, value); }),
    removeItem: jest.fn(async (key: string) => { values.delete(key); }),
    clear: jest.fn(async () => { values.clear(); }),
  };
});

import EncryptedStorage from 'react-native-encrypted-storage';
import LoginScreen from '../screens/LoginScreen';
import { getServerBaseUrl, initServerConfig } from '../lib/serverConfig';
import { allElements, flush, press, pressableByLabel, renderComponent } from '../testUtils/componentHarness';

const FIRST = 'https://first.example.invalid';
const SECOND = 'https://second.example.invalid';
const PASSWORD = 'placeholder-saved-password';

function field(label: string) {
  const element = allElements().find((entry) => entry.props.accessibilityLabel === label && typeof entry.props.onChangeText === 'function');
  if (!element) throw new Error('Missing input ' + label);
  return element;
}

function type(label: string, value: string) {
  (field(label).props.onChangeText as (text: string) => void)(value);
}

async function mount() {
  renderComponent(() => (LoginScreen as unknown as () => ReactTypes.ReactNode)());
  mockEffects.splice(0).forEach((effect) => effect());
  await flush();
}

beforeEach(async () => {
  jest.clearAllMocks();
  mockEffects.splice(0);
  mockLogin.mockResolvedValue(undefined);
  await EncryptedStorage.clear();
  await EncryptedStorage.setItem('fedlearn.serverBaseUrl', FIRST);
  await EncryptedStorage.setItem('fedlearn.savedCredentials', JSON.stringify({ username: 'probe', password: PASSWORD }));
  await initServerConfig();
});

test('changing the server clears the displayed saved password before sign-in', async () => {
  await mount();
  expect(field('Password').props.value).toBe(PASSWORD);
  await press(pressableByLabel('Server settings'));
  type('Server URL', SECOND);
  expect(field('Password').props.value).toBe('');
  await press(pressableByLabel('Sign in'));
  expect(mockLogin).not.toHaveBeenCalled();
});

test('equivalent server formatting preserves the saved password for its issuer', async () => {
  await mount();
  await press(pressableByLabel('Server settings'));
  type('Server URL', 'HTTPS://FIRST.example.invalid:443/');
  expect(field('Password').props.value).toBe(PASSWORD);
  await press(pressableByLabel('Sign in'));
  expect(mockLogin).toHaveBeenCalledWith('probe', PASSWORD, true);
});

test('a delayed saved-password load cannot fill another server form', async () => {
  const read = EncryptedStorage.getItem as jest.Mock;
  const original = read.getMockImplementation()!;
  let resolve!: (value: string) => void;
  const pending = new Promise<string>((done) => { resolve = done; });
  read.mockImplementation((key: string) => key === 'fedlearn.savedCredentials' ? pending : original(key));
  try {
    await mount();
    await press(pressableByLabel('Server settings'));
    type('Server URL', SECOND);
    resolve(JSON.stringify({ schemaVersion: 1, origin: FIRST, username: 'probe', password: PASSWORD }));
    await flush();
    expect(field('Password').props.value).toBe('');
  } finally {
    read.mockImplementation(original);
  }
});

test('creating an account uses the server selected on the login form', async () => {
  await mount();
  await press(pressableByLabel('Server settings'));
  type('Server URL', SECOND);
  await press(pressableByLabel('Create an account'));
  await expect(getServerBaseUrl()).resolves.toBe(SECOND);
  expect(mockNavigate).toHaveBeenCalledWith('Register');
  expect(mockLogin).not.toHaveBeenCalled();
});
