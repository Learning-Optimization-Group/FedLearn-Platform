import networking from 'react-native/Libraries/Network/RCTNetworking';
import { clearNativeCookies } from '../lib/nativeCookies';

jest.mock('react-native/Libraries/Network/RCTNetworking', () => ({
  __esModule: true, default: { clearCookies: jest.fn() },
}));

const clear = networking.clearCookies as jest.Mock;
beforeEach(() => { jest.useFakeTimers(); clear.mockReset(); });
afterEach(() => { jest.clearAllTimers(); jest.useRealTimers(); });

test('an empty native cookie store is successful initialization', async () => {
  clear.mockImplementation((callback: (removed: boolean) => void) => callback(false));
  await expect(clearNativeCookies()).resolves.toBeUndefined();
});

test('an unavailable native cookie callback fails within a bounded startup window', async () => {
  const rejected = expect(clearNativeCookies()).rejects.toThrow();
  jest.advanceTimersByTime(6000);
  await rejected;
});

test('a native cookie error fails closed instead of proceeding with a previous session', async () => {
  clear.mockImplementation(() => { throw new Error('native unavailable'); });
  await expect(clearNativeCookies()).rejects.toThrow();
});
