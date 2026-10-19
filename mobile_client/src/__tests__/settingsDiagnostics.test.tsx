import type * as ReactTypes from 'react';

jest.mock('react', () =>
  jest.requireActual<typeof import('../testUtils/componentHarness')>('../testUtils/componentHarness')
    .createReactMock(),
);
jest.mock('react-native-safe-area-context', () => ({ useSafeAreaInsets: () => ({ top: 0 }) }));
jest.mock('react-native-device-info', () => ({
  getSystemVersion: () => '10', getModel: () => 'vivo 1805', getVersion: () => '2.0', getBuildNumber: () => '1',
}));
jest.mock('lucide-react-native', () => new Proxy({}, { get: () => () => null }));
jest.mock('../context/AuthContext', () => ({ useAuth: () => ({ username: 'tester', logout: jest.fn() }) }));
jest.mock('../theme/useThemeTokens', () => ({
  useThemeTokens: () => ({ colors: new Proxy({}, { get: () => '#000000' }) }),
}));
import { SettingsScreen } from '../screens/SettingsScreen';
import EncryptedStorage from 'react-native-encrypted-storage';
import { press, pressableByLabel, renderComponent, screenText } from '../testUtils/componentHarness';

test('Settings can clear the persisted diagnostic journal', async () => {
  (EncryptedStorage.removeItem as jest.Mock).mockResolvedValue(undefined);
  renderComponent(() => (SettingsScreen as unknown as () => ReactTypes.ReactNode)());
  await press(pressableByLabel('Clear training diagnostics'));
  expect(EncryptedStorage.removeItem).toHaveBeenCalledWith('fedlearn.diagnostics.v1');
  expect(screenText()).toContain('Diagnostics cleared');
});
