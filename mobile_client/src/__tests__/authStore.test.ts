import EncryptedStorage from 'react-native-encrypted-storage';
import { getToken, setToken, clearToken } from '../lib/authStore';

jest.mock('react-native-encrypted-storage', () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
}));

// jest.Mocked<typeof EncryptedStorage> doesn't infer static-method types well;
// cast through unknown to get the spyable mock shape.
const ES = EncryptedStorage as unknown as {
  getItem: jest.MockedFunction<(key: string) => Promise<string | null>>;
  setItem: jest.MockedFunction<(key: string, value: string) => Promise<void>>;
  removeItem: jest.MockedFunction<(key: string) => Promise<void>>;
};

describe('authStore', () => {
  beforeEach(() => jest.clearAllMocks());

  test('setToken writes under the auth key', async () => {
    await setToken('jwt-123');
    expect(ES.setItem).toHaveBeenCalledWith('fedlearn.authToken', 'jwt-123');
  });

  test('getToken returns the stored token', async () => {
    ES.getItem.mockResolvedValue('jwt-123');
    await expect(getToken()).resolves.toBe('jwt-123');
  });

  test('getToken returns null and swallows a storage error', async () => {
    ES.getItem.mockRejectedValue(new Error('keystore locked'));
    await expect(getToken()).resolves.toBeNull();
  });

  test('clearToken removes the key', async () => {
    await clearToken();
    expect(ES.removeItem).toHaveBeenCalledWith('fedlearn.authToken');
  });
});
