import EncryptedStorage from 'react-native-encrypted-storage';
import {
  getSavedCredentials,
  saveCredentials,
  clearSavedCredentials,
} from '../lib/credentialStore';

jest.mock('react-native-encrypted-storage', () => ({
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
}));

const ES = EncryptedStorage as unknown as {
  getItem: jest.MockedFunction<(key: string) => Promise<string | null>>;
  setItem: jest.MockedFunction<(key: string, value: string) => Promise<void>>;
  removeItem: jest.MockedFunction<(key: string) => Promise<void>>;
};

const KEY = 'fedlearn.savedCredentials';

describe('credentialStore', () => {
  beforeEach(() => jest.clearAllMocks());

  test('saveCredentials writes JSON under the credentials key', async () => {
    await saveCredentials({ username: 'alice', password: 's3cret' });
    expect(ES.setItem).toHaveBeenCalledWith(
      KEY,
      JSON.stringify({ username: 'alice', password: 's3cret' }),
    );
  });

  test('getSavedCredentials returns the parsed pair', async () => {
    ES.getItem.mockResolvedValue(JSON.stringify({ username: 'alice', password: 's3cret' }));
    await expect(getSavedCredentials()).resolves.toEqual({ username: 'alice', password: 's3cret' });
  });

  test('getSavedCredentials returns null when nothing is stored', async () => {
    ES.getItem.mockResolvedValue(null);
    await expect(getSavedCredentials()).resolves.toBeNull();
  });

  test('getSavedCredentials returns null on malformed JSON', async () => {
    ES.getItem.mockResolvedValue('{not json');
    await expect(getSavedCredentials()).resolves.toBeNull();
  });

  test('getSavedCredentials returns null on a partial payload (missing password)', async () => {
    ES.getItem.mockResolvedValue(JSON.stringify({ username: 'alice' }));
    await expect(getSavedCredentials()).resolves.toBeNull();
  });

  test('getSavedCredentials swallows a keystore error and returns null', async () => {
    ES.getItem.mockRejectedValue(new Error('keystore locked'));
    await expect(getSavedCredentials()).resolves.toBeNull();
  });

  test('clearSavedCredentials removes the key', async () => {
    await clearSavedCredentials();
    expect(ES.removeItem).toHaveBeenCalledWith(KEY);
  });

  test('clearSavedCredentials swallows a removal error', async () => {
    ES.removeItem.mockRejectedValue(new Error('keystore locked'));
    await expect(clearSavedCredentials()).resolves.toBeUndefined();
  });
});
