// Manual mock for react-native-encrypted-storage — satisfies the native module
// requirement in test environments where no bridge is available.
const EncryptedStorage = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};

export default EncryptedStorage;
