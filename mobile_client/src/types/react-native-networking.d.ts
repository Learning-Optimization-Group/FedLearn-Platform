declare module 'react-native/Libraries/Network/RCTNetworking' {
  const networking: { clearCookies(callback: (removed: boolean) => void): void };
  export default networking;
}
