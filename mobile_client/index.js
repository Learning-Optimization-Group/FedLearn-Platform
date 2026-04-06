/**
 * @format
 */

import {AppRegistry, NativeModules} from 'react-native';
import App from './src/App';
import {name as appName} from './app.json';

console.log('[FL] index.js loaded');
console.log('[FL] NativeModules keys:', Object.keys(NativeModules));
const turboProxy = typeof global !== 'undefined' && global.__turboModuleProxy;
console.log('[FL] turboModuleProxy available:', !!turboProxy);
if (turboProxy) {
  try {
    const mod = turboProxy('NativeFedLearnCore');
    console.log('[FL] NativeFedLearnCore via turboProxy:', mod);
  } catch (e) {
    console.log('[FL] turboProxy error:', e.message);
  }
}

AppRegistry.registerComponent(appName, () => App);
