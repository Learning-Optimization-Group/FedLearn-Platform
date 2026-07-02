// App entry — registers the root component. One product name: "FedLearn" (C5 §8; retires "FedMob").
import './src/lib/polyfills'; // TextEncoder/TextDecoder for @stomp/stompjs — must load before App
import { AppRegistry } from 'react-native';
import App from './src/App';
import { name as appName } from './app.json';

AppRegistry.registerComponent(appName, () => App);
