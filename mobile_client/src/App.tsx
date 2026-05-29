import './theme/global.css'; // NativeWind: load the Tailwind layers
import React, { useEffect } from 'react';
import { StatusBar } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { NavigationContainer } from '@react-navigation/native';

import { AppNavigator } from './navigation/AppNavigator';
import { configureApi } from './lib/restClient';

// FEDLEARN_API_URL comes from native build config (15-LLD §8). Wire it via react-native-config /
// .env in the app project; this env read is a placeholder until that wiring lands.
const API_URL = (typeof process !== 'undefined' && process.env.FEDLEARN_API_URL) || '';

export default function App() {
  useEffect(() => {
    if (API_URL) configureApi(API_URL);
  }, []);

  return (
    <SafeAreaProvider>
      <StatusBar barStyle="dark-content" />
      <NavigationContainer>
        <AppNavigator />
      </NavigationContainer>
    </SafeAreaProvider>
  );
}
