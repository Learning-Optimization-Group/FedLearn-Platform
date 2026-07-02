import './theme/global.css'; // NativeWind: load the Tailwind layers
import React, { useEffect } from 'react';
import { StatusBar } from 'react-native';
import { useColorScheme } from 'nativewind';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { NavigationContainer } from '@react-navigation/native';

import { AuthProvider } from './context/AuthContext';
import RootNavigator from './navigation/RootNavigator';
import { initServerConfig } from './lib/serverConfig';

export default function App() {
  const { colorScheme } = useColorScheme();
  // Bind the REST client to the persisted backend URL (Settings → Backend server) before any call.
  useEffect(() => {
    void initServerConfig().catch(() => {});
  }, []);

  return (
    <SafeAreaProvider>
      {/* Match the status bar to the OS scheme: dark glyphs on the light canvas, light on dark. */}
      <StatusBar barStyle={colorScheme === 'dark' ? 'light-content' : 'dark-content'} />
      <AuthProvider>
        <NavigationContainer>
          <RootNavigator />
        </NavigationContainer>
      </AuthProvider>
    </SafeAreaProvider>
  );
}
