import './theme/global.css'; // NativeWind: load the Tailwind layers
import React, { useEffect } from 'react';
import { StatusBar } from 'react-native';
import { useColorScheme } from 'nativewind';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { DarkTheme, DefaultTheme, NavigationContainer, type Theme } from '@react-navigation/native';

import { AuthProvider } from './context/AuthContext';
import RootNavigator from './navigation/RootNavigator';
import { initServerConfig } from './lib/serverConfig';
import { tokens } from './theme/tokens.generated';

// Navigation chrome (stack/tab backgrounds, headers, borders) themed from the same generated
// palette that drives the semantic classes, so react-navigation surfaces stay in lockstep with the
// token system in both OS schemes. Without this, react-navigation falls back to its own default
// palette — most visibly a default-styled tab bar on a token-styled canvas.
function buildNavTheme(
  base: Theme,
  palette: typeof tokens.colorLight | typeof tokens.colorDark,
): Theme {
  return {
    ...base,
    colors: {
      ...base.colors,
      primary: palette.accent,
      background: palette.canvas,
      card: palette['surface-1'],
      text: palette.fg,
      border: palette.hairline,
      notification: palette.danger,
    },
    fonts: {
      regular: { fontFamily: tokens.font.sans, fontWeight: '400' },
      medium: { fontFamily: tokens.font.sans, fontWeight: '500' },
      bold: { fontFamily: tokens.font.sans, fontWeight: '600' },
      heavy: { fontFamily: tokens.font.sans, fontWeight: '700' },
    },
  };
}

const lightNavTheme = buildNavTheme(DefaultTheme, tokens.colorLight);
const darkNavTheme = buildNavTheme(DarkTheme, tokens.colorDark);

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
        <NavigationContainer theme={colorScheme === 'dark' ? darkNavTheme : lightNavTheme}>
          <RootNavigator />
        </NavigationContainer>
      </AuthProvider>
    </SafeAreaProvider>
  );
}
