import './theme/global.css'; // NativeWind: load the Tailwind layers
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, StatusBar, Text, TextInput, View } from 'react-native';
import { useColorScheme } from 'nativewind';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { DarkTheme, DefaultTheme, NavigationContainer, type Theme } from '@react-navigation/native';

import { AuthProvider } from './context/AuthContext';
import RootNavigator from './navigation/RootNavigator';
import { initServerConfig, recoverServerConfig } from './lib/serverConfig';
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
  const [configuration, setConfiguration] = useState<'loading' | 'ready' | 'failed'>('loading');
  const [attempt, setAttempt] = useState(0);
  const [recoveryUrl, setRecoveryUrl] = useState('');
  const [recovering, setRecovering] = useState(false);
  const [recoveryError, setRecoveryError] = useState('');
  const recover = async () => {
    setRecovering(true);
    setRecoveryError('');
    try {
      await recoverServerConfig(recoveryUrl);
      setConfiguration('ready');
    } catch {
      setRecoveryError('Could not save this server and clear the previous sign-in. Try again.');
    } finally { setRecovering(false); }
  };
  // Bind the REST client to the persisted backend URL (Settings → Backend server) before any call.
  useEffect(() => {
    let mounted = true;
    setConfiguration('loading');
    void initServerConfig().then(
      () => { if (mounted) setConfiguration('ready'); },
      () => { if (mounted) setConfiguration('failed'); },
    );
    return () => { mounted = false; };
  }, [attempt]);

  return (
    <SafeAreaProvider>
      {/* Match the status bar to the OS scheme: dark glyphs on the light canvas, light on dark. */}
      <StatusBar barStyle={colorScheme === 'dark' ? 'light-content' : 'dark-content'} />
      {configuration !== 'ready' ? (
        <View className="flex-1 bg-canvas items-center justify-center px-6">
          {configuration === 'loading' ? <>
            <ActivityIndicator />
            <Text className="mt-3 text-body text-fg">Loading saved settings…</Text>
          </> : <>
            <Text className="text-body text-fg">Could not load saved server settings.</Text>
            <Pressable accessibilityRole="button" accessibilityLabel="Retry loading settings"
              disabled={recovering} className="mt-4 px-5 py-3 rounded-md bg-accent" onPress={() => setAttempt((value) => value + 1)}>
              <Text className="text-label text-accent-fg">Retry</Text>
            </Pressable>
            <Text className="mt-6 text-body text-fg">Or choose a server and sign in again. Saved sign-in details will be cleared; datasets stay on this device.</Text>
            <TextInput accessibilityLabel="Recovery server URL" value={recoveryUrl} onChangeText={setRecoveryUrl}
              autoCapitalize="none" autoCorrect={false} keyboardType="url" editable={!recovering}
              placeholder="https://your-backend.example.com" className="mt-3 w-full px-3 py-2 rounded-md bg-surface-2 text-fg" />
            <Pressable accessibilityRole="button" accessibilityLabel="Use recovery server" disabled={recovering}
              className="mt-3 px-5 py-3 rounded-md bg-accent" onPress={() => { void recover(); }}>
              <Text className="text-label text-accent-fg">{recovering ? 'Saving…' : 'Use this server'}</Text>
            </Pressable>
            {!!recoveryError && <Text className="mt-3 text-body text-danger">{recoveryError}</Text>}
          </>}
        </View>
      ) : <AuthProvider>
        <NavigationContainer theme={colorScheme === 'dark' ? darkNavTheme : lightNavTheme}>
          <RootNavigator />
        </NavigationContainer>
      </AuthProvider>}
    </SafeAreaProvider>
  );
}
