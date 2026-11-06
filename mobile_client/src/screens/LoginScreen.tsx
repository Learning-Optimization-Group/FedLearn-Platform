import React, { useEffect, useRef, useState } from 'react';
import { View, Text, TextInput, Pressable, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { ChevronDown, ChevronRight, Server, Eye, EyeOff, Check } from 'lucide-react-native';
import { useAuth } from '../context/AuthContext';
import { useThemeTokens } from '../theme/useThemeTokens';
import { getServerBaseUrl, normalizeBaseUrl, setServerBaseUrl } from '../lib/serverConfig';
import { getSavedCredentials } from '../lib/credentialStore';
import type { RootStackParamList } from '../navigation/types';

export default function LoginScreen() {
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const { login } = useAuth();
  const { colors } = useThemeTokens();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // New login affordances: editable server URL up front, show/hide password, and opt-in save.
  const [serverUrl, setServerUrl] = useState('');
  const [showServer, setShowServer] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [savePassword, setSavePassword] = useState(false);
  const formOrigin = useRef<string | null>(null);
  const formVersion = useRef(0);

  // On mount: prefill the server URL (persisted or default) and any saved credentials.
  useEffect(() => {
    const version = formVersion.current;
    let mounted = true;
    void getServerBaseUrl().then(async (base) => {
      if (!mounted || version !== formVersion.current) return;
      const origin = normalizeBaseUrl(base);
      formOrigin.current = origin;
      setServerUrl(origin);
      const credentials = await getSavedCredentials(origin);
      if (!mounted || version !== formVersion.current || formOrigin.current !== origin || !credentials) return;
      setUsername(credentials.username);
      setPassword(credentials.password);
      setSavePassword(true);
    }).catch(() => { if (mounted) setError('Could not load saved server settings.'); });
    return () => { mounted = false; };
  }, []);

  const onServerChange = (value: string) => {
    let origin: string | null = null;
    try { origin = normalizeBaseUrl(value); } catch { /* Do not preserve a password for an incomplete authority. */ }
    if (origin !== formOrigin.current) {
      formVersion.current++;
      formOrigin.current = origin;
      setUsername('');
      setPassword('');
      setSavePassword(false);
      setError(null);
    }
    setServerUrl(value);
  };

  const onSubmit = async () => {
    let origin: string;
    try { origin = normalizeBaseUrl(serverUrl); }
    catch (e) { setError(e instanceof Error ? e.message : 'Enter a valid server URL.'); return; }
    if (origin !== formOrigin.current) {
      setPassword('');
      setError('Enter credentials for this server.');
      return;
    }
    if (!username.trim() || !password) { setError('Enter your username and password.'); return; }
    setBusy(true);
    setError(null);
    try {
      // Point the REST client at the entered backend BEFORE logging in, so the credentials
      // and the resulting token go to the host the user actually chose (not a stale default).
      try {
        await setServerBaseUrl(origin);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Enter a valid server URL.');
        return;
      }
      await login(username.trim(), password, savePassword);
    } catch {
      setError('Sign-in failed. Check your username and password.');
    } finally {
      setBusy(false);
    }
  };

  const onCreateAccount = async () => {
    setBusy(true);
    setError(null);
    try {
      const origin = await setServerBaseUrl(normalizeBaseUrl(serverUrl));
      setServerUrl(origin);
      navigation.navigate('Register');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save the server settings.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <SafeAreaView className="flex-1 bg-canvas">
      <View className="px-4 pt-4">
        <Text className="text-h2 font-sans text-fg">Sign in</Text>
      </View>
      <View className="mx-4 mt-4 p-4 rounded-card bg-surface-1 border border-hairline">
        {/* Server — configurable up front, collapsed to the current URL (parity with desktop). */}
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Server settings"
          accessibilityState={{ expanded: showServer }}
          className="flex-row items-center active:opacity-80"
          hitSlop={{ top: 6, bottom: 6, left: 4, right: 4 }}
          onPress={() => setShowServer((v) => !v)}>
          {showServer
            ? <ChevronDown color={colors.fg} size={16} strokeWidth={1.5} />
            : <ChevronRight color={colors.fg} size={16} strokeWidth={1.5} />}
          <View className="ml-1">
            <Server color={colors.fg} size={16} strokeWidth={1.5} />
          </View>
          <Text className="ml-2 text-label font-sans text-fg">Server</Text>
          <Text
            className="ml-2 flex-1 text-caption font-mono text-fg-subtle"
            numberOfLines={1}>
            {serverUrl}
          </Text>
        </Pressable>
        {showServer && (
          <View className="mt-2 mb-1">
            <TextInput
              className="px-3 py-2 rounded-md bg-surface-2 text-body font-mono text-fg border border-hairline"
              accessibilityLabel="Server URL"
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
              value={serverUrl}
              onChangeText={onServerChange}
              editable={!busy}
              placeholder="https://your-backend.example.com"
              placeholderTextColor={colors['fg-subtle']}
            />
            <Text className="mt-1 text-caption font-sans text-fg-subtle">
              The FedLearn backend this device signs in to.
            </Text>
          </View>
        )}

        <Text className="mt-3 text-label font-sans text-fg-muted">Username</Text>
        <TextInput
          className="mt-1 mb-3 px-3 py-2 rounded-md bg-surface-2 text-body font-sans text-fg border border-hairline"
          accessibilityLabel="Username"
          autoCapitalize="none"
          autoCorrect={false}
          value={username}
          onChangeText={(value) => { formVersion.current++; setUsername(value); }}
          editable={!busy}
        />

        <Text className="text-label font-sans text-fg-muted">Password</Text>
        <View className="mt-1 flex-row items-center rounded-md bg-surface-2 border border-hairline">
          <TextInput
            className="flex-1 px-3 py-2 text-body font-sans text-fg"
            accessibilityLabel="Password"
            secureTextEntry={!showPassword}
            value={password}
            onChangeText={(value) => { formVersion.current++; setPassword(value); }}
            editable={!busy}
          />
          <Pressable
            accessibilityRole="button"
            accessibilityLabel={showPassword ? 'Hide password' : 'Show password'}
            accessibilityState={{ selected: showPassword }}
            className="px-3 py-2 active:opacity-70"
            hitSlop={{ top: 8, bottom: 8, left: 4, right: 8 }}
            onPress={() => setShowPassword((v) => !v)}>
            {showPassword
              ? <EyeOff color={colors['fg-subtle']} size={18} strokeWidth={1.5} />
              : <Eye color={colors['fg-subtle']} size={18} strokeWidth={1.5} />}
          </Pressable>
        </View>

        {/* Save password — opt-in, encrypted at rest (Android Keystore). */}
        <Pressable
          accessibilityRole="checkbox"
          accessibilityLabel="Save password"
          accessibilityState={{ checked: savePassword }}
          className="mt-3 flex-row items-center active:opacity-80"
          hitSlop={{ top: 6, bottom: 6, left: 4, right: 8 }}
          onPress={() => setSavePassword((v) => !v)}>
          <View
            className={`w-5 h-5 rounded border items-center justify-center ${
              savePassword ? 'bg-accent border-accent' : 'bg-surface-2 border-hairline'
            }`}>
            {savePassword ? <Check color={colors['accent-fg']} size={14} strokeWidth={2.5} /> : null}
          </View>
          <Text className="ml-2 text-caption font-sans text-fg-muted">Save password</Text>
        </Pressable>

        {error && <Text className="mt-3 text-caption font-sans text-danger">{error}</Text>}
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Sign in"
          accessibilityState={{ disabled: busy }}
          className={`mt-4 py-3 rounded-md bg-accent items-center active:opacity-80 ${busy ? 'opacity-50' : ''}`}
          disabled={busy}
          onPress={onSubmit}>
          {busy
            ? <ActivityIndicator color={colors['accent-fg']} />
            : <Text className="text-label font-sans text-accent-fg">Sign in</Text>}
        </Pressable>
        <Pressable
          accessibilityRole="link"
          accessibilityLabel="Create an account"
          className="mt-3 py-2 items-center active:opacity-80"
          hitSlop={{ top: 8, bottom: 8, left: 16, right: 16 }}
          disabled={busy}
          onPress={onCreateAccount}>
          <Text className="text-caption font-sans text-fg-muted">
            No account? <Text className="text-accent">Create one</Text>
          </Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}
