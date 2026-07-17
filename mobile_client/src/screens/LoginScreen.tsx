import React, { useState } from 'react';
import { View, Text, TextInput, Pressable, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { useAuth } from '../context/AuthContext';
import { useThemeTokens } from '../theme/useThemeTokens';

export default function LoginScreen() {
  const navigation = useNavigation<{ navigate: (screen: string) => void }>();
  const { login } = useAuth();
  const { colors } = useThemeTokens();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async () => {
    setBusy(true); setError(null);
    try { await login(username.trim(), password); }
    catch { setError('Sign-in failed. Check your username and password.'); }
    finally { setBusy(false); }
  };

  return (
    <SafeAreaView className="flex-1 bg-canvas">
      <View className="px-4 pt-4">
        <Text className="text-h2 font-sans text-fg">Sign in</Text>
      </View>
      <View className="mx-4 mt-4 p-4 rounded-card bg-surface-1 border border-hairline">
        <Text className="text-label font-sans text-fg-muted">Username</Text>
        <TextInput
          className="mt-1 mb-3 px-3 py-2 rounded-md bg-surface-2 text-body font-sans text-fg border border-hairline"
          accessibilityLabel="Username"
          autoCapitalize="none"
          autoCorrect={false}
          value={username}
          onChangeText={setUsername}
        />
        <Text className="text-label font-sans text-fg-muted">Password</Text>
        <TextInput
          className="mt-1 px-3 py-2 rounded-md bg-surface-2 text-body font-sans text-fg border border-hairline"
          accessibilityLabel="Password"
          secureTextEntry
          value={password}
          onChangeText={setPassword}
        />
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
          onPress={() => navigation.navigate('Register')}>
          <Text className="text-caption font-sans text-fg-muted">
            No account? <Text className="text-accent">Create one</Text>
          </Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}
