import React, { useState } from 'react';
import { View, Text, TextInput, Pressable, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAuth } from '../context/AuthContext';

export default function LoginScreen() {
  const { login } = useAuth();
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
          autoCapitalize="none"
          autoCorrect={false}
          value={username}
          onChangeText={setUsername}
        />
        <Text className="text-label font-sans text-fg-muted">Password</Text>
        <TextInput
          className="mt-1 px-3 py-2 rounded-md bg-surface-2 text-body font-sans text-fg border border-hairline"
          secureTextEntry
          value={password}
          onChangeText={setPassword}
        />
        {error && <Text className="mt-3 text-caption font-sans text-danger">{error}</Text>}
        <Pressable
          className="mt-4 py-3 rounded-pill bg-accent items-center"
          disabled={busy}
          onPress={onSubmit}>
          {busy
            ? <ActivityIndicator color="#FFFFFF" />
            : <Text className="text-body font-sans text-accent-fg">Sign in</Text>}
        </Pressable>
      </View>
    </SafeAreaView>
  );
}
