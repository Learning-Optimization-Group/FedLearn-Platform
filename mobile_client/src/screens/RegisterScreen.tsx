import React, { useState } from 'react';
import { View, Text, TextInput, Pressable, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { useAuth } from '../context/AuthContext';

/** Sign-up screen: registers a new account (POST /api/auth/register) and auto-signs-in on success.
 *  New accounts get the default USER platform role — they can join and train, and request owner
 *  promotion later. */
export default function RegisterScreen() {
  const navigation = useNavigation();
  const { register } = useAuth();
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async () => {
    const u = username.trim();
    const e = email.trim();
    if (u.length < 3) { setError('Username must be at least 3 characters.'); return; }
    if (!/^\S+@\S+\.\S+$/.test(e)) { setError('Enter a valid email address.'); return; }
    if (password.length < 6) { setError('Password must be at least 6 characters.'); return; }
    setBusy(true); setError(null);
    try {
      await register(u, e, password);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { message?: string } } })?.response?.data?.message;
      setError(msg ?? 'Sign-up failed. That username or email may already be taken.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <SafeAreaView className="flex-1 bg-canvas">
      <View className="px-4 pt-4">
        <Text className="text-h2 font-sans text-fg">Create account</Text>
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
        <Text className="text-label font-sans text-fg-muted">Email</Text>
        <TextInput
          className="mt-1 mb-3 px-3 py-2 rounded-md bg-surface-2 text-body font-sans text-fg border border-hairline"
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="email-address"
          value={email}
          onChangeText={setEmail}
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
            : <Text className="text-body font-sans text-accent-fg">Create account</Text>}
        </Pressable>
        <Pressable className="mt-3 items-center" disabled={busy} onPress={() => navigation.goBack()}>
          <Text className="text-caption font-sans text-fg-muted">
            Already have an account? <Text className="text-accent">Sign in</Text>
          </Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}
