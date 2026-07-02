import React, { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Pressable, ScrollView, Text, TextInput, View } from 'react-native';
import DeviceInfo from 'react-native-device-info';
import { LogOut, Server, Cpu } from 'lucide-react-native';

import { api } from '../lib/restClient';
import { getServerBaseUrl, setServerBaseUrl } from '../lib/serverConfig';
import { maxSupportedTier, type ModelTier } from '../lib/deviceClass';
import { useAuth } from '../context/AuthContext';
import { useThemeTokens } from '../theme/useThemeTokens';

type ProbeState = 'idle' | 'probing' | 'reachable' | 'unreachable';

// Settings: point the device at a backend (parity with desktop's electron-store server config),
// see the on-device compute tier (the mobile stand-in for desktop's GPU/accelerator profile — there
// is nothing to pick, the native core selects the backend), and sign out.
export function SettingsScreen() {
  const { colors } = useThemeTokens();
  const { username, logout } = useAuth();

  const [url, setUrl] = useState('');
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [probe, setProbe] = useState<ProbeState>('idle');
  const [tier, setTier] = useState<ModelTier | null>(null);

  useEffect(() => {
    void getServerBaseUrl().then(setUrl);
    void maxSupportedTier().then(setTier);
  }, []);

  const onSave = useCallback(async () => {
    setError(null);
    setSavedMsg(null);
    setSaving(true);
    try {
      const clean = await setServerBaseUrl(url);
      setUrl(clean);
      setSavedMsg('Saved');
      // Reachability probe: /api/auth/me answers (200 or a silent 401) if the host is up. A network
      // error means the URL is wrong or the backend is down.
      setProbe('probing');
      try {
        await api.get('/api/auth/me');
        setProbe('reachable');
      } catch (e: unknown) {
        const status = (e as { response?: { status?: number } })?.response?.status;
        setProbe(status ? 'reachable' : 'unreachable');
      }
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
      setProbe('idle');
    } finally {
      setSaving(false);
    }
  }, [url]);

  const probeLabel: Record<ProbeState, string> = {
    idle: '',
    probing: 'Checking…',
    reachable: '✅ Backend reachable',
    unreachable: '⚠️ Could not reach that URL',
  };

  return (
    <ScrollView className="flex-1 bg-canvas" keyboardShouldPersistTaps="handled">
      <View className="px-4 pt-4 pb-2">
        <Text className="text-h2 font-sans text-fg">Settings</Text>
        {username ? (
          <Text className="text-caption text-fg-muted mt-1">Signed in as {username}</Text>
        ) : null}
      </View>

      {/* Backend server */}
      <View className="mx-4 mt-2 p-4 rounded-card bg-surface-1 border border-hairline">
        <View className="flex-row items-center mb-2">
          <Server color={colors.fg} size={18} strokeWidth={1.5} />
          <Text className="text-body font-sans text-fg ml-2">Backend server</Text>
        </View>
        <Text className="text-caption text-fg-muted mb-2">
          The FedLearn API this device talks to. Point it at your demo/AWS/Tailscale host.
        </Text>
        <TextInput
          className="rounded-md bg-surface-2 border border-hairline px-3 py-2 text-body font-mono text-fg"
          value={url}
          onChangeText={setUrl}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          placeholder="https://your-backend.example.com"
          placeholderTextColor={colors['fg-subtle']}
        />
        {probe !== 'idle' ? (
          <Text
            className={`text-caption mt-2 ${
              probe === 'unreachable' ? 'text-danger' : 'text-fg-muted'
            }`}>
            {probeLabel[probe]}
          </Text>
        ) : null}
        <Pressable
          className="mt-3 flex-row items-center justify-center bg-accent rounded-md py-3"
          disabled={saving}
          onPress={onSave}>
          {saving ? (
            <ActivityIndicator color={colors['accent-fg']} />
          ) : (
            <Text className="text-accent-fg text-label font-sans">
              {savedMsg ?? 'Save & test'}
            </Text>
          )}
        </Pressable>
      </View>

      {/* On-device compute */}
      <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
        <View className="flex-row items-center mb-1">
          <Cpu color={colors.fg} size={18} strokeWidth={1.5} />
          <Text className="text-body font-sans text-fg ml-2">Compute</Text>
        </View>
        <View className="mt-1 flex-row justify-between">
          <Text className="text-caption font-sans text-fg-muted">Runs on</Text>
          <Text className="text-caption font-sans text-fg">On-device (native)</Text>
        </View>
        <View className="mt-1 flex-row justify-between">
          <Text className="text-caption font-sans text-fg-muted">Max model tier</Text>
          <Text className="text-caption font-mono text-fg">{tier ?? '…'}</Text>
        </View>
        <Text className="text-caption text-fg-subtle mt-2">
          The native core picks the on-device accelerator automatically — there is no GPU profile to
          choose as on desktop.
        </Text>
      </View>

      {/* Account */}
      <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
        <Text className="text-body font-sans text-fg mb-2">Account</Text>
        <View className="flex-row justify-between mb-3">
          <Text className="text-caption font-sans text-fg-muted">App version</Text>
          <Text className="text-caption font-mono text-fg">
            {DeviceInfo.getVersion()} ({DeviceInfo.getBuildNumber()})
          </Text>
        </View>
        <Pressable
          className="flex-row items-center justify-center bg-surface-2 border border-hairline rounded-md py-3"
          onPress={() => {
            void logout();
          }}>
          <LogOut color={colors.danger} size={18} strokeWidth={1.5} />
          <Text className="text-danger text-label font-sans ml-2">Sign out</Text>
        </Pressable>
      </View>

      {error ? (
        <View className="mx-4 my-3 p-3 rounded-card bg-danger">
          <Text className="text-accent-fg text-body">{error}</Text>
        </View>
      ) : null}
      <View className="h-8" />
    </ScrollView>
  );
}
