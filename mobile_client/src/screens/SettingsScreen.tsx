import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Platform, Pressable, ScrollView, Text, TextInput, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import DeviceInfo from 'react-native-device-info';
import { LogOut, Server, Cpu, Smartphone } from 'lucide-react-native';

import { api } from '../lib/restClient';
import { getServerBaseUrl, setServerBaseUrl } from '../lib/serverConfig';
import { maxSupportedTier, type ModelTier } from '../lib/deviceClass';
import { useAuth } from '../context/AuthContext';
import { ErrorBanner } from '../components/ErrorBanner';
import { StatusBadge } from '../components/StatusBadge';
import { useThemeTokens } from '../theme/useThemeTokens';

type ProbeState = 'idle' | 'probing' | 'reachable' | 'unreachable';

// Settings: point the device at a backend (parity with desktop's electron-store server config),
// see the on-device compute tier (the mobile stand-in for desktop's GPU/accelerator profile — there
// is nothing to pick, the native core selects the backend), and sign out.
export function SettingsScreen() {
  const { colors } = useThemeTokens();
  const insets = useSafeAreaInsets(); // top safe area on the root; bottom belongs to the tab bar
  const { username, logout } = useAuth();

  const [url, setUrl] = useState('');
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [probe, setProbe] = useState<ProbeState>('idle');
  const [tier, setTier] = useState<ModelTier | null>(null);
  // The saved confirmation is a transient caption under the field (the button label never mutates).
  const savedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    void getServerBaseUrl().then(setUrl);
    void maxSupportedTier().then(setTier);
    return () => {
      if (savedTimer.current) clearTimeout(savedTimer.current);
    };
  }, []);

  const onSave = useCallback(async () => {
    setError(null);
    setSavedMsg(null);
    setSaving(true);
    try {
      const clean = await setServerBaseUrl(url);
      setUrl(clean);
      setSavedMsg('Saved');
      if (savedTimer.current) clearTimeout(savedTimer.current);
      savedTimer.current = setTimeout(() => setSavedMsg(null), 2500);
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

  // Probe result renders through the shared status vocabulary (StatusBadge), not ad-hoc glyphs.
  const probeBadge: Record<Exclude<ProbeState, 'idle'>, { label: string; variant: 'running' | 'success' | 'danger' }> = {
    probing: { label: 'Checking…', variant: 'running' },
    reachable: { label: 'Backend reachable', variant: 'success' },
    unreachable: { label: 'Could not reach that URL', variant: 'danger' },
  };

  return (
    <ScrollView
      className="flex-1 bg-canvas"
      style={{ paddingTop: insets.top }}
      keyboardShouldPersistTaps="handled">
      <View className="px-4 pt-4 pb-2">
        <Text className="text-h2 font-sans text-fg">Settings</Text>
        {username ? (
          <Text className="text-caption font-sans text-fg-muted mt-1">Signed in as {username}</Text>
        ) : null}
      </View>

      {/* This device — what hardware/app the rest of the screens talk about. All values come
          from the existing device-info source (react-native-device-info sync getters). */}
      <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
        <View className="flex-row items-center mb-1">
          <Smartphone color={colors.fg} size={18} strokeWidth={1.5} />
          <Text className="text-label font-sans font-semibold text-fg ml-2">This device</Text>
        </View>
        <View className="mt-1 flex-row justify-between">
          <Text className="text-caption font-sans text-fg-muted">Platform</Text>
          <Text className="text-caption font-sans text-fg">
            {Platform.OS === 'ios' ? 'iOS' : 'Android'} {DeviceInfo.getSystemVersion()}
          </Text>
        </View>
        <View className="mt-1 flex-row justify-between">
          <Text className="text-caption font-sans text-fg-muted">Model</Text>
          <Text className="text-caption font-sans text-fg">{DeviceInfo.getModel()}</Text>
        </View>
        <View className="mt-1 flex-row justify-between">
          <Text className="text-caption font-sans text-fg-muted">App version</Text>
          <Text className="text-caption font-mono text-fg">
            {DeviceInfo.getVersion()} ({DeviceInfo.getBuildNumber()})
          </Text>
        </View>
      </View>

      {/* Backend server */}
      <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
        <View className="flex-row items-center mb-2">
          <Server color={colors.fg} size={18} strokeWidth={1.5} />
          <Text className="text-label font-sans font-semibold text-fg ml-2">Backend server</Text>
        </View>
        <Text className="text-caption font-sans text-fg-muted mb-2">
          The FedLearn API this device talks to. Point it at your demo/AWS/Tailscale host.
        </Text>
        <TextInput
          className="rounded-md bg-surface-2 border border-hairline px-3 py-2 text-body font-mono text-fg"
          accessibilityLabel="Backend server URL"
          value={url}
          onChangeText={setUrl}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          placeholder="https://your-backend.example.com"
          placeholderTextColor={colors['fg-subtle']}
        />
        {savedMsg ? (
          <Text className="text-caption font-sans text-success mt-2">Saved</Text>
        ) : null}
        {probe !== 'idle' ? (
          <View className="mt-2 self-start">
            <StatusBadge label={probeBadge[probe].label} variant={probeBadge[probe].variant} />
          </View>
        ) : null}
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Save and test"
          accessibilityState={{ disabled: saving }}
          className={`mt-3 flex-row items-center justify-center bg-accent rounded-md py-3 active:opacity-80 ${
            saving ? 'opacity-50' : ''
          }`}
          disabled={saving}
          onPress={onSave}>
          {saving ? (
            <ActivityIndicator color={colors['accent-fg']} />
          ) : (
            <Text className="text-accent-fg text-label font-sans">Save & test</Text>
          )}
        </Pressable>
      </View>

      {/* On-device compute */}
      <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
        <View className="flex-row items-center mb-1">
          <Cpu color={colors.fg} size={18} strokeWidth={1.5} />
          <Text className="text-label font-sans font-semibold text-fg ml-2">Compute</Text>
        </View>
        <View className="mt-1 flex-row justify-between">
          <Text className="text-caption font-sans text-fg-muted">Runs on</Text>
          <Text className="text-caption font-sans text-fg">On-device (native)</Text>
        </View>
        <View className="mt-1 flex-row justify-between">
          <Text className="text-caption font-sans text-fg-muted">Max model tier</Text>
          <Text className="text-caption font-mono text-fg">{tier ?? '…'}</Text>
        </View>
        <Text className="text-caption font-sans text-fg-subtle mt-2">
          The native core picks the on-device accelerator automatically — there is no GPU profile to
          choose as on desktop.
        </Text>
      </View>

      {/* Account (the app-version row moved into the This-device card above) */}
      <View className="mx-4 mt-3 p-4 rounded-card bg-surface-1 border border-hairline">
        <Text className="text-label font-sans font-semibold text-fg mb-2">Account</Text>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Sign out"
          className="flex-row items-center justify-center bg-surface-1 border border-hairline rounded-md py-3 active:opacity-80"
          onPress={() => {
            void logout();
          }}>
          <LogOut color={colors.danger} size={18} strokeWidth={1.5} />
          <Text className="text-danger text-label font-sans ml-2">Sign out</Text>
        </Pressable>
      </View>

      {error ? <ErrorBanner message={error} className="mx-4 my-3" /> : null}
      <View className="h-8" />
    </ScrollView>
  );
}
