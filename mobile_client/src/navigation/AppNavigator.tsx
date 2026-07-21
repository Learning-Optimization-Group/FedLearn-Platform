import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Boxes, Compass, House, Settings } from 'lucide-react-native';

import { HomeScreen } from '../screens/HomeScreen';
import { ProjectPickerScreen } from '../screens/ProjectPickerScreen';
import { ProjectDetailScreen } from '../screens/ProjectDetailScreen';
import { ModelLibraryScreen } from '../screens/ModelLibraryScreen';
import { ModelTestingScreen } from '../screens/ModelTestingScreen';
import { PlaygroundScreen } from '../screens/PlaygroundScreen';
import { SettingsScreen } from '../screens/SettingsScreen';
import { TrainingProvider } from '../state/TrainingContext';
import { useThemeTokens } from '../theme/useThemeTokens';
import type { AppStackParamList, MainTabParamList } from './types';

const Tab = createBottomTabNavigator<MainTabParamList>();
const AppStack = createNativeStackNavigator<AppStackParamList>();

// Four-tab bottom bar (was six) with lucide icons (NO emoji tab icons — C5 §9): Home is the
// default landing tab ("what is my phone doing"), Projects joins runs, Models is the single
// model hub, Settings is unchanged. Testing/Playground demote to stack pushes over the tabs.
// Tab-bar styles are raw values (not classNames), so they come from the active palette via
// useThemeTokens: active tab = `accent`, inactive = `fg-muted`, bar = `surface-1` over a
// hairline top border.
function MainTabs() {
  const { colors, text, font } = useThemeTokens();
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: colors.accent,
        tabBarInactiveTintColor: colors['fg-muted'],
        tabBarStyle: {
          backgroundColor: colors['surface-1'],
          borderTopWidth: 1,
          borderTopColor: colors.hairline,
        },
        // Smallest on-scale type size (caption). Never below 11 — the old hand-typed 10 was
        // off-scale.
        tabBarLabelStyle: { fontSize: text.caption.size, fontFamily: font.sans },
      }}>
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{ tabBarIcon: ({ color, size }) => <House color={color} size={size} /> }}
      />
      <Tab.Screen
        name="Projects"
        component={ProjectPickerScreen}
        options={{ tabBarIcon: ({ color, size }) => <Compass color={color} size={size} /> }}
      />
      <Tab.Screen
        name="Models"
        component={ModelLibraryScreen}
        options={{ tabBarIcon: ({ color, size }) => <Boxes color={color} size={size} /> }}
      />
      <Tab.Screen
        name="Settings"
        component={SettingsScreen}
        options={{ tabBarIcon: ({ color, size }) => <Settings color={color} size={size} /> }}
      />
    </Tab.Navigator>
  );
}

// Authenticated shell: TrainingProvider (the single owner of the shared run — STOMP stream,
// heartbeat, stop semantics, log ring) wraps a native stack whose root is the tab bar, so
// every tab AND every pushed screen reads the same run.
//
// The pushes (ModelTesting / Playground / ProjectDetail) render the native-stack header —
// standard title + back affordance from the active palette — so the screens themselves carry
// no tab-root assumptions (no h2 self-titles, no manual top insets).
export function AppNavigator() {
  const { colors, font } = useThemeTokens();
  const pushedHeader = {
    headerShown: true,
    headerStyle: { backgroundColor: colors['surface-1'] },
    headerTintColor: colors.fg,
    headerTitleStyle: { fontFamily: font.sans, color: colors.fg },
    headerShadowVisible: false,
  } as const;
  return (
    <TrainingProvider>
      <AppStack.Navigator screenOptions={{ headerShown: false }}>
        <AppStack.Screen name="MainTabs" component={MainTabs} />
        <AppStack.Screen
          name="ModelTesting"
          component={ModelTestingScreen}
          options={{ ...pushedHeader, title: 'Model testing' }}
        />
        <AppStack.Screen
          name="Playground"
          component={PlaygroundScreen}
          options={{ ...pushedHeader, title: 'Playground' }}
        />
        <AppStack.Screen
          name="ProjectDetail"
          component={ProjectDetailScreen}
          options={{ ...pushedHeader, title: 'Project' }}
        />
      </AppStack.Navigator>
    </TrainingProvider>
  );
}
