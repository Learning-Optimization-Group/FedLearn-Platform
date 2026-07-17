import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Activity, Boxes, CirclePlay, FlaskConical, FolderOpen, Settings } from 'lucide-react-native';

import { ProjectPickerScreen } from '../screens/ProjectPickerScreen';
import { TrainingScreen } from '../screens/TrainingScreen';
import { ModelLibraryScreen } from '../screens/ModelLibraryScreen';
import { ModelTestingScreen } from '../screens/ModelTestingScreen';
import { PlaygroundScreen } from '../screens/PlaygroundScreen';
import { SettingsScreen } from '../screens/SettingsScreen';
import { useThemeTokens } from '../theme/useThemeTokens';

const Tab = createBottomTabNavigator();

// Six-tab bottom bar with lucide icons (NO emoji tab icons — C5 §9).
// Tab-bar styles are raw values (not classNames), so they come from the active palette via
// useThemeTokens: active tab = `accent`, inactive = `fg-muted`, bar = `surface-1` over a hairline
// top border. Projects is first so it's the default landing tab after login.
export function AppNavigator() {
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
        name="Projects"
        component={ProjectPickerScreen}
        options={{ tabBarIcon: ({ color, size }) => <FolderOpen color={color} size={size} /> }}
      />
      <Tab.Screen
        name="Training"
        component={TrainingScreen}
        options={{ tabBarIcon: ({ color, size }) => <Activity color={color} size={size} /> }}
      />
      <Tab.Screen
        name="Playground"
        component={PlaygroundScreen}
        options={{ tabBarIcon: ({ color, size }) => <CirclePlay color={color} size={size} /> }}
      />
      <Tab.Screen
        name="Library"
        component={ModelLibraryScreen}
        options={{ tabBarIcon: ({ color, size }) => <Boxes color={color} size={size} /> }}
      />
      <Tab.Screen
        name="Testing"
        component={ModelTestingScreen}
        options={{ tabBarIcon: ({ color, size }) => <FlaskConical color={color} size={size} /> }}
      />
      <Tab.Screen
        name="Settings"
        component={SettingsScreen}
        options={{ tabBarIcon: ({ color, size }) => <Settings color={color} size={size} /> }}
      />
    </Tab.Navigator>
  );
}
