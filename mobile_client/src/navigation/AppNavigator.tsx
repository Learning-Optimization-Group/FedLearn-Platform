import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Activity, Boxes, FlaskConical, FolderOpen, Settings, Wand2 } from 'lucide-react-native';

import { ProjectPickerScreen } from '../screens/ProjectPickerScreen';
import { TrainingScreen } from '../screens/TrainingScreen';
import { ModelLibraryScreen } from '../screens/ModelLibraryScreen';
import { ModelTestingScreen } from '../screens/ModelTestingScreen';
import { PlaygroundScreen } from '../screens/PlaygroundScreen';
import { SettingsScreen } from '../screens/SettingsScreen';
import { useThemeTokens } from '../theme/useThemeTokens';

const Tab = createBottomTabNavigator();

// 4-tab bottom bar with lucide icons (NO emoji tab icons — C5 §9).
// Tint colors are raw values (not classNames), so they come from the active palette: active tab =
// `accent`, inactive = `fg-muted`. status table: running→accent, idle→fg-muted.
// Projects is first so it's the default landing tab after login.
export function AppNavigator() {
  const { colors } = useThemeTokens();
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: colors.accent,
        tabBarInactiveTintColor: colors['fg-muted'],
        // Six tabs — keep labels compact so they don't truncate on narrow devices.
        tabBarLabelStyle: { fontSize: 10 },
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
        options={{ tabBarIcon: ({ color, size }) => <Wand2 color={color} size={size} /> }}
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
