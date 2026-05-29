import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Activity, Boxes, FlaskConical } from 'lucide-react-native';

import { TrainingScreen } from '../screens/TrainingScreen';
import { ModelLibraryScreen } from '../screens/ModelLibraryScreen';
import { ModelTestingScreen } from '../screens/ModelTestingScreen';
import { tokens } from '../theme/tokens';

const Tab = createBottomTabNavigator();

// 3-tab bottom bar with lucide icons (NO emoji tab icons — C5 §9).
export function AppNavigator() {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: tokens.primary,
        tabBarInactiveTintColor: tokens.muted,
      }}>
      <Tab.Screen
        name="Training"
        component={TrainingScreen}
        options={{ tabBarIcon: ({ color, size }) => <Activity color={color} size={size} /> }}
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
    </Tab.Navigator>
  );
}
