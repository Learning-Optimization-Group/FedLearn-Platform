import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { View, Text } from 'react-native';

// Import screens
import TrainingScreen from '../screens/TrainingScreen';
import ModelLibraryScreen from '../screens/ModelLibraryScreen';
import InferenceScreen from '../screens/InferenceScreen';

const Tab = createBottomTabNavigator();

// Simple text-based tab icons
const TabIcon = ({ name, focused }) => (
  <View style={{ alignItems: 'center', paddingVertical: 4 }}>
    <Text
      style={{
        fontSize: 12,
        color: focused ? '#007bff' : '#6c757d',
        fontWeight: focused ? 'bold' : 'normal',
      }}
    >
      {name}
    </Text>
  </View>
);

export default function AppNavigator() {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={{
          headerShown: true,
          headerStyle: {
            backgroundColor: '#f8f9fa',
          },
          headerTitleStyle: {
            fontWeight: 'bold',
            color: '#2c3e50',
          },
          tabBarStyle: {
            backgroundColor: '#ffffff',
            borderTopWidth: 1,
            borderTopColor: '#e9ecef',
            paddingBottom: 5,
            paddingTop: 5,
            height: 60,
          },
          tabBarActiveTintColor: '#007bff',
          tabBarInactiveTintColor: '#6c757d',
        }}
      >
        <Tab.Screen
          name="Training"
          component={TrainingScreen}
          options={{
            title: 'Federated Learning',
            tabBarIcon: ({ focused }) => (
              <TabIcon name="🤖" focused={focused} />
            ),
          }}
        />
        <Tab.Screen
          name="Library"
          component={ModelLibraryScreen}
          options={{
            title: 'Model Library',
            tabBarIcon: ({ focused }) => (
              <TabIcon name="📚" focused={focused} />
            ),
          }}
        />
        <Tab.Screen
          name="Inference"
          component={InferenceScreen}
          options={{
            title: 'Model Testing',
            tabBarIcon: ({ focused }) => (
              <TabIcon name="🔍" focused={focused} />
            ),
          }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
