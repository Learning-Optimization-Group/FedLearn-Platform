import React from 'react';
import { View, ActivityIndicator } from 'react-native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { useAuth } from '../context/AuthContext';
import { AppNavigator } from './AppNavigator';
import LoginScreen from '../screens/LoginScreen';
import RegisterScreen from '../screens/RegisterScreen';
import { useThemeTokens } from '../theme/useThemeTokens';

const Stack = createNativeStackNavigator();

export default function RootNavigator() {
  const { status } = useAuth();
  const { colors } = useThemeTokens();

  if (status === 'unknown') {
    return (
      <View className="flex-1 bg-canvas items-center justify-center">
        <ActivityIndicator color={colors.accent} />
      </View>
    );
  }

  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      {status === 'authenticated'
        ? <Stack.Screen name="App" component={AppNavigator} />
        : (
          <>
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="Register" component={RegisterScreen} />
          </>
        )}
    </Stack.Navigator>
  );
}
