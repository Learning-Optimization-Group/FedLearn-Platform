import React from 'react';
import { Text, View } from 'react-native';

import { useThemeTokens } from '../theme/useThemeTokens';
import { withAlpha } from '../theme/withAlpha';

/**
 * The one error-banner style (Ledger): quiet danger tint on the card radius —
 * rounded-md, danger bg at low opacity, danger border, danger text. Margins are
 * the caller's business via `className`.
 */
export function ErrorBanner({ message, className = '' }: { message: string; className?: string }) {
  const { colors } = useThemeTokens();
  return (
    <View
      accessibilityRole="alert"
      className={`p-3 rounded-md border ${className}`}
      style={{
        backgroundColor: withAlpha(colors.danger, 0.1),
        borderColor: withAlpha(colors.danger, 0.3),
      }}>
      <Text className="text-body font-sans text-danger">{message}</Text>
    </View>
  );
}

export default ErrorBanner;
