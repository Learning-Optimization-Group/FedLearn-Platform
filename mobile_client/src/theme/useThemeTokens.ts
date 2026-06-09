// Active-palette accessor for the rare cases that cannot use a NativeWind className —
// react-native-svg fills/strokes, charting libs, and navigator tint colors that take a raw value.
//
// Components should style via semantic classes (bg-canvas, text-fg, border-hairline, …) wherever
// possible; reach for this hook ONLY for inline/SVG/raw-value props. It returns the same semantic
// vocabulary as the classes, resolved for the current OS color scheme.
import { useColorScheme } from 'nativewind';

import { tokens } from './tokens.generated';

export type Palette = typeof tokens.colorLight;

/**
 * Returns the active semantic palette (light or dark) following the OS color scheme — the same
 * source NativeWind uses to drive the `.dark` class, so inline values stay in lockstep with classes.
 * `series` (scheme-invariant Okabe–Ito ramp) and `radius`/`space`/`text`/`font` are also exposed.
 */
export function useThemeTokens() {
  const { colorScheme } = useColorScheme();
  const colors: Palette = colorScheme === 'dark' ? tokens.colorDark : tokens.colorLight;
  return {
    colorScheme: colorScheme ?? 'light',
    colors,
    series: tokens.series,
    radius: tokens.radius,
    space: tokens.space,
    text: tokens.text,
    font: tokens.font,
  };
}

export default useThemeTokens;
