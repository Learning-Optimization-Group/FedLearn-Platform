// Programmatic access to the brand tokens (for charts / inline styles that cannot use className).
//
// PLACEHOLDER mirror of tailwind.config.js. Both will be replaced by the shared @fedlearn/tokens
// OKLCH package (C5 design-system workstream) so web/desktop/mobile share ONE brand. Keep these
// in lockstep with tailwind.config.js until then. Screens prefer className; use these only where
// a raw color value is unavoidable (e.g. react-native-svg fills).
export const tokens = {
  background: 'oklch(0.98 0.004 250)',
  foreground: 'oklch(0.22 0.02 250)',
  surface: 'oklch(1 0 0)',
  surfaceMuted: 'oklch(0.96 0.006 250)',
  border: 'oklch(0.90 0.008 250)',
  muted: 'oklch(0.55 0.02 250)',
  primary: 'oklch(0.58 0.16 256)',
  primaryForeground: 'oklch(0.99 0.002 250)',
  accent: 'oklch(0.70 0.15 190)',
  success: 'oklch(0.62 0.15 150)',
  warning: 'oklch(0.78 0.16 80)',
  danger: 'oklch(0.58 0.20 27)',
} as const;

export type TokenName = keyof typeof tokens;
