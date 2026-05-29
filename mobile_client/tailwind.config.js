// Tailwind / NativeWind config — the LOCAL brand-token source for the mobile app.
//
// PLACEHOLDER: this is a minimal stand-in for the shared @fedlearn/tokens OKLCH package
// (the C5 design-system workstream, README §1.1 / 02-TECH-STACK §16). When that package lands,
// replace this `colors` block with its NativeWind export so web/desktop/mobile share ONE brand.
// Colors are defined HERE (the token source) and referenced by semantic className in screens —
// so `git grep "#0" src/screens` stays empty (C5 §9 "no inline hex").
//
// NOTE (verify-before-use): OKLCH at runtime depends on the RN/NativeWind color pipeline; the
// shared token package is expected to emit RN-safe values. Pinned here in oklch() for parity
// with the web/desktop token space.
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{ts,tsx}', './index.js'],
  presets: [require('nativewind/preset')],
  theme: {
    extend: {
      colors: {
        background: 'oklch(0.98 0.004 250)',
        foreground: 'oklch(0.22 0.02 250)',
        surface: 'oklch(1 0 0)',
        'surface-muted': 'oklch(0.96 0.006 250)',
        border: 'oklch(0.90 0.008 250)',
        muted: 'oklch(0.55 0.02 250)',
        primary: 'oklch(0.58 0.16 256)',
        'primary-foreground': 'oklch(0.99 0.002 250)',
        accent: 'oklch(0.70 0.15 190)',
        success: 'oklch(0.62 0.15 150)',
        warning: 'oklch(0.78 0.16 80)',
        danger: 'oklch(0.58 0.20 27)',
      },
    },
  },
  plugins: [],
};
