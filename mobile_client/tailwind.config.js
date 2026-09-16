// Tailwind / NativeWind config — derived ENTIRELY from the single token source.
//
// The semantic color/radius/spacing/type scales are built from
// `src/theme/tokens.generated.ts` (itself generated from design/tokens.json — DO NOT hand-edit).
// This file no longer hardcodes any hex/oklch; it maps semantic names → values, and for COLORS it
// maps to CSS variables so the OS color scheme drives light↔dark automatically (NativeWind v4).
//
// Dark mode mechanism (see src/theme/global.css):
//   - `darkMode: 'class'` + NativeWind v4 applies the `.dark` class from the OS scheme.
//   - Colors resolve to `var(--color-<name>)`; `:root` holds the light palette and `.dark` the dark
//     palette. One semantic class (e.g. `bg-canvas`, `text-fg`) therefore switches automatically —
//     no `dark:` duplication needed in components.
//
// Vocabulary for components (className): bg-/text-/border- + {canvas, surface-1, surface-2,
// surface-3, code-well, hairline, line, fg, fg-muted, fg-subtle, accent, accent-hover, accent-fg,
// success, warning, danger, running, series-1..8}; rounded-{sm,md,lg,card,pill}; spacing scale 1,2,3,
// 4,6,8,12,16,0.5; text-{caption,label,body,body-lg,h4,h3,h2,h1}; font-{sans,mono}.
//
// We read the single token source as TEXT and JSON.parse its object literal, rather than
// `require()`-ing the .ts file. The generated body is pure JSON-compatible data (string/number
// leaves, no trailing commas), so this needs no TS loader/jiti/babel and works in any Node the
// Tailwind/Metro toolchain runs under — the config stays a plain CommonJS module.
const fs = require('fs');
const path = require('path');

function loadTokens() {
  const src = fs.readFileSync(
    path.join(__dirname, 'src/theme/tokens.generated.ts'),
    'utf8',
  );
  const m = src.match(/export const tokens = (\{[\s\S]*?\}) as const;/);
  if (!m) {
    throw new Error('tailwind.config: could not locate `tokens` object in tokens.generated.ts');
  }
  return JSON.parse(m[1]);
}

const tokens = loadTokens();

// Semantic color names (web vocabulary). Each maps to a CSS variable set per-scheme in global.css.
// The two `series` ramps are scheme-invariant (Okabe–Ito), so they map to literal values.
const semanticColorNames = Object.keys(tokens.colorLight);
const colors = Object.fromEntries(
  semanticColorNames.map((name) => [name, `var(--color-${name})`]),
);
tokens.series.forEach((hex, i) => {
  colors[`series-${i + 1}`] = hex;
});

// radius → rounded-{sm,md,lg,card,pill}
const borderRadius = Object.fromEntries(
  Object.entries(tokens.radius).map(([k, v]) => [k, `${v}px`]),
);

// spacing → p-/m-/gap- etc. Keys mirror the token scale (1,2,3,4,6,8,12,16,0.5).
const spacing = Object.fromEntries(
  Object.entries(tokens.space).map(([k, v]) => [k, `${v}px`]),
);

// type → text-{caption,label,body,body-lg,h4,h3,h2,h1}
const fontSize = Object.fromEntries(
  Object.entries(tokens.text).map(([k, t]) => [
    k,
    [
      `${t.size}px`,
      { lineHeight: `${t.line}px`, letterSpacing: `${t.tracking}px`, fontWeight: String(t.weight) },
    ],
  ]),
);

const fontFamily = {
  sans: [tokens.font.sans],
  mono: [tokens.font.mono],
};

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{ts,tsx}', './index.js'],
  presets: [require('nativewind/preset')],
  darkMode: 'class',
  theme: {
    extend: {
      colors,
      borderRadius,
      spacing,
      fontSize,
      fontFamily,
    },
  },
  plugins: [],
};
