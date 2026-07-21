# UI and Components

> **Current design system: Ledger** — navy structural ink (`#1C314D`) on quiet paper
> surfaces (`#F6F3EE` canvas, white cards), one type family (Hanken Grotesk; JetBrains
> Mono for logs/ids), 4px-grid spacing, quiet card shadows, and a light-first theme.
> Ledger superseded **Ember** (AMOLED black + burnt orange, 2026-06), which had
> superseded **Instrument**. Tokens are generated — see below — and the dark palette
> (a navy-dark family, not pure black) stays wired for a future deliberate dark mode.

## Design tokens are generated, not hand-written

`design/tokens.json` is the single source of truth. `node design/build-tokens.mjs`
regenerates the per-platform artifacts:

| Artifact | Consumer |
|---|---|
| `frontend/src/styles/tokens.css` | Tailwind v4 `@theme` (semantic utilities: `bg-canvas`, `text-fg`, `border-hairline`, `bg-accent`, `shadow-card`, `text-h3`, …) |
| `fedlearn-desktop/src/renderer/tokens.css` | plain CSS custom properties incl. the `--text-*` type ramp |
| `mobile_client/src/theme/tokens.generated.ts` + `global.css` | typed token object + NativeWind semantic classes |

Never hardcode a color, radius, shadow, or font in a component — consume the semantic
tokens. Palette changes are made once in `tokens.json` and propagate to all three
surfaces.

**Gotcha — custom class merging:** `src/lib/utils.ts` extends `tailwind-merge` with
the token scales so custom text *sizes* (`text-body`) and text *colors*
(`text-accent-fg`) merge independently. Without that config, stock tailwind-merge
treats them as one group and silently drops one — keep the config in sync when adding
token names.

## Component conventions (the rules the UI follows)

- **One primary action per view** — the navy fill. Secondary = white surface +
  hairline border; ghost = borderless; danger = solid destructive fill (a destructive
  confirm must never read weaker than Cancel).
- **Dialogs** — everything goes through `ui/Modal`: scrim backdrop, overlay shadow,
  Escape + one close affordance, footer slot with right-aligned natural-width
  `[Cancel] [Primary]`. `ConfirmDialog` for confirmations (`danger` prop for
  destructive ones).
- **Forms** — every control sits in `ui/FormField` (label association via
  `htmlFor`/`id`, help/error slots). Specific verb labels ("Create project", not
  "Submit"); loading state renders inside the submit button.
- **Page shell** — every routed view renders `PageHeader` (h3 title, subtitle,
  right-aligned actions) and a `max-w-[1400px]` content container. Live-connection
  state is a quiet dot+caption chip in the header actions, never part of the title.
- **Status** — `ui/StatusPill` is the only status vocabulary (no emoji, no unicode
  glyphs, never repurposed for non-status data).
- **Stats** — `ui/StatGroup` (one card, hairline-divided) instead of card-per-number
  grids.
- **Micro-labels** — `ui/SectionLabel` is the one uppercase label style.
- **Empty states** — neutral lucide icon in a muted circle + title + one-line body.
  The brand mark is a logo, not illustration.
- **Icons** — `lucide-react` only.
- **Charts** — series colors come from the `--color-series-N` tokens (a
  colorblind-validated categorical ramp, fixed assignment order); one axis per chart
  (no dual-axis), legend for ≥2 series.
- **Focus** — every interactive element has a visible `focus-visible` ring (2px
  accent, offset 2).

## Primitives (`src/components/ui/`)

`Button` (primary/secondary/ghost/danger × sm/md/lg) · `Card` · `Modal` ·
`ConfirmDialog` · `Input` · `Select` · `StatusPill` · `MetricTile` · `StatGroup` ·
`SectionLabel` · `FormField` · `LogConsole` (recessed mono well) · `Skeleton`.

## App structure

Views live in `src/components/redesign/` (the `redesign/` path is historical — it is
the only UI; the old `/v2` routes redirect to the canonical paths). `LayoutV2`
composes `Sidebar` + routed view; auth pages live in `src/pages/`. `ErrorBoundary`
wraps the app; `DiskLoader` is the standard async spinner. Brand assets
(`favicon.svg`, icons, `og-image.png`) carry the flat navy network mark and are
regenerated alongside palette changes.

## Responsive design

Layouts reflow with Tailwind breakpoint prefixes (`md:flex`, `lg:grid-cols-3`);
tables scroll horizontally inside their cards on narrow viewports.
