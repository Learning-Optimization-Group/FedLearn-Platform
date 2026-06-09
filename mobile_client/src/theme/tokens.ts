// RETIRED hand-authored mirror — now a thin re-export of the single token source.
//
// The brand is owned by design/tokens.json → src/theme/tokens.generated.ts. This module exists only
// so older `import { tokens } from '../theme/tokens'` paths keep resolving. New code should import
// from './tokens.generated' directly, or use semantic NativeWind classes; for raw inline/SVG values
// that must follow the active OS scheme, use `useThemeTokens()`.
export { tokens, default } from './tokens.generated';
export type { ColorToken } from './tokens.generated';
