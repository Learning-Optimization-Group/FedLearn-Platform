// Tint helper for token-derived translucent fills. RN styles can't use CSS
// color-mix()/opacity modifiers on var()-backed NativeWind colors, so the rare
// tinted-background surfaces (status badges, error/warning banners) derive their
// fill from the active palette at runtime instead of hardcoding a color.
export function withAlpha(color: string, alpha: number): string {
  const m = /^#([0-9a-fA-F]{6})$/.exec(color);
  const hex = m?.[1];
  if (!hex) return color;
  const r = parseInt(hex.slice(0, 2), 16);
  const g = parseInt(hex.slice(2, 4), 16);
  const b = parseInt(hex.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export default withAlpha;
