import { clsx, type ClassValue } from 'clsx';
import { extendTailwindMerge } from 'tailwind-merge';

// tailwind-merge cannot tell this design system's custom text-<size> utilities
// (text-body, text-h3, …) apart from text-<color> utilities (text-fg,
// text-accent-fg, …) — by default it lumps them into one group and silently
// drops whichever comes first. Teach it the generated token scales so size and
// color merge independently.
const TEXT_SIZES = ['caption', 'label', 'body', 'body-lg', 'h4', 'h3', 'h2', 'h1'];
const COLORS = [
  'canvas', 'surface-1', 'surface-2', 'surface-3', 'code-well', 'code-fg',
  'hairline', 'line', 'fg', 'fg-muted', 'fg-subtle', 'accent', 'accent-hover',
  'accent-fg', 'success', 'warning', 'danger', 'running', 'scrim',
  ...Array.from({ length: 8 }, (_, i) => `series-${i + 1}`),
];

const twMerge = extendTailwindMerge({
  extend: {
    theme: {
      color: COLORS,
      radius: ['sm', 'md', 'lg', 'card', 'pill'],
      shadow: ['card', 'card-hover', 'overlay'],
    },
    classGroups: {
      'font-size': [{ text: TEXT_SIZES }],
    },
  },
});

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
