import { MoonStar, SunMedium } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

interface ThemeToggleProps {
  /** Icon-only compact variant (for headers/toolbars). Default is the labelled pill. */
  compact?: boolean;
}

export function ThemeToggle({ compact = false }: ThemeToggleProps) {
  const { isDark, toggleTheme } = useTheme();
  const label = isDark ? 'Switch to light mode' : 'Switch to dark mode';
  const Icon = isDark ? SunMedium : MoonStar;

  if (compact) {
    return (
      <button
        type="button"
        onClick={toggleTheme}
        className="w-8 h-8 inline-flex items-center justify-center rounded-lg transition-colors hover:bg-(--background-card)"
        style={{ color: 'var(--text-secondary)' }}
        aria-label={label}
        title={label}
      >
        <Icon className="h-4 w-4" />
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-all duration-200 hover:-translate-y-0.5"
      style={{
        border: '1px solid var(--border-color)',
        backgroundColor: 'var(--background-secondary)',
        color: 'var(--text-primary)',
        boxShadow: 'var(--shadow-soft)',
      }}
      aria-label={label}
      title={label}
    >
      <Icon className="h-4 w-4" />
      <span>{isDark ? 'Light' : 'Dark'} mode</span>
    </button>
  );
}
