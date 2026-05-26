import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react';
import { useToast, type ToastLevel } from '../context/ToastContext';

const LEVEL_STYLE: Record<ToastLevel, { icon: typeof CheckCircle2; accent: string }> = {
  success: { icon: CheckCircle2, accent: 'var(--accent-primary)' },
  info: { icon: Info, accent: 'var(--accent-primary)' },
  error: { icon: AlertCircle, accent: 'var(--destructive)' },
};

export function ToastStack() {
  const { toasts, dismiss } = useToast();

  if (toasts.length === 0) return null;

  return (
    <div className="fixed top-5 right-5 z-[100] flex flex-col gap-2 max-w-[360px] font-sans pointer-events-none">
      {toasts.map((t) => {
        const { icon: Icon, accent } = LEVEL_STYLE[t.level];
        return (
          <div
            key={t.id}
            role={t.level === 'error' ? 'alert' : 'status'}
            className="pointer-events-auto flex items-start gap-3 px-4 py-3 rounded-lg shadow-lg"
            style={{
              background: 'var(--background-card)',
              border: '1px solid var(--border-color)',
              boxShadow: 'var(--shadow-strong)',
              borderLeft: `3px solid ${accent}`,
            }}
          >
            <Icon className="w-5 h-5 mt-0.5 shrink-0" style={{ color: accent }} />
            <span className="flex-1 text-[13.5px] leading-snug text-(--text-primary)">
              {t.message}
            </span>
            <button
              type="button"
              onClick={() => dismiss(t.id)}
              className="shrink-0 w-6 h-6 -mr-1 -mt-1 flex items-center justify-center rounded-md transition-colors hover:bg-(--accent)"
              style={{ color: 'var(--text-secondary)' }}
              aria-label="Dismiss"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
