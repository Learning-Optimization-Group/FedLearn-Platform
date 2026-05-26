import { createContext, useCallback, useContext, useMemo, useState, ReactNode } from 'react';

/**
 * Client-side action feedback (success/error/info toasts).
 *
 * Distinct from NotificationContext, which surfaces server-pushed STOMP
 * notifications in the bell icon. This one is for ephemeral "the thing you
 * just clicked worked" / "the thing you just clicked failed" toasts.
 */

export type ToastLevel = 'success' | 'error' | 'info';

export interface Toast {
  id: number;
  level: ToastLevel;
  message: string;
}

interface ToastContextValue {
  toasts: Toast[];
  show: (level: ToastLevel, message: string) => void;
  dismiss: (id: number) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

/** Default auto-dismiss in ms. Errors stay slightly longer than successes. */
const AUTODISMISS_MS: Record<ToastLevel, number> = {
  success: 3000,
  info: 4000,
  error: 5000,
};

let nextId = 1;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const show = useCallback(
    (level: ToastLevel, message: string) => {
      const id = nextId++;
      setToasts((prev) => [...prev, { id, level, message }]);
      window.setTimeout(() => dismiss(id), AUTODISMISS_MS[level]);
    },
    [dismiss]
  );

  const value = useMemo<ToastContextValue>(
    () => ({ toasts, show, dismiss }),
    [toasts, show, dismiss]
  );

  return <ToastContext.Provider value={value}>{children}</ToastContext.Provider>;
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
}
