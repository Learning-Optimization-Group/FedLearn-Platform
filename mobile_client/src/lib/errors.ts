// MO-16: a single place to turn an unknown thrown value into a human-readable string for the UI.
// TrainingScreen used to render join errors with String(e); on an axios error that yields
// "[object Object]" (the live phone test's mystery error), hiding the real cause (e.g. a 409 whose
// backend message is "Run is not currently running (status=COMPLETED)"). readError prefers the
// backend-supplied message, then Error.message, and never returns "[object Object]".

const FALLBACK = 'Something went wrong. Please try again.';

interface MaybeAxiosError {
  response?: { data?: { message?: unknown } };
  message?: unknown;
}

function firstNonBlank(...vals: unknown[]): string | null {
  for (const v of vals) {
    if (typeof v === 'string' && v.trim().length > 0) return v;
  }
  return null;
}

/** Best-effort human-readable message for any thrown value (string, Error, axios error, object, null). */
export function readError(e: unknown): string {
  if (typeof e === 'string') return e.trim().length > 0 ? e : FALLBACK;
  const err = (e ?? {}) as MaybeAxiosError;
  const picked = firstNonBlank(err.response?.data?.message, err.message);
  if (picked) return picked;
  const s = String(e);
  return s === '[object Object]' || s === 'null' || s === 'undefined' ? FALLBACK : s;
}
