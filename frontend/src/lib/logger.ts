/**
 * Tiny leveled logger that wraps the browser console.
 *
 * Why not just use `console.*` directly?
 *   1. `debug` and `info` are stripped in production builds, so we never ship
 *      verbose dev traces to end-users' devtools.
 *   2. A single chokepoint for later wiring to Sentry / Datadog RUM — replace
 *      the implementation here, not 30 call sites.
 *   3. Forces a `scope` so log lines are greppable ("[DashboardPage] foo").
 *
 * Severity guide (matches industry SLF4J / pino conventions):
 *   - error: failures the user notices or that prevent flow completion
 *   - warn:  recoverable / degraded paths (retries, fallbacks, deprecated calls)
 *   - info:  notable lifecycle events (login, logout, project created)
 *   - debug: per-render / per-event detail useful only while developing
 */

type Level = 'debug' | 'info' | 'warn' | 'error';

const isProd = import.meta.env.PROD;

function emit(level: Level, scope: string, message: string, ...rest: unknown[]): void {
    // Strip dev chatter from production bundles.
    if (isProd && (level === 'debug' || level === 'info')) {
        return;
    }
    const tag = `[${scope}]`;
    console[level](tag, message, ...rest);
}

export interface ScopedLogger {
    debug: (message: string, ...rest: unknown[]) => void;
    info: (message: string, ...rest: unknown[]) => void;
    warn: (message: string, ...rest: unknown[]) => void;
    error: (message: string, ...rest: unknown[]) => void;
}

/**
 * Create a logger bound to a scope (typically the component or module name).
 *
 *   const log = createLogger('DashboardPage');
 *   log.error('Failed to load projects', err);
 */
export function createLogger(scope: string): ScopedLogger {
    return {
        debug: (msg, ...rest) => emit('debug', scope, msg, ...rest),
        info: (msg, ...rest) => emit('info', scope, msg, ...rest),
        warn: (msg, ...rest) => emit('warn', scope, msg, ...rest),
        error: (msg, ...rest) => emit('error', scope, msg, ...rest),
    };
}
