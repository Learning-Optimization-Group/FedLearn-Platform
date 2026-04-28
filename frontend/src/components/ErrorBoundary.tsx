import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  /** Optional override for the rendered fallback UI. */
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Top-level error boundary. Without one, any render-time exception in the
 * component tree blanks the page entirely (React unmounts the whole root).
 *
 * Errors are reported to console.error so they surface in the browser devtools
 * and any monitoring agent (Sentry, Datadog RUM, etc.) that wraps it.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('[ErrorBoundary] Render-time error caught', error, info.componentStack);
  }

  private reset = (): void => {
    this.setState({ error: null });
  };

  render() {
    const { error } = this.state;
    if (!error) {
      return this.props.children;
    }

    if (this.props.fallback) {
      return this.props.fallback(error, this.reset);
    }

    return (
      <div
        role="alert"
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2rem',
          background: '#0b0b10',
          color: '#e5e7eb',
          fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
        }}
      >
        <div style={{ maxWidth: 560, width: '100%' }}>
          <h1 style={{ fontSize: '1.5rem', marginBottom: '0.75rem' }}>Something went wrong</h1>
          <p style={{ color: '#9ca3af', marginBottom: '1.5rem' }}>
            The page encountered an unexpected error and could not finish rendering.
            You can try recovering, or reload the page if the problem persists.
          </p>
          <pre
            style={{
              background: '#111827',
              padding: '0.75rem 1rem',
              borderRadius: 6,
              fontSize: '0.8rem',
              overflowX: 'auto',
              color: '#fca5a5',
              marginBottom: '1.5rem',
            }}
          >
            {error.message}
          </pre>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button
              type="button"
              onClick={this.reset}
              style={{
                padding: '0.5rem 1rem',
                borderRadius: 6,
                border: '1px solid #374151',
                background: '#1f2937',
                color: '#e5e7eb',
                cursor: 'pointer',
              }}
            >
              Try again
            </button>
            <button
              type="button"
              onClick={() => window.location.reload()}
              style={{
                padding: '0.5rem 1rem',
                borderRadius: 6,
                border: 'none',
                background: '#2563eb',
                color: 'white',
                cursor: 'pointer',
              }}
            >
              Reload page
            </button>
          </div>
        </div>
      </div>
    );
  }
}

export default ErrorBoundary;
