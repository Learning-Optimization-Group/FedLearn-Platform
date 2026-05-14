import { useEffect, useRef, useState, useCallback } from 'react';

interface PollingState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function usePolling<T>(
  fetcher: () => Promise<T>,
  deps: React.DependencyList,
  intervalMs: number = 30_000,
): PollingState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const run = useCallback(async () => {
    try {
      const result = await fetcherRef.current();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    run();
    const id = window.setInterval(run, intervalMs);
    const onFocus = () => run();
    window.addEventListener('focus', onFocus);
    return () => {
      window.clearInterval(id);
      window.removeEventListener('focus', onFocus);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, ...deps]);

  return { data, loading, error, refresh: run };
}
