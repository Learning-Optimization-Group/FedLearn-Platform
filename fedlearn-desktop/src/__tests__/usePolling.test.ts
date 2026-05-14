/**
 * @jest-environment jsdom
 */
import { renderHook, act, waitFor } from '@testing-library/react';
import { usePolling } from '../renderer/hooks/usePolling';

jest.useFakeTimers();

test('runs fetcher on mount', async () => {
  const fetcher = jest.fn().mockResolvedValue([1, 2]);
  renderHook(() => usePolling(fetcher, [], 30_000));
  await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1));
});

test('runs fetcher again after interval', async () => {
  const fetcher = jest.fn().mockResolvedValue([1]);
  renderHook(() => usePolling(fetcher, [], 30_000));
  await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1));
  act(() => { jest.advanceTimersByTime(30_000); });
  await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));
});

test('refresh function triggers fetcher manually', async () => {
  const fetcher = jest.fn().mockResolvedValue([1]);
  const { result } = renderHook(() => usePolling(fetcher, [], 30_000));
  await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1));
  await act(async () => { await result.current.refresh(); });
  expect(fetcher).toHaveBeenCalledTimes(2);
});
