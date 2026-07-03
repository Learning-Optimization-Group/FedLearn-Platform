import { vi } from 'vitest';
import type { useAuth } from '../context/AuthContext';

export type AuthValue = ReturnType<typeof useAuth>;

/** A complete AuthContext value with anonymous defaults; override per test. */
export function makeAuth(over: Partial<AuthValue> = {}): AuthValue {
  return {
    currentUser: null,
    isLoading: false,
    isAdmin: false,
    isOwner: false,
    setSession: vi.fn(),
    logout: vi.fn(),
    ...over,
  };
}
