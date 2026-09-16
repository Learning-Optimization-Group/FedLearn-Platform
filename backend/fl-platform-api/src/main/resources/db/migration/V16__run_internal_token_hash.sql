-- Re-adopted FL servers (BA-3) keep the per-run internal token they were handed at spawn, but the
-- RunTokenRegistry is in-memory and empty after a backend restart, so a survivor's result/benchmark
-- callbacks would 401. Persist the SHA-256 hash of that token (never the plaintext) so the
-- StartupReconciler can rehydrate the registry for exactly the runs it re-adopts as live. Reaped runs
-- are never rehydrated, so their tokens stay dead. 64 hex chars = a 256-bit digest.
ALTER TABLE runs ADD COLUMN internal_token_hash VARCHAR(64);
