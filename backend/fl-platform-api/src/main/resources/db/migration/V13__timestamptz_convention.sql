-- Restore the TIMESTAMPTZ convention for the run aggregate.
--
-- V1 established TIMESTAMP WITH TIME ZONE (timestamptz) for every point-in-time
-- column. V8 introduced the runs / run_enrollments tables with raw TIMESTAMP
-- (timestamp without time zone), which stores a naive wall-clock and silently
-- drops the offset -- a drift hazard the moment a reader and a writer disagree on
-- the session timezone. The JPA entities map these columns to java.time.Instant
-- (an absolute UTC instant), so the values already written are UTC wall-clocks;
-- reinterpreting them AT TIME ZONE 'UTC' is a lossless, order-preserving cast.
--
-- Runs on a fresh DB (columns created as TIMESTAMP by V8, retyped here) and on a
-- DB already populated at V12 (existing rows are reinterpreted, not rewritten).
--
-- Scope: the raw-TIMESTAMP columns added since V8. V9/V10 add no timestamp
-- columns; V11/V12 already use timestamptz. (Pre-V8 raw-TIMESTAMP columns from V5
-- -- organizations, organization_memberships, users.*_at, audit_events.occurred_at
-- -- are out of scope for this migration.)
--
-- CONVENTION: every future timestamp column MUST be TIMESTAMP WITH TIME ZONE
-- (timestamptz). Do not add raw TIMESTAMP columns.

ALTER TABLE runs
    ALTER COLUMN created_at TYPE timestamptz USING created_at AT TIME ZONE 'UTC';
ALTER TABLE runs
    ALTER COLUMN started_at TYPE timestamptz USING started_at AT TIME ZONE 'UTC';
ALTER TABLE runs
    ALTER COLUMN ended_at   TYPE timestamptz USING ended_at   AT TIME ZONE 'UTC';

ALTER TABLE run_enrollments
    ALTER COLUMN enrolled_at     TYPE timestamptz USING enrolled_at     AT TIME ZONE 'UTC';
ALTER TABLE run_enrollments
    ALTER COLUMN token_issued_at TYPE timestamptz USING token_issued_at AT TIME ZONE 'UTC';
