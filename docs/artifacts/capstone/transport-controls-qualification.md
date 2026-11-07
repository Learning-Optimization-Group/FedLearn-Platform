# Transport Controls & Failure Acceptance Qualification Report

## 1. Executive Summary

This report documents the qualification of physical transport security controls, release integrity enforcement, and failure acceptance mechanisms across mobile, runtime, and server subsystems (Areas 3 and 4 of the MSc Capstone reliability program).

All qualification tests were conducted against physical hardware (Android mobile device) and real network processes under an isolated qualification profile. Baseline environments were restored and all local dataset hashes were verified bitwise identical after test execution.

---

## 2. Release & Transport Negative Controls Matrix

The physical mobile client (Release signed APK v2/v3) was evaluated against negative network and certificate controls to verify fail-closed transport behavior:

| Test Control ID | Attack / Failure Scenario | Expected System Behavior | Observed Verification Output | Status |
|---|---|---|---|---|
| **TC-SEC-01** | **Mismatched Hostname**: Server presents valid certificate for different domain name | Client aborts TLS handshake; zero credentials or data transmitted | gRPC Error Code 14 `UNAVAILABLE`: `Peername is not in peer certificate`. 0 updates accepted. | **PASS** |
| **TC-SEC-02** | **Untrusted Root CA**: Server presents self-signed certificate untrusted by truststore | Client rejects TLS handshake; refuses communication | gRPC Error Code 14 `UNAVAILABLE`: `CERTIFICATE_VERIFY_FAILED` (`cert unknown`). 0 updates accepted. | **PASS** |
| **TC-SEC-03** | **Plaintext Refusal**: Server or client misconfigured for unencrypted HTTP/gRPC | Client enforces TLS requirement from execution contract; refuses unencrypted join | Connection refused at client gateway; zero requests dispatched to cleartext port. | **PASS** |
| **TC-SEC-04** | **Connection Token Enforcement**: Client connects without valid token or with mismatched `FEDLEARN_RUN_ID` | Server rejects update; refuses to aggregate | Server servicer logs `PERMISSION_DENIED` / `UNAUTHENTICATED`. Accepted updates count = 0. | **PASS** |
| **TC-SEC-05** | **Origin-Bound Credentials**: Backend URL updated in settings | Client isolates authorization header; old JWT not forwarded to foreign origin | Cross-origin request dispatched without prior origin Bearer token. | **PASS** |
| **TC-SEC-06** | **Same-Key App Upgrade**: In-place upgrade from v2 to v3 signed release | Application updates cleanly without uninstall; preserves private local datasets | 36 of 36 dataset hashes match before/after upgrade. Saved account and receipts retained. | **PASS** |

---

## 3. Physical Device Failure Acceptance Matrix

The failure acceptance suite validates that physical device dropouts, process terminations, and network interruptions do not corrupt federated state or produce invalid aggregates:

| Fault Scenario | Trigger Mechanism | Subsystem Response | Recovery & Terminal Invariant | Status |
|---|---|---|---|---|
| **Client Disconnect During Upload** | Client process killed or Wi-Fi dropped during chunked streaming | Server gRPC streaming detects broken pipe; discards partial chunk buffer | Incomplete update is not submitted to coordinator; accepted count is unchanged. | **PASS** |
| **Partial Cohort Dropout ($M \ge M_{\min}$)** | 1 of $N$ clients disconnects before upload; round deadline expires | Coordinator `wait_for_round_to_complete` trips after `round_timeout_s` | Coordinator force-aggregates received updates, flags `last_round_failed=True`, advances round. | **PASS** |
| **Under-Threshold Dropout ($M < M_{\min}$)** | $N - 1$ clients disconnect; received count $<$ `min_clients_for_aggregation` | Round timeout expires with insufficient updates | Coordinator aborts aggregation, requests stop (`stop_requested=True`), halts server cleanly. | **PASS** |
| **Stale Round Submission** | Client delayed by thermal throttling submits update for round $r-1$ after round $r$ starts | Coordinator checks `trained_on_round < self.current_round` | Coordinator returns `False` without aggregation; stale update discarded. | **PASS** |
| **Duplicate / Retried Update** | Client encounters transient network glitch and retries same round update | Coordinator checks `client_id` in `_client_updates_received` | Idempotent duplicate rejection: second update ignored, client weight uncorrupted. | **PASS** |
| **Server Process Exit / Restart** | FL server terminated abruptly (`SIGKILL`) mid-round | Operating system releases exclusive POSIX file lease | Server restarts via `/resume`, restores last committed generation, retrains round from model. | **PASS** |

---

## 4. Preservation & Baseline Restoration Audit

1. **Dataset Integrity**: All 36 on-device dataset partitions were hashed before test execution. Following the physical test suite, post-test SHA-256 hashes matched pre-test hashes bit-for-bit (0 mismatches).
2. **Environment Isolation**: Temporary test listeners on ports 55100 and 8444 were terminated. Baseline API configuration and certificates were restored cleanly.
3. **Model Weights Freeze**: All historical and benchmark model weights remained completely immutable during qualification.
