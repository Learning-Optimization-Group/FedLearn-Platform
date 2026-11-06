# Mobile credential origin isolation implementation plan

Execute inline in the existing isolated capstone worktree. Apply the test-first, debugging and verification workflows to each boundary; request read-only review before the final commit.

**Problem:** Saving another backend URL forwards the previous backend JWT to the new `/api/auth/me` probe. An absolute foreign-origin request also inherits the JWT. Both are reproduced with placeholder credentials through the actual Axios adapter boundary. Existing WebSocket setup independently reads an unscoped token and server URL, so it needs the same origin invariant.

**Decision:** Store origin and token together in one atomic encrypted record and require each consumer to request the credential for its captured destination. Clearing a token only on Settings save does not protect absolute requests or asynchronous consumers. Two separate storage keys can pair an old token with a new origin after a partial write. Preserve the existing session by migrating the legacy raw token only after a successful strict read of its persisted backend, before mounting authentication consumers. These are implementation decisions within the delegated scope.

**Success:** No credential crosses origins in REST or WebSocket upgrade/CONNECT, including endpoint changes while storage/login is pending. Same-origin saves and a same-key app update preserve a valid session. A failed setting write keeps the old endpoint and credential together. Malformed or unavailable storage fails closed. The original app, datasets, scientific studies and backend callback protocol remain untouched.

## Task 1: Enforce the credential boundary

Files: create `mobile_client/src/lib/backendOrigin.ts`; modify `authStore.ts`, `serverConfig.ts`, `restClient.ts`, `stompClient.ts`, `context/AuthContext.tsx`, `App.tsx` and the relevant tests.

- [x] Reproduce real forwarding with `backendCredentialIsolation.test.ts`: two failures, two controls passing, placeholder token only; actual Axios target and outbound Authorization captured without network.
- [x] Inspect installed React Native URL support before selecting a parser. Prefer a small pure origin validator over a new dependency; require HTTP(S) authorities, canonical scheme/host/default port and valid nondefault/IPv6 ports. Reject userinfo, base paths, query/fragment, controls and backslashes. Request-target parsing must handle absolute and protocol-relative URLs safely; ambiguous inputs never receive Authorization.
- [x] Add failing tests for canonical scope, malformed URLs, JSON records, same-origin token retrieval, mismatched-origin refusal, legacy migration and failed/locked storage. Use behavior checks with an in-memory encrypted-storage boundary.
- [x] Implement a single versioned JSON `{schemaVersion, origin, token}` record under the existing auth key. `getToken(origin)` and `setToken(token, origin)` require explicit scope; `clearToken(origin?)` must not erase another origin's session. Invalid records fail closed. Migration accepts legacy raw tokens only during verified startup configuration; malformed JSON is not a legacy token.
- [x] Make startup read the persisted server setting strictly (default only if absent), migrate its legacy credential, configure the client, then mount AuthProvider. On initialization failure show a plain retry action; do not probe a fallback endpoint with a legacy token.
- [x] Make the request interceptor derive the actual captured destination and attach only its bound credential. A successful real origin change signals auth loss; same-origin saves preserve the session. A failed URL persistence write does not rebind the client. A stale 401 from another origin cannot clear the new origin's session or log out its UI.
- [x] Capture the login origin before the request, reject stale responses after endpoint changes and bind storage to that original destination. Session probes must also reject results for a superseded destination. Preserve current behavior for missing tokens and ordinary login errors.
- [x] Read the WebSocket base before requesting its token; use the same captured origin for upgrade and STOMP CONNECT. Add real constructor/header-boundary tests with delayed storage/configuration changes.
- [x] Run the mobile Jest coverage gate, lint and type checks using the installed pinned G14 dependencies. Resolve failures; retain exact test counts and meaningful warnings. Request read-only review and act on concrete findings. Do not claim full token revocation or runtime execution-attempt isolation from this change.

## Task 2: Qualify the updated signed app

- [x] Synchronize and hash-check canonical mobile source in the separate G14 stage; preserve native build outputs, retained signer and original checkout.
- [x] Build qualification Release version 3 with the retained key. Verify package, signer, embedded bundle, nondebuggable status and unchanged transport policy before in-place installation.
- [x] Verify migration of the saved trusted origin/session, dataset and receipts after a cold launch. Recheck the 36 original-app dataset hashes. Exercise a deliberate server-origin change with disposable credentials and return to the trusted endpoint through normal login.
- [x] Run the physical untrusted-HTTPS control using a fresh correct-host self-signed certificate and a bounded private-interface listener. Use a signed-out session and fake credentials; prove reachability separately and zero HTTP credential requests from the refused phone handshake. Do not install trust roots or weaken hostname/TLS checks. Restore the saved trusted origin/account and remove only owned listener/helpers.
- [x] Retain evidence and limits, update the two-line progress source, commit the verified implementation on the isolated branch, and continue the seven-area program.

## Review refinements and native transport evidence

The first parser slice fixes the eight invalid-authority cases and canonical same-origin save; 11 of 14 initial regressions pass while the two token-forwarding cases and cookie override remain red. Expanded real-Axios regressions cover registration chaining, pending login after backend change/logout, per-request baseURL/protocol-relative targets and old 401 responses after replacement sessions. The native ArtifactService regression reproduces a shared placeholder cookie crossing same-host ports in a real MockWebServer/Robolectric test. No real credentials are used.

Include saved passwords and the LoginScreen form in the same boundary: bind encrypted opt-in credentials to origin; clear the displayed password when the form moves to another origin; ignore delayed saved-password loads for a superseded form. Capture registration and its follow-up login under one admission scope. Serialize credential writes and conditional deletes; attach a monotonic session revision to outgoing requests so a stale 401 cannot delete a replacement login, including A→B→A. One in-memory origin/epoch must drive all transports after initialization; do not reread fallback configuration independently for STOMP.

Native cookies require an explicit boundary. Cookies are shared across host ports ([RFC 6265](https://www.rfc-editor.org/rfc/rfc6265.html#section-8.5)); an origin-bound Bearer alone cannot isolate native cookie authentication. Installed RN0.80 Android `NetworkingModule.kt` uses `NO_COOKIES` when Axios explicitly forwards `withCredentials=false`; force false per request. Its WebSocket module separately reads shared cookies. Clear legacy cookies at startup and before publishing an origin change through the installed `RCTNetworking.clearCookies(callback)` wrapper. Callback false is a successful empty store; a missing callback must time out and offer startup retry rather than hang.

Change Android ArtifactService to use no ambient cookie jar, request-specific scoped Bearer/native-client headers and no automatic redirects. Validate the captured origin against the actual artifact URL using native HttpUrl; do not store tokens in singleton client defaults. Preserve hashing, size, resume, quarantine and verified-cache behavior. Update the JS bridge wrapper to supply only a credential authorized for that captured destination. Qualify an uncached artifact on the real signed app, since an existing verified cache bypasses networking. Add same-host/different-port native request/redirect fixtures. Backend/browser cookie behavior remains unchanged.

Android source and phone evidence do not qualify iOS: its RN redirect handler reloads shared cookies independently of the initial request flag. Keep this platform limit explicit until its transport behavior is separately verified.

Implementation layout: `backendOrigin.ts` pure validation; `backendScope.ts` authoritative runtime origin/epoch and authentication admission guards; a small shared serialized credential-mutation helper; versioned atomic auth and saved-credential records; `nativeCookies.ts` bounded platform callback wrapper; strict server startup/switching and AuthProvider gating; transport consumers use captured scope and session revisions. Origin changes synchronously invalidate authenticated UI without an unconditional second token deletion. Explicit logout invalidates pending admissions before awaiting storage. Compare current scope/admission before and after asynchronous writes; newer-origin writes must not be erased by old callbacks.


## Verified implementation and retained limits

All 749 mobile tests and 72 Android unit tests passed. TypeScript, lint and the
same-key signed version-3 build passed; all 426 staged source hashes matched.
The physical update retained a valid legacy session, four receipts and the
32-example imported dataset. A new eight-example synthetic vector snapshot was
imported normally and completed one phone-only round; its registered final model
matched the accepted update bit for bit. Earlier models and all 36 original-app
dataset hashes were preserved. The correct-host untrusted-HTTPS control produced
two certificate refusals and zero phone HTTP requests. Trusted sign-in was restored;
the temporary listener was stopped, helpers removed and the API baseline verified.

The fresh native downloads exposed a separate backend defect: successful streamed
bytes were followed by failed asynchronous authorization and final container status
500. Preserve that failure and address request-scoped security-context persistence
next. Do not count the current backend as having clean stream completion. iOS,
production distribution identity/backup and immediate runtime token revocation
remain outside this qualification. The generic login error does not identify TLS
failures, and durable logout under storage-removal failure remains unqualified.
