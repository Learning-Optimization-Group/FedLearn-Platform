# Mobile credentials follow their issuing server

The Android client binds its session token and opt-in saved password to the exact
HTTP(S) origin that issued them. Changing the scheme, host or port signs the client
out. Saving the same canonical origin preserves the session. Absolute request
URLs cannot borrow another server's token, and changing the login form's server
clears the old username and password.

Startup loads the saved server before mounting authentication, migrates legacy
sign-in records against that saved origin, and clears legacy native cookies.
Unreadable settings stop startup with retry and signed-out recovery controls;
recovery clears sign-in details while preserving datasets. Pending logins,
registration follow-up requests and stale 401 responses cannot replace or erase a
newer session. WebSocket upgrade and CONNECT share one captured origin and token.

Android artifact downloads use request-specific scoped Bearer authentication,
without a shared cookie jar or automatic redirects. Artifact size, SHA-256,
resumption, quarantine and verified-cache checks still apply. REST explicitly
turns off native cookies, and an origin switch clears legacy WebSocket cookies
before publishing its new destination. Backend browser-cookie authentication is
unchanged.

## Verification on 6 November

The mobile suite passed 749 tests; the complete Android unit suite passed 72.
TypeScript and lint passed. Regressions reproduced token forwarding, saved-password
reuse, stale authentication callbacks, delayed WebSocket construction and native
cross-port cookie forwarding before their fixes.

A separately packaged, nondebuggable signed qualification APK updated in place
from version 2 to 3 using the same retained test identity. Its valid legacy session,
four contribution receipts and imported 32-example image dataset survived. All 36
original-app dataset file hashes remained unchanged. The new embedded JavaScript
was present; the Android transport XML and native training library matched the
previous APK byte for byte.

The updated phone imported eight newly generated synthetic vector examples through
Android's file picker and completed one TinyNet FedAvg round. Three new native
program downloads preceded the accepted update. This exposed a separate API
defect: asynchronous response completion lost authentication after streaming the
bytes, producing authorization errors and container access status 500. The completed
phone round does not qualify clean backend streaming completion; that defect needs
its own correction and regression coverage.

A correct-host self-signed HTTPS listener passed a host-side positive control with
explicit test trust. The signed phone rejected its certificate on both the
reachability probe and disposable-credential sign-in, and sent zero HTTP requests
to that listener. Old sign-in fields were empty after the origin switch. The app
currently reports a generic sign-in failure rather than identifying TLS failure.
No trust root or transport restriction was changed.

These results qualify this Android boundary. iOS cookie and redirect behavior,
production distribution identity and backup, production-profile API deployment,
and immediate runtime connection-token revocation remain unqualified. Storage
removal failure during logout still has only in-process revocation guarantees.
