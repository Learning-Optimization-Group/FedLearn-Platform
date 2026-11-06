# Preserve JWT authentication through streamed-response completion

The signed Android qualification downloads three new programs and trains successfully,
but embedded Tomcat records 500 after each stream: asynchronous completion reaches the
security chain without the JWT authentication that admitted the original request.
The response is already committed, so client status/body alone do not expose the defect.

Keep stateless authentication and every existing authorization rule. Save a successfully
validated JWT context into the request-attribute security context repository before
continuing the filter chain. Reuse that request identity during redispatch; do not
permit all asynchronous requests or persist authentication into an HTTP session.

- [x] Add a real HTTP regression using embedded Tomcat, the production security chain,
  persisted owner/CLIENT/tenant fixtures and a staged opaque contract artifact. Observe
  asynchronous dispatch completion and exceptions, as well as exact full/range bytes.
- [x] Demonstrate original-source failure for cookie and marked-Bearer streaming;
  deny anonymous, unmarked Bearer, revoked, unrelated-user and foreign-tenant requests.
  Verify a subsequent credential-free request remains denied.
- [x] Save validated authentication in request attributes; run the targeted regression,
  existing JWT/stream tests and full backend coverage gate in a separate source stage.
- [x] Review the narrow change, build a distinct jar, replace only the idle owned API,
  and verify actual full/range native-token artifact HTTP and final access status 200/206
  without committed-response errors. Preserve the prior mobile evidence and models.
- [x] Record limits and commit the fix; continue the broader seven-area program.


## Verification

Original-source regression: four ASYNC completion failures and six refusal controls
passing. The correction passed all ten cases plus sixteen existing JWT/artifact
checks. The full gate passed 1,344 tests, with two opt-in worker tests skipped;
both opt-in checks passed separately. Line coverage: 86.55%. All 1,193 staged source
hashes matched before the independent jar build. Read-only review found no
introduced issues.

The isolated idle API was replaced with the tested jar, preserving its routes and
database. Actual native-Bearer HTTPS delivered three full 200 responses and three
exact ranged 206 responses, with the same completed container statuses and no late
authorization exceptions. Anonymous and unmarked requests remained denied. Earlier
model hashes and the original 36 dataset hashes remained unchanged.

One initial client control received a gateway 502. The proxy journal reported a
malformed upstream MIME header while the API recorded a complete 200 response.
The failed attempt remains retained. Eighteen subsequent direct upstream reads,
eighteen proxied reads and the final six-response control passed. The gateway anomaly remains unexplained. The request-authentication correction
has separate regression evidence; production deployment and broader network-fault
qualification remain open. The corrected jar is retained with the saved API
environment restored and owned training/negative-HTTPS listeners closed.
