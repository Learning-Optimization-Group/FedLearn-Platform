⏺ FedLearn Platform — AWS Readiness Audit (Consolidated)

  Bottom line: NOT ready to deploy as-is. Five concurrent agents covered backend, FL framework, frontend, desktop+docker, and infra. ~30 issues are deploy-blocking; the rest are correctness bugs and improvements.

---

  🔴 P0 — Deploy blockers (fix before any AWS launch)

  Auth & secrets

1. Hardcoded JWT secret fallback — application.properties:47 ships MTIzNDU2... as default. Any task without APP_JWT_SECRET set signs JWTs with a publicly-known key. Remove the :default fallback so Spring fails
   fast.
2. Hardcoded internal API key fallback — same file, line 69 (local_default_internal_api_key_12345). Lets anyone forge FL-server callbacks. Remove.
3. H2 console exposed by default — application.properties:19 + SecurityConfig.java:92 allow /h2-console/** for any task that didn't activate production profile. Disable + remove from permit list.
4. JWT also returned in JSON body — AuthController.java:124 defeats the HttpOnly cookie. Remove accessToken from response body.
5. JWT fallback storage in Electron uses base64, not encryption — auth.service.ts:226. On Linux without keyring, token is plaintext on disk. Refuse persistence or derive a real key.

  Authorization holes

6. GET /api/users returns the entire user table to any authenticated user — UserController.java:30. Add admin-role guard.
7. No project-ownership checks — any user can start/stop/delete any other user's project (ProjectController.java:127, ProjectService.java:243). Add ownership assertion in every service method.
8. WebSocket /ws-logs has no STOMP auth — WebSocketConfig.java:32. Anyone can subscribe to any project's logs. Add a ChannelInterceptor validating JWT on CONNECT.
9. WebSocket allowed-origins includes a personal ngrok subdomain hardcoded — same file, line 30. Drive from app.cors.allowed-origins.

  AWS / Fargate architectural blockers

10. ModelInitializer spawns Python via bash from the Java container — backend Docker image has no Python interpreter, no venv. Project creation will hit exit code 127 on Fargate. Move to a separate task or
    pre-generate at image build.
11. run_fl_server.sh / run_init_model.sh walk to $PROJECT_ROOT/venv/bin/python3 — that path only exists in your dev machine. Replace with python3 baked into the image.
12. Models written to models/`<uuid>`.npz (CWD-relative) — ProjectService.java:91. Ephemeral container storage, lost on restart, invisible to other replicas. Move to S3.
13. runningServers is an in-memory ConcurrentHashMap — FlowerServerManager.java:70. Any horizontal scale breaks status/stop semantics. Move to DB.
14. Flyway runs on every replica boot with no lock — application-production.properties:21. Concurrent migrations on >=2 tasks will race. Run as a one-shot task before service updates.
15. No server.shutdown=graceful — ECS SIGTERM kills mid-request. Add to prod profile.
16. No infrastructure-as-code at all — no Terraform/CDK, no ECS task defs, no IAM, no ALB config, no RDS config. Required before deploy.
17. No ECR push step in CI — only release-desktop.yml exists. No backend / FL-server image pipeline.

  Frontend deploy blockers

18. .env.production uses http:// (not https://) at IP 18.218.164.141:8081 → mixed-content blocks on any HTTPS origin.
19. Env-var name mismatch — axiosConfig.ts requires VITE_FEDLEARN_API_URL and throws if missing; .env.production only sets VITE_API_BASE_URL. Production build is broken.
20. RegisterPage.tsx uses raw fetch() bypassing the Axios interceptor entirely, reading the unset env var.
21. vite.config.ts: sourcemap: true in prod — ships full TS source to CDN.
22. Postgres password injected as plain env var — no Secrets Manager integration.

  Electron security

23. macOS build ships hardenedRuntime: false + identity: null — trivially injectable. Get an Apple Developer ID and notarize.
24. datasetPath not normalized before bind-mount — docker.service.ts:295. A malicious renderer can mount /etc into the container. Add path.resolve + .. rejection.

---

  🟠 P1 — Correctness bugs that will burn you

  FL framework math is broken

25. DeComFL learning-rate double-scaling — decomfl_strategy.py:174. Net effect: eta/N instead of paper's spec.
26. DeComFL client reverts its local update — decomfl_client.py:203-212. Round's training is discarded before being stored.
27. Server vs client rebuild formula diverge — decomfl_strategy.py:174 vs decomfl_client.py:109. Late-joining clients drift permanently out of sync.
28. Seed history corrupted by per-client appends — grpc_servicer.py:295. Every round adds N entries instead of 1; index lookups misaligned.
29. seed_history[server_round] off-by-one — rounds are 1-indexed but list is 0-indexed → IndexError on round 1.
30. DeComFL coordinator unpacks None — coordinator.py:262. Missing the same guard the FedAvg path has.
31. Client streaming upload missing num_examples wrapper — grpc_client.py:194 saves bare params; server-side chunks_to_parameters does data['parameters'] → KeyError.
32. Round can hang forever on client dropout — coordinator.py:41, async_coordinator.py:72. No timeout. Production training will deadlock.
33. Coordinator holds _lock during aggregation + evaluation — blocks all client RPCs for tens of seconds.
34. async_coordinator.py references pika without importing it → NameError at runtime.
35. LocalStepSeeds.seeds is int32 but server generates 31-bit values → silent overflow risk. Change proto to int64.
36. bfloat16 round-trip silently upcasts to float32 — serializer.py:64. Numpy can't represent bfloat16. Special-case it.
37. gRPC traffic is insecure_channel — gradients in clear over the public internet. Switch to ssl_channel_credentials.

  Backend bugs

38. ProjectService.java:156 — duplicate-start guard discards return value, falls through to start a 2nd process anyway.
39. FlowerServerManager 3-second Thread.sleep on Tomcat HTTP thread — every /start blocks 2s.
40. createProject not @Transactional — orphaned project rows on init failure.
41. server_logs has no FK to projects — orphaned logs survive cascade.
42. No request-body validation on RoundResultDto, StartProject, CreateProjectRequest (no @Valid, no @Size / @Min / @Max).
43. POST /api/projects/{id}/delete — should be DELETE. Method semantics matter for audit/AWS WAF rules.
44. EcsClient recreated per call — FlowerServerManager.java:110. Heavyweight, not built for this. Make it a @Bean.
45. No @PreDestroy on FlowerServerManager — child Python processes orphaned on container shutdown.
46. System.out.println of usernames + commands — PII / path leakage in logs (CustomUserDetailsService:21, FlowerServerManager:174).

  Frontend bugs

47. 401 interceptor window.location.href = '/login' — full-page reload races with React Router state. Use authError event only.
48. Log entries keyed by array index — historical logs are prepended, so React patches wrong nodes.
49. Paused log viewer drops messages instead of buffering — redesign/LogViewer.tsx:88.
50. No JWT runtime expiry watch — only checked on bootstrap.
51. No ErrorBoundary anywhere — render error = blank white screen.

---

  🔴 P0 — Deploy blockers (fix before any AWS launch)

  Auth & secrets

1. Hardcoded JWT secret fallback — application.properties:47 ships MTIzNDU2... as default. Any task without APP_JWT_SECRET set signs JWTs with a publicly-known key. Remove the :default fallback so Spring fails
   fast.
2. Hardcoded internal API key fallback — same file, line 69 (local_default_internal_api_key_12345). Lets anyone forge FL-server callbacks. Remove.
3. H2 console exposed by default — application.properties:19 + SecurityConfig.java:92 allow /h2-console/** for any task that didn't activate production profile. Disable + remove from permit list.
4. JWT also returned in JSON body — AuthController.java:124 defeats the HttpOnly cookie. Remove accessToken from response body.
5. JWT fallback storage in Electron uses base64, not encryption — auth.service.ts:226. On Linux without keyring, token is plaintext on disk. Refuse persistence or derive a real key.

  Authorization holes

6. GET /api/users returns the entire user table to any authenticated user — UserController.java:30. Add admin-role guard.
7. No project-ownership checks — any user can start/stop/delete any other user's project (ProjectController.java:127, ProjectService.java:243). Add ownership assertion in every service method.
8. WebSocket /ws-logs has no STOMP auth — WebSocketConfig.java:32. Anyone can subscribe to any project's logs. Add a ChannelInterceptor validating JWT on CONNECT.
9. WebSocket allowed-origins includes a personal ngrok subdomain hardcoded — same file, line 30. Drive from app.cors.allowed-origins.

  AWS / Fargate architectural blockers

10. ModelInitializer spawns Python via bash from the Java container — backend Docker image has no Python interpreter, no venv. Project creation will hit exit code 127 on Fargate. Move to a separate task or
    pre-generate at image build.
11. run_fl_server.sh / run_init_model.sh walk to $PROJECT_ROOT/venv/bin/python3 — that path only exists in your dev machine. Replace with python3 baked into the image.
12. Models written to models/`<uuid>`.npz (CWD-relative) — ProjectService.java:91. Ephemeral container storage, lost on restart, invisible to other replicas. Move to S3.
13. runningServers is an in-memory ConcurrentHashMap — FlowerServerManager.java:70. Any horizontal scale breaks status/stop semantics. Move to DB.
14. Flyway runs on every replica boot with no lock — application-production.properties:21. Concurrent migrations on >=2 tasks will race. Run as a one-shot task before service updates.
15. No server.shutdown=graceful — ECS SIGTERM kills mid-request. Add to prod profile.
16. No infrastructure-as-code at all — no Terraform/CDK, no ECS task defs, no IAM, no ALB config, no RDS config. Required before deploy.
17. No ECR push step in CI — only release-desktop.yml exists. No backend / FL-server image pipeline.

  Frontend deploy blockers

18. .env.production uses http:// (not https://) at IP 18.218.164.141:8081 → mixed-content blocks on any HTTPS origin.
19. Env-var name mismatch — axiosConfig.ts requires VITE_FEDLEARN_API_URL and throws if missing; .env.production only sets VITE_API_BASE_URL. Production build is broken.
20. RegisterPage.tsx uses raw fetch() bypassing the Axios interceptor entirely, reading the unset env var.
21. vite.config.ts: sourcemap: true in prod — ships full TS source to CDN.
22. Postgres password injected as plain env var — no Secrets Manager integration.

  Electron security

23. macOS build ships hardenedRuntime: false + identity: null — trivially injectable. Get an Apple Developer ID and notarize.
24. datasetPath not normalized before bind-mount — docker.service.ts:295. A malicious renderer can mount /etc into the container. Add path.resolve + .. rejection.

---

  🟠 P1 — Correctness bugs that will burn you

  FL framework math is broken

25. DeComFL learning-rate double-scaling — decomfl_strategy.py:174. Net effect: eta/N instead of paper's spec.
26. DeComFL client reverts its local update — decomfl_client.py:203-212. Round's training is discarded before being stored.
27. Server vs client rebuild formula diverge — decomfl_strategy.py:174 vs decomfl_client.py:109. Late-joining clients drift permanently out of sync.
28. Seed history corrupted by per-client appends — grpc_servicer.py:295. Every round adds N entries instead of 1; index lookups misaligned.
29. seed_history[server_round] off-by-one — rounds are 1-indexed but list is 0-indexed → IndexError on round 1.
30. DeComFL coordinator unpacks None — coordinator.py:262. Missing the same guard the FedAvg path has.
31. Client streaming upload missing num_examples wrapper — grpc_client.py:194 saves bare params; server-side chunks_to_parameters does data['parameters'] → KeyError.
32. Round can hang forever on client dropout — coordinator.py:41, async_coordinator.py:72. No timeout. Production training will deadlock.
33. Coordinator holds _lock during aggregation + evaluation — blocks all client RPCs for tens of seconds.
34. async_coordinator.py references pika without importing it → NameError at runtime.
35. LocalStepSeeds.seeds is int32 but server generates 31-bit values → silent overflow risk. Change proto to int64.
36. bfloat16 round-trip silently upcasts to float32 — serializer.py:64. Numpy can't represent bfloat16. Special-case it.
37. gRPC traffic is insecure_channel — gradients in clear over the public internet. Switch to ssl_channel_credentials.

  Backend bugs

38. ProjectService.java:156 — duplicate-start guard discards return value, falls through to start a 2nd process anyway.
39. FlowerServerManager 3-second Thread.sleep on Tomcat HTTP thread — every /start blocks 2s.
40. createProject not @Transactional — orphaned project rows on init failure.
41. server_logs has no FK to projects — orphaned logs survive cascade.
42. No request-body validation on RoundResultDto, StartProject, CreateProjectRequest (no @Valid, no @Size / @Min / @Max).
43. POST /api/projects/{id}/delete — should be DELETE. Method semantics matter for audit/AWS WAF rules.
44. EcsClient recreated per call — FlowerServerManager.java:110. Heavyweight, not built for this. Make it a @Bean.
45. No @PreDestroy on FlowerServerManager — child Python processes orphaned on container shutdown.
46. System.out.println of usernames + commands — PII / path leakage in logs (CustomUserDetailsService:21, FlowerServerManager:174).

  Frontend bugs

47. 401 interceptor window.location.href = '/login' — full-page reload races with React Router state. Use authError event only.
48. Log entries keyed by array index — historical logs are prepended, so React patches wrong nodes.
49. Paused log viewer drops messages instead of buffering — redesign/LogViewer.tsx:88.
50. No JWT runtime expiry watch — only checked on bootstrap.
51. No ErrorBoundary anywhere — render error = blank white screen.

  Desktop / Docker bugs

52. MODEL_TYPE env var injected by host but never read by entrypoint.sh — falls back to LLM default silently.
53. NUM_PARTITIONS=10 validated, but main loop hardcodes num_clients=2 — partition_id ≥ 3 → IndexError.
54. Docker log demuxer doesn't persist partial 8-byte header across chunks — emits binary garbage as log lines.
55. Dirichlet split cache on container writable layer, not /data volume — recomputed every restart.
56. 4 stray scratch files (test.js, test2.js, test3.js, test-docker.js) at fedlearn-desktop/ root — bundled into the shipped .asar. Delete them.

---

  🟡 P2 — Improvements (do soon, but won't block launch)

- Rate limiting on /api/auth/login and /register (Bucket4j).
- CSP, HSTS, X-Content-Type-Options headers — missing entirely.
- HikariCP pool not tuned in prod profile.
- ServerLogRepository.findByProjectIdOrderByTimestampAsc returns unbounded list — needs Pageable.
- Outdated AWS SDK (2.25.11, currently 2.30+).
- WebSocket needs withSockJS() + ALB stickiness + 3600s idle timeout, OR replace in-memory broker for multi-replica.
- No CI security scanning (Trivy, gitleaks, Dependabot, OWASP-DC).
- Code-signing disabled in release-desktop.yml — release notes tell users to bypass Gatekeeper. Bad habit to instill.
- Frontend bundle not code-split — recharts (~300KB) + StompJS (~90KB) in main chunk.
- withCredentials: true + Bearer-token-from-localStorage is contradictory — pick one model.
- __pycache__/ files tracked in backend/.../scripts/; .gitignore uses venv/ not **/venv/. Repo bloat.
- CorsTestController.java still in tree — debug controller, delete it.
- README still describes the legacy ProcessBuilder pattern as current architecture.
- FL server thread pool sized for 1 channel/client but dual-stub heartbeat uses 2.
- No seeding in DCFLClient → non-reproducible runs.
- DeComFL clients still call get_global_model (full streaming) instead of get_decomfl_config (seeds only) — defeats the entire bandwidth savings.

---

  Suggested fix order

1. Day 1 (security): items 1–9, 18–21, 23–24. None require architecture changes.
2. Week 1 (FL correctness): items 25–37. The DeComFL path is currently mathematically wrong; this is research-credibility-affecting.
3. Week 2 (AWS architecture): items 10–17, 22. ProcessBuilder→ECS RunTask migration is the biggest structural lift; pick S3 for model storage, write the IaC, wire CI to ECR.
4. Backlog: P1 bugs, then P2 hygiene.

  Full per-finding detail (file:line, fix sketch) is in the five sub-audit reports above this synthesis. Want me to draft fixes for any specific group — e.g., the JWT/secrets + ownership-check pass, or the DeComFL
  math correction?

✻ Worked for 1m 25s

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Showing detailed transcript · ctrl+o to toggle · ctrl+e to show all                                                                                                                                         verbose

  Desktop / Docker bugs

52. MODEL_TYPE env var injected by host but never read by entrypoint.sh — falls back to LLM default silently.
53. NUM_PARTITIONS=10 validated, but main loop hardcodes num_clients=2 — partition_id ≥ 3 → IndexError.
54. Docker log demuxer doesn't persist partial 8-byte header across chunks — emits binary garbage as log lines.
55. Dirichlet split cache on container writable layer, not /data volume — recomputed every restart.
56. 4 stray scratch files (test.js, test2.js, test3.js, test-docker.js) at fedlearn-desktop/ root — bundled into the shipped .asar. Delete them.

---

  🟡 P2 — Improvements (do soon, but won't block launch)

- Rate limiting on /api/auth/login and /register (Bucket4j).
- CSP, HSTS, X-Content-Type-Options headers — missing entirely.
- HikariCP pool not tuned in prod profile.
- ServerLogRepository.findByProjectIdOrderByTimestampAsc returns unbounded list — needs Pageable.
- Outdated AWS SDK (2.25.11, currently 2.30+).
- WebSocket needs withSockJS() + ALB stickiness + 3600s idle timeout, OR replace in-memory broker for multi-replica.
- No CI security scanning (Trivy, gitleaks, Dependabot, OWASP-DC).
- Code-signing disabled in release-desktop.yml — release notes tell users to bypass Gatekeeper. Bad habit to instill.
- Frontend bundle not code-split — recharts (~300KB) + StompJS (~90KB) in main chunk.
- withCredentials: true + Bearer-token-from-localStorage is contradictory — pick one model.
- __pycache__/ files tracked in backend/.../scripts/; .gitignore uses venv/ not **/venv/. Repo bloat.
- CorsTestController.java still in tree — debug controller, delete it.
- README still describes the legacy ProcessBuilder pattern as current architecture.
- FL server thread pool sized for 1 channel/client but dual-stub heartbeat uses 2.
- No seeding in DCFLClient → non-reproducible runs.
- DeComFL clients still call get_global_model (full streaming) instead of get_decomfl_config (seeds only) — defeats the entire bandwidth savings.x`
