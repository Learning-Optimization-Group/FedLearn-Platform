# Transport security (TLS) — end-to-end posture

How every hop of the platform is (or is not) encrypted on the EC2 demo
deployment, what is automated, and what still needs a real host to verify.

> This note lives in `deploy/` (not `docs/`) because the top-level `docs/`
> folder is gitignored — deploy docs need to travel with the deploy assets.

## Hop-by-hop

| # | Hop | Transport | Status |
|---|-----|-----------|--------|
| 1 | Browser → nginx `:443` | HTTPS (Let's Encrypt, TLS 1.2+, HSTS) | **Automated** by `scripts/ec2-bootstrap.sh` step 6 |
| 2 | Browser → nginx `:443` (`/ws-logs`) | WSS (same cert; nginx upgrades the connection) | **Automated** — upgrade block in `deploy/nginx/fedlearn.conf` |
| 3 | nginx → Spring Boot `127.0.0.1:8081` | Plain HTTP over loopback | Accepted — never leaves the host; keep `:8081` closed in the security group |
| 4 | Backend → spawned FL server (env/stdout) | Local process, no network | N/A |
| 5 | FL clients → FL server gRPC `:50000-50010` | **Plaintext by default** (audit #37) | **Wired, opt-in** — SE-2 fail-closed path + bootstrap-provisioned keypair; one flip away (below) |
| 6 | FL server → backend `/api/internal/**` callbacks | `FEDLEARN_BACKEND_URL` (localhost on EC2) | Loopback on the demo; use the HTTPS origin if ever split across hosts |

## Hop 1–2: HTTPS at the edge (automated)

- `deploy/nginx/fedlearn.conf` is the committed server block: `:80` serves only
  ACME challenges and 301-redirects to HTTPS; `:443` terminates TLS
  (TLS 1.2/1.3, Mozilla intermediate ciphers, HSTS `max-age=31536000`) and
  proxies to `127.0.0.1:8081`, with a dedicated `/ws-logs` location doing the
  WebSocket upgrade for STOMP log streaming. `__FEDLEARN_DOMAIN__` tokens are
  rendered by bootstrap from `FEDLEARN_DOMAIN` (default `fedlearn.duckdns.org`).
- `scripts/deploy-to-aws.sh` ships the template to the instance;
  `scripts/ec2-bootstrap.sh` step 6 installs nginx + certbot, obtains the cert
  with `certbot certonly --webroot` (two-phase: HTTP-only config first, because
  the TLS block references cert paths that don't exist before issuance), then
  installs the TLS block.
- **Idempotent**: an existing cert is never re-requested; re-running bootstrap
  after a failed issuance (DNS not pointed yet, `:80` closed) picks up where it
  left off.
- **Renewal**: the Ubuntu certbot package's `certbot.timer` runs `certbot renew`
  twice daily (bootstrap enables it explicitly); the `--deploy-hook` recorded at
  issuance reloads nginx after each renewal.
- After HTTPS is live, flip `APP_AUTH_COOKIE_SECURE=true` in the systemd unit
  (commented line is already there) so the auth cookie is marked `Secure`.

## Hop 5: FL gRPC boundary (wired, opt-in, default OFF)

The FL servers bind their own ports and bypass nginx, so the edge cert does not
cover them. The pieces (all pre-existing SE-2 hooks — nothing new invented):

- **Backend**: `app.fl.require-tls` (`APP_FL_REQUIRE_TLS=true`). When set,
  `FlowerServerManager` spawns each FL server with `FEDLEARN_GRPC_USE_TLS=1`
  **and** `FEDLEARN_REQUIRE_TLS=1`; the cert paths are inherited from the
  backend process env.
- **FL server** (`framework/src/fedlearn/security/tls.py`, `server/server.py`):
  serves `grpc.ssl_server_credentials` from `FEDLEARN_GRPC_SERVER_KEY` /
  `FEDLEARN_GRPC_SERVER_CERT`. Fail-closed: with `FEDLEARN_REQUIRE_TLS=1` it
  refuses to start rather than fall back to plaintext. Optional mTLS via
  `FEDLEARN_GRPC_ROOT_CERT` + `FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH=1`.
- **FL client** (`framework/src/fedlearn/client/grpc_client.py`): dials TLS when
  `FEDLEARN_GRPC_USE_TLS=1`, trusting `FEDLEARN_GRPC_ROOT_CERT` as the root CA.

**What bootstrap does (step 7)**: generates a self-signed RSA-2048 keypair at
`/etc/fedlearn/grpc/{server.key,server.crt}` (CN/SAN = `FEDLEARN_DOMAIN`,
825 days, key `0600` owned by the app user, never clobbered on re-run) and
writes the three `Environment=` lines into `fedlearn.service` — **commented out
by default**, or active when bootstrap runs with `FEDLEARN_GRPC_TLS=1`.

**Why default OFF**: every current client (desktop bundle, Docker, scripts)
dials `grpc.insecure_channel`-style plaintext. Because the SE-2 policy fails
closed, flipping the server side alone would make every training run refuse to
start or reject all clients. Enable it once clients are configured:

1. Server: uncomment the three lines in `/etc/systemd/system/fedlearn.service`
   (or re-run bootstrap with `FEDLEARN_GRPC_TLS=1`), then
   `sudo systemctl daemon-reload && sudo systemctl restart fedlearn`.
2. Each client: copy `/etc/fedlearn/grpc/server.crt` (public — not a secret) to
   the client machine, set `FEDLEARN_GRPC_USE_TLS=1` and
   `FEDLEARN_GRPC_ROOT_CERT=/path/to/server.crt`, and dial the server by the
   DNS name in the cert SAN (`FEDLEARN_DOMAIN`), not by raw IP — TLS hostname
   verification fails otherwise.

**Why a self-signed cert instead of reusing the Let's Encrypt one**: the LE
live dir is root-only while FL servers run as the unprivileged app user, and
LE rotates every ~90 days, which would break clients pinning the cert file. A
long-lived self-signed cert pinned as the client root CA is simpler and equally
strong for this closed federation. If public-CA trust is preferred (clients
without cert distribution), a certbot `--deploy-hook` can copy
`fullchain.pem`/`privkey.pem` into `/etc/fedlearn/grpc/` with app-user
ownership on every renewal — then clients can leave `FEDLEARN_GRPC_ROOT_CERT`
unset and use the system trust store; the trade-off is that running FL rounds
need a restart when the cert rotates.

## Alternative for hop 5: Tailscale tunnel

For cross-network demos (e.g. Jetson clients behind NAT), running the gRPC
plaintext channel **inside a Tailscale mesh** gives WireGuard encryption +
authenticated peers without touching the FL protocol: install Tailscale on the
EC2 host and each client, have clients dial the server's tailnet address, and
keep the FL ports closed to the public internet in the security group (see
`docs/guides/pneumonia_demo_plan.md`). In that topology leave
`APP_FL_REQUIRE_TLS` off — transport encryption is provided by the tunnel, and
double-encrypting adds operational cost for no threat-model gain. The gRPC TLS
path above is for federations that must traverse the open WAN.

## Verified vs. not verified

Verified by inspection/syntax on a workstation (no EC2 host in the loop):

- `bash -n` on both deploy scripts; nginx config reviewed for a valid TLS
  block (cert/key paths, TLS 1.2+, redirect, HSTS, WS upgrade).
- Existing hardening intact: secrets stay in the `0600` root-only
  `EnvironmentFile=/etc/fedlearn/secrets.env`; SSH host-key pinning unchanged.

**Not executed here — needs a real host + DNS to verify:**

- Actual certbot issuance against Let's Encrypt (DNS A record + open `:80`).
- The real TLS handshake on `:443`, the `wss://` upgrade, and HSTS in response
  headers (`curl -I https://<domain>`).
- A renewal dry run (`sudo certbot renew --dry-run`) and the nginx reload hook.
- An FL round over gRPC TLS end-to-end (`FEDLEARN_GRPC_TLS=1` server +
  configured clients), including the fail-closed refusal path.
