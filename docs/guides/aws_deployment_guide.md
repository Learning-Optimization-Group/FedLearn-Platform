## Step-by-Step Deployment Guide

### Scope

This guide covers a **single-EC2 demo deployment** of the Spring Boot backend
plus FL-server Python processes on one host. It is **not** the multi-replica /
ECS Fargate path:

- Spring profile: `ec2demo` (activated by the deploy script). The `production`
  profile in `application-production.properties` is the unfinished ECS Fargate
  path — do not activate it from this guide.
- Database: H2 file-mode at `~/app/data/federated_platform_db.mv.db`,
  EBS-backed, persists across reboots. PostgreSQL is opt-in via
  `SPRING_DATASOURCE_URL` (not covered here).
- FL servers: spawned as local Python processes by `FlowerServerManager`, using
  the `python3` installed on the EC2 host (no Fargate `RunTask`).
- Single replica, no horizontal scaling, no Flyway race risk.

For multi-replica with model storage on S3 and FL servers as ECS tasks, see
`AWS_AUDIT.md` Tier 2 items 10–17 — that work is not in this guide.

---

### Phase 1 — One-Time: Provision EC2 (do this in AWS Console)

1. **Launch instance:** Ubuntu 22.04, `t3.medium`, 20GB EBS
2. **Security Group inbound rules:**
   - `22/TCP` — SSH (your IP only)
   - `8081/TCP` — Spring Boot API + WebSocket (your IP only)
   - `50000-50010/TCP` — gRPC FL server ports (your IP only)
3. Download your `.pem` key file

---

### Phase 2 — One-Time: Bootstrap the EC2 Instance

```bash
export EC2_HOST=<your-ec2-public-ip>
export EC2_KEY_PATH=~/.ssh/your-key.pem

./scripts/deploy-to-aws.sh --bootstrap
```

This runs `ec2-bootstrap.sh` remotely to install Java 21 JRE, Python 3, CPU-only PyTorch, and all Python dependencies.

---

### Phase 3 — Every Deploy (code changes)

```bash
# Set these once in your shell session (or add to ~/.zshrc)
export EC2_HOST=<your-ec2-public-ip>
export EC2_KEY_PATH=~/.ssh/your-key.pem

./scripts/deploy-to-aws.sh
```

This will:

1. Build the fat JAR locally with Gradle (`-x test` to skip tests)
2. `scp` the JAR + all Python scripts to EC2
3. Print the manual SSH commands to configure and start the service

---

### Phase 4 — Configure & Start (first time only)

SSH into the instance and edit the systemd service to inject your secrets:

```bash
sudo nano /etc/systemd/system/fedlearn.service
# Uncomment + fill in the Environment= lines

sudo systemctl daemon-reload
sudo systemctl enable fedlearn  # auto-start on reboot
sudo systemctl start fedlearn
```

**For quick debugging**, skip systemd and run in the foreground — the script prints the exact `export` + `java -jar` commands for you.

---

### Phase 5 — Verify

```bash
# Health check
curl http://<EC2-IP>:8081/actuator/health

# Live logs
ssh -i ~/.ssh/your-key.pem ubuntu@<EC2-IP> 'sudo journalctl -u fedlearn -f'
```

---

### Subsequent Deploys (fast path)

```bash
# Build + upload + restart in one command:
./scripts/deploy-to-aws.sh --restart
```

Or if you only changed Python scripts (no Java changes):

```bash
./scripts/deploy-to-aws.sh --skip-build --restart
```

The two scripts are at:

- [`scripts/deploy-to-aws.sh`](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/scripts/deploy-to-aws.sh) — runs on your Mac
- [`scripts/ec2-bootstrap.sh`](file:///Users/anurag/codebase/personalProjects/FedLearn-Platform/scripts/ec2-bootstrap.sh) — runs on EC2 (automatically via `--bootstrap`)

---

### Phase 6 — Frontend & TLS

The backend on port `8081` is plain HTTP. Browsers block `http://` API calls
from any `https://` origin (mixed-content), so a SPA hosted on Vercel /
CloudFront / Netlify *cannot* talk to a bare-IP EC2 backend. You need TLS in
front of `8081`. Pick one of three paths.

#### Path A — nginx + Let's Encrypt on the same EC2 (cheapest)

Use this when you already have a DNS name pointing at the EC2 IP (DuckDNS,
your own domain, etc.). Run on the instance:

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
sudo certbot --nginx -d api.example.com   # solves ACME, writes nginx config

# /etc/nginx/sites-available/fedlearn (replace api.example.com)
server {
    listen 443 ssl http2;
    server_name api.example.com;
    ssl_certificate     /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    location / {
        proxy_pass         http://127.0.0.1:8081;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        # WebSocket upgrade for /ws-logs (STOMP)
        proxy_set_header   Upgrade           $http_upgrade;
        proxy_set_header   Connection        "upgrade";
        proxy_read_timeout 3600s;             # long-lived STOMP connection
    }
}
```

Then update the EC2 security group: open `443/TCP` to `0.0.0.0/0`, keep
`8081/TCP` locked to your IP (or remove it — nginx is the only client now).
Set `app.auth.cookie.secure=true` in `application-ec2demo.properties` once
HTTPS is live.

#### Path B — ALB + ACM (cleanest, costs ~$16/mo)

1. Request an ACM certificate for `api.example.com` (us-east-1 if fronting
   CloudFront, otherwise the region of the ALB).
2. Create an Application Load Balancer, listener on `443` with the ACM cert,
   target group pointing at the EC2 instance on `8081`.
3. Set the target-group **idle timeout to 3600s** so STOMP WebSockets don't
   get culled mid-training (default is 60s — way too short for FL rounds).
4. Point `api.example.com` (Route 53 alias) at the ALB DNS.
5. Update the EC2 SG: allow `8081/TCP` only from the ALB's SG, drop public
   `8081` access.

#### Path C — accept HTTP-only (demo / classroom only)

If you genuinely don't need HTTPS, host the SPA on the same EC2 over HTTP
(serve `frontend/dist/` from nginx on port 80). No mixed-content issue
because both origins are HTTP. Don't ship credentials over this; cookies
won't be `Secure`.

#### Build the frontend pointing at your backend

```bash
cd frontend
# Edit .env.production: replace REPLACE_WITH_YOUR_API_HOST with your TLS host.
# For Path A/B:   https://api.example.com
# For Path C:     http://<EC2-IP>:8081
npm ci
npm run build      # outputs dist/
```

Host `dist/` somewhere:
- **Vercel / Netlify**: drag-drop or `vercel --prod`. Set the project's
  environment variable to match `.env.production`.
- **S3 + CloudFront**: `aws s3 sync dist/ s3://your-bucket/` then invalidate
  the CloudFront distribution.
- **nginx on the same EC2** (Path C): `sudo cp -r dist/* /var/www/html/`.

#### Update CORS on the backend

Whatever host serves the SPA must be in `CORS_ALLOWED_ORIGINS`. Edit
`/etc/systemd/system/fedlearn.service` and replace the `localhost:5173`
default with the SPA's real origin(s):

```ini
Environment="CORS_ALLOWED_ORIGINS=https://app.example.com,https://my-app.vercel.app"
```

Then `sudo systemctl daemon-reload && sudo systemctl restart fedlearn`.
Multiple origins are comma-separated, no spaces, no trailing slash, no
wildcards. The backend fails fast on boot if this var is unset.

---

### Phase 7 — FL clients (Jetson / Mac / Zephyrus)

The FL clients connect to the **gRPC** ports (`50000-50010`), not `8081`. A
few things the deploy guide doesn't otherwise cover:

- **Security group:** the SG rule from Phase 1 only opens `50000-50010` to
  *your* IP. Each client machine on a different network needs its public IP
  added (or, for a classroom demo, `0.0.0.0/0` — but understand that gRPC
  traffic is currently `insecure_channel` in `framework/.../grpc_client.py`,
  meaning gradients fly over the WAN unencrypted. Audit item #37.)
- **Client config:** point each client at `<EC2-public-IP>:<port>`. The port
  for a given project is logged when the FL server starts — visible in the
  dashboard's log viewer or via `journalctl -u fedlearn -f` filtered for
  `Started FL server on port`.
- **NAT / idle timeout:** if you put an ALB or NLB in front of the gRPC
  ports, configure idle timeout ≥ 350s and enable HTTP/2 — otherwise long
  local-training phases will trigger `GOAWAY` mid-round. Direct EC2 IP
  exposure has no idle-timeout problem.

---

### Common pitfalls

- **First deploy missing Python deps.** Was a real bug in `deploy-to-aws.sh`
  (bootstrap ran before `requirements.txt` arrived). Fixed: requirements are
  now SCP'd in step 2/5, before bootstrap. If you're upgrading from an older
  copy of the script, re-run `--bootstrap` once after pulling the fix, or
  just `pip install -r ~/requirements.txt` on the host.
- **JVM OOM on `t3.medium`.** 4GB RAM, default JVM heap = 1GB, plus a
  forked Python+torch process per project can push past the limit. Cap the
  heap in the systemd unit:
  `Environment="JAVA_TOOL_OPTIONS=-Xmx1g"`.
- **CORS empty / boot fails.** The `ec2demo` profile requires
  `CORS_ALLOWED_ORIGINS` — Spring fails fast if unset. The deploy script's
  printed `export` line uses `http://localhost:5173`; update it to your real
  SPA origin before going to production.
- **Stray files in `~/app/scripts/`.** `scp -r scripts/*` copies
  `__pycache__/`, `debug.npz`, `fl_server_deep_debug.log`, etc. Cosmetic
  only — they don't run — but worth `rm`-ing for hygiene.
