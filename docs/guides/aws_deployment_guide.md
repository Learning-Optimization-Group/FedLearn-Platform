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

If you have never used AWS before, follow these exact steps to create your server:

1. **Log in to AWS:** Open your browser and go to the [AWS Management Console](https://console.aws.amazon.com/). Log in with your account.
2. **Navigate to EC2:** In the top search bar, type `EC2` and click on the **EC2 (Virtual Servers in the Cloud)** service.
3. **Start the Launch Wizard:** On the EC2 Dashboard, click the orange **Launch instance** button.
4. **Name your server:** Under "Name and tags", enter `FedLearn-Demo`.
5. **Choose the OS (AMI):** 
   - Under "Application and OS Images", select the **Ubuntu** logo.
   - For the Amazon Machine Image (AMI), choose **Ubuntu Server 24.04 LTS (HVM)** or **22.04 LTS**.
6. **Choose the Instance Type:** 
   - Scroll down to "Instance type" and select **`t3.medium`**. *(Note: The free tier `t2.micro` only has 1GB of RAM, which will crash during Python PyTorch processing. You must use `t3.medium` which has 4GB of RAM).*
7. **Create a Key Pair:**
   - Scroll to "Key pair (login)".
   - Click **Create new key pair**.
   - Name it `fedlearn-key`.
   - Leave the type as `RSA` and format as `.pem` (for Mac/Linux) or `.ppk` (if you use PuTTY on Windows).
   - Click **Create key pair**. Your browser will download a file (e.g., `fedlearn-key.pem`). Move this file to a safe folder on your computer (like `~/.ssh/`) and keep it secret!
8. **Configure Storage:**
   - Scroll to "Configure storage".
   - Change the size from `8 GiB` to **`20 GiB`** (PyTorch and Docker take up significant space).
9. **Configure Network and Security Groups:**
   - Scroll to "Network settings" and click the **Edit** button in that box.
   - Ensure "Auto-assign public IP" is **Enable**.
   - Under "Firewall (security groups)", select **Create security group**.
   - Name it `fedlearn-sg`.
   - **Rule 1 (SSH):** Type: SSH, Port: 22, Source type: **My IP**. *(This ensures only your current Wi-Fi can log into the server).*
   - **Click "Add security group rule"** to add another rule.
   - **Rule 2 (Backend API):** Type: Custom TCP, Port Range: **8081**, Source type: **My IP**.
   - **Click "Add security group rule"** again.
   - **Rule 3 (gRPC Federated Learning):** Type: Custom TCP, Port Range: **50000-50010**, Source type: **My IP**.
10. **Launch!** Click the orange **Launch instance** button on the bottom right.
11. **Get your IP Address:** 
    - Click on the Instance ID link (e.g., `i-0abcd1234...`) to view your new server.
    - Wait until the "Instance state" says **Running**.
    - Find the **Public IPv4 address** on the page. Copy this number. You will use this as your `<your-ec2-public-ip>` in the scripts below.

*Mac/Linux users: Make sure your key is not publicly viewable before using it. Open your local terminal and run:*
```bash
chmod 400 ~/.ssh/fedlearn-key.pem
```

---

### Phase 2 — One-Time: Bootstrap the EC2 Instance

Open your local terminal (on your Mac/PC, not the server) and run these commands to set your environment variables. Replace the IP and key path with your actual details:

```bash
# 1. Tell your terminal where your server is
export EC2_HOST=54.123.45.67  # Replace with your actual Public IPv4 address

# 2. Tell your terminal where your downloaded .pem key is
export EC2_KEY_PATH=~/.ssh/fedlearn-key.pem  # Replace with where you saved the file

# 3. Run the bootstrap script
./scripts/deploy-to-aws.sh --bootstrap
```

This runs `ec2-bootstrap.sh` remotely to install Java 21 JRE, Python 3, CPU-only PyTorch, and all Python dependencies.

---

### Phase 3 — Every Deploy (code changes)

```bash
# Set these once in your shell session (or add to ~/.zshrc)
export EC2_HOST=54.123.45.67
export EC2_KEY_PATH=~/.ssh/fedlearn-key.pem

./scripts/deploy-to-aws.sh
```

This will:

1. Build the fat JAR locally with Gradle (`-x test` to skip tests)
2. `scp` the JAR + all Python scripts to EC2
3. Print the manual SSH commands to configure and start the service

---

### Phase 4 — Configure & Start (first time only)

Now you need to log into the AWS server itself to start the Java application.

1. SSH into the instance using the command the script printed out for you, or manually type:
```bash
ssh -i ~/.ssh/fedlearn-key.pem ubuntu@$EC2_HOST
```

2. Open the system service file in a text editor (`nano`):
```bash
sudo nano /etc/systemd/system/fedlearn.service
```

3. Find the `Environment=` lines at the top. They look like this:
```ini
# Environment="CORS_ALLOWED_ORIGINS=http://localhost:5173"
# Environment="APP_JWT_SECRET=your-base64-secret-here"
# Environment="APP_INTERNAL_API_KEY=your-internal-api-key"
```

4. **Uncomment them** (delete the `#` at the start of the line) and fill in your actual values:
   - `CORS_ALLOWED_ORIGINS`: If testing locally with frontend, leave it as `http://localhost:5173`. If you have a hosted frontend on Vercel, put that URL here.
   - `APP_JWT_SECRET`: Generate a random base64 string (e.g., `dGVzdHNlY3JldGtleWZvcmp3dHRlc3RpbmcxMjM0NTY3ODk=`).
   - `APP_INTERNAL_API_KEY`: Any secure random string (e.g., `my-super-secret-key-123`).

5. Save the file and exit nano:
   - Press `Ctrl + O` to save, then `Enter` to confirm.
   - Press `Ctrl + X` to exit.

6. Start the background service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable fedlearn  # Auto-starts the app if the server reboots
sudo systemctl start fedlearn   # Starts the app right now
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
