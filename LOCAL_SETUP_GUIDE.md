# FedLearn Platform — Complete Local Setup Guide

> Every command we ran to go from zero to a fully distributed FL training session.

---

## Phase 1: Backend Server (Mac)

### 1.1 Migrate from Maven to Gradle
```bash
cd backend/fl-platform-api

# Created build.gradle (from pom.xml dependencies)
# Created settings.gradle (with foojay-resolver for JDK 21 auto-download)

# Generate the Gradle wrapper
gradle wrapper --gradle-version 8.11
```

### 1.2 Fix Local Configuration
**`application.properties`** — Updated these values:
```properties
# Changed port from 1234 to default PostgreSQL port
spring.datasource.url=${SPRING_DATASOURCE_URL:jdbc:postgresql://localhost:5432/federated_platform_db}
spring.datasource.password=${SPRING_DATASOURCE_PASSWORD:Coloreal@1}

# Changed .bat scripts to .sh for Mac
python.executable.path=${PYTHON_EXECUTABLE_PATH:src/main/resources/scripts/run_init_model.sh}
python.flbat.path=${PYTHON_EXECUTABLE_PATH:src/main/resources/scripts/run_fl_server.sh}
python.script.fl-server.path=${PYTHON_SCRIPT_FL_SERVER_PATH:src/main/resources/scripts/run_fl_server.sh}
```

### 1.3 Fix Shell Scripts for Local Mac (No AWS venv)
**`run_init_model.sh`** and **`run_fl_server.sh`** — Commented out:
```bash
# source /home/ec2-user/app/venv/bin/activate   # <-- commented out
# deactivate                                     # <-- commented out
```

### 1.4 Install Missing Python Dependencies on Mac
```bash
pip3 install 'protobuf==6.32.1'
pip3 install flwr-datasets lz4
```

### 1.5 Start the Backend
```bash
cd backend/fl-platform-api
./gradlew clean bootRun
# Server starts on http://localhost:8081
```

---

## Phase 2: Frontend Dashboard (Mac)

```bash
cd frontend
npm install
npm run dev
# Dashboard starts on http://localhost:5173
```

---

## Phase 3: Sync Docker Client Code

```bash
# From the project root — sync the latest framework code into client-docker
cp -r framework/src/fedlearn client-docker/
```

---

## Phase 4: Create & Start a Project (React Dashboard)

1. Open `http://localhost:5173` in browser
2. Sign up / Log in (`al5150`)
3. Create a new project:
   - Project Name: `Demo2`
   - Model Architecture: `CNN`
   - Model Name: `net`
   - Optimizer: `Adam`
   - Pre-train Epochs: `0`
4. Click **"Toggle Server"** to start the FL server
5. Watch backend terminal for:
   ```
   gRPC server started and listening on 0.0.0.0:<PORT>
   ```
6. Note the **port number** (e.g., `61386`) — it's dynamically assigned each time

---

## Phase 5: Connect VPN

1. Install Cisco Secure Client from `vpn03b.rit.edu`
2. Open Cisco Secure Client → Enter `vpn.rit.edu` → Connect
3. Log in with RIT credentials + Duo push
4. Find your Mac's Wi-Fi IP:
   ```bash
   ifconfig | grep 'inet '
   # Look for the 10.117.x.x address (NOT the 10.100.x.x VPN address)
   ```

---

## Phase 6: Client Setup on Glados (RIT Lab Machine)

### 6.1 Transfer the Code
```bash
# From your Mac
scp -r client-docker/ al5150@glados.cs.rit.edu:~/codebase/federatedLearning/client-docker/
```

### 6.2 Setup Python Environment (Bypass Docker)
```bash
ssh al5150@glados.cs.rit.edu
cd ~/codebase/federatedLearning/client-docker

python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 6.3 Run the Federated Client
```bash
cd scripts

PYTHONPATH=.. python3 client.py \
  --server-address 10.117.51.89:<PORT_FROM_STEP_4> \
  --project-id <PROJECT_ID_FROM_DASHBOARD> \
  --partition-id 0
```

---

## Key Gotchas We Discovered

| Issue | Root Cause | Fix |
|---|---|---|
| `Connection refused` on startup | Wrong PostgreSQL port (1234 vs 5432) | Update `application.properties` |
| `model_path = null` in database | `run_init_model.sh` crashed on `deactivate` | Comment out `deactivate` |
| `protobuf gencode 6.x runtime 5.x` | System Python had old protobuf | `pip3 install protobuf==6.32.1` |
| `No module 'flwr_datasets'` | Not installed on Mac system Python | `pip3 install flwr-datasets` |
| `No module 'fedlearn'` on glados | Python couldn't find parent package | Prefix with `PYTHONPATH=..` |
| `No module 'torchvision'` on glados | PyTorch commented out in requirements.txt | `pip install torch torchvision torchaudio` |
| VPN IP refused connections | RIT VPN blocks inbound traffic | Use Wi-Fi IP (`10.117.x.x`) instead |
| `.bat` scripts on Mac | Windows scripts configured by default | Switch to `.sh` in `application.properties` |
| AWS venv path on Mac | Scripts hardcoded to `/home/ec2-user/...` | Comment out `source` and `deactivate` |
