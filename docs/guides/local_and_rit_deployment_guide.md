# FedLearn Platform — Complete Local & RIT Lab Deployment Guide

> Every step to go from zero to a fully distributed FL training session, including deploying clients to RIT Lab machines (e.g., glados).

---

## 1. The Architecture Strategy

**Why this setup is optimal:**

* **MacBook (Central Server):** Hosts the Spring Boot API, PostgreSQL database, and React Dashboard. The server needs very little compute power—it only aggregates incoming weights. It connects to the RIT VPN to be perfectly reachable by the lab clients.
* **RIT Lab Machines (Workers/Clients):** Machines like `glados`, `weasley`, and `granger` will execute the actual machine learning tasks. They do the computational heavy-lifting utilizing their powerful hardware.

---

## 2. Local Setup (Mac Central Server)

### 2.1 Migrate from Maven to Gradle
```bash
cd backend/fl-platform-api

# Created build.gradle (from pom.xml dependencies)
# Created settings.gradle (with foojay-resolver for JDK 21 auto-download)

# Generate the Gradle wrapper
gradle wrapper --gradle-version 8.11
```

### 2.2 Fix Local Configuration
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

### 2.3 Fix Shell Scripts for Local Mac (No AWS venv)
**`run_init_model.sh`** and **`run_fl_server.sh`** — Commented out:
```bash
# source /home/ec2-user/app/venv/bin/activate   # <-- commented out
# deactivate                                     # <-- commented out
```

### 2.4 Install Missing Python Dependencies on Mac
```bash
pip3 install 'protobuf==6.32.1'
pip3 install flwr-datasets lz4
```

### 2.5 Start the Backend
```bash
cd backend/fl-platform-api
./gradlew clean bootRun
# Server starts on http://localhost:8081
```

---

## 3. Frontend Dashboard (Mac)

```bash
cd frontend
npm install
npm run dev
# Dashboard starts on http://localhost:5173
```

---

## 4. Connect VPN

1. Install Cisco Secure Client from `vpn03b.rit.edu`
2. Open Cisco Secure Client → Enter `vpn.rit.edu` → Connect
3. Log in with RIT credentials + Duo push
4. Find your Mac's Wi-Fi IP:
   ```bash
   ifconfig | grep 'inet '
   # Look for the 10.117.x.x address (NOT the 10.100.x.x VPN address)
   ```

---

## 5. Sync Docker Client Code

```bash
# From the project root — sync the latest framework code into client-docker
cp -r framework/src/fedlearn client-docker/
```

---

## 6. Client Setup on Glados (RIT Lab Machine)

Because you do not have `root` access on RIT servers, you cannot use Docker. Fortunately, the `client-docker/` codebase works perfectly as standard Python scripts!

### 6.1 Transfer the Code
From your Mac terminal, securely copy the framework to the RIT machine:
```bash
scp -r client-docker/ al5150@glados.cs.rit.edu:~/client-docker/
```

### 6.2 Setup the Python Environment (Bypass Docker)
Log into your SSH machine and navigate to the folder:
```bash
ssh al5150@glados.cs.rit.edu
cd ~/client-docker
```

Create and activate a virtual environment (this exactly replicates what Docker would have done inside its container):
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the Machine Learning dependencies (PyTorch, Transformers, etc.) matching the Linux architecture:
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### 6.3 Create & Start a Project (React Dashboard)
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

### 6.4 Run the Federated Client
Ensure your MacBook is connected to the **RIT VPN(vpn.rit.edu)** to get an internal address (e.g., `129.21.x.x` or `10.117.x.x`).

Start a Federated Learning project from your React Dashboard on your Mac. When the API spawns the Python aggregation server, run this command on `glados` to assign it to the network:

```bash
cd scripts

PYTHONPATH=.. python3 client.py \
  --server-address <YOUR_MAC_VPN_IP>:<PORT_FROM_STEP_3> \
  --project-id <PROJECT_ID_FROM_DASHBOARD> \
  --partition-id 0 \
  --use-llm
```

*(Remove the `--use-llm` flag if you are testing the CNN architecture instead).*

**Pro Tip:** Repeat the login and execution process simultaneously on multiple different RIT machines (changing the `--partition-id` to 1, 2, 3...) to truly showcase your distributed parallel network!

---

## 7. Key Gotchas We Discovered

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
