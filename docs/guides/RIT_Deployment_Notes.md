# RIT Lab Deployment Guide

## 1. The Architecture Strategy

**Why this setup is optimal:**

* **MacBook (Central Server):** Hosts the Spring Boot API, PostgreSQL database, and React Dashboard. The server needs very little compute power—it only aggregates incoming weights. It connects to the RIT VPN to be perfectly reachable by the lab clients.
* **RIT Lab Machines (Workers/Clients):** Machines like `glados`, `weasley`, and `granger` will execute the actual machine learning tasks. They do the computational heavy-lifting utilizing their powerful hardware.

## 2. Deploying a Client "Bare-Metal" on Glados

Because you do not have `root` access on RIT servers, you cannot use Docker. Fortunately, the `client-docker/` codebase works perfectly as standard Python scripts!

### Step 1: Transfer the Code

From your Mac terminal, securely copy the framework to the RIT machine:

```bash
scp -r client-docker/ al5150@glados.cs.rit.edu:~/client-docker/
```

### Step 2: Setup the Python Environment

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
pip install -r requirements.txt
```

### Step 3: Run the Federated Client

Ensure your MacBook is connected to the **RIT VPN(vpn.rit.edu)** to get an internal address (e.g., `129.21.x.x`).

Start a Federated Learning project from your React Dashboard on your Mac. When the API spawns the Python aggregation server, run this command on `glados` to assign it to the network:

```bash
python3 main.py \
  --server-address <YOUR_MAC_VPN_IP>:50051 \
  --project-id <PROJECT_ID_FROM_UI> \
  --partition-id 0 \
  --use-llm
```

*(Remove the `--use-llm` flag if you are testing the CNN architecture instead).*

**Pro Tip:** Repeat the login and execution process simultaneously on multiple different RIT machines (changing the `--partition-id` to 1, 2, 3...) to truly showcase your distributed parallel network!
