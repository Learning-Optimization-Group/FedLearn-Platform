# FedLearn Platform — Federated Pneumonia Detection Demo Plan

---

## 1. Networking Strategy — Tailscale Mesh VPN

**Problem:** Traditional port-forwarding through university network infrastructure is unreliable — institutional firewalls frequently block inbound connections, router admin access is restricted, and the configuration is fragile across different physical locations.

**Solution — Tailscale:**

Tailscale was used as the networking layer for all distributed testing. Tailscale creates a **private, encrypted peer-to-peer mesh VPN** using the [WireGuard](https://www.wireguard.com/) protocol. Each participating device (Jetson, Mac, ROG) is enrolled in a shared Tailscale network ("tailnet") and assigned a stable private IP address (e.g., `100.x.x.x`).

**Practical benefits for this project:**

- **Zero network infrastructure changes required** — no router configuration, no IT department involvement, no firewall rules to request.
- **Stable addresses** — each device always has the same Tailscale IP, making gRPC client `--server_address` arguments deterministic and reproducible.
- **Encrypted by default** — all gRPC traffic between clients and server travels over WireGuard-encrypted tunnels.
- **Location-independent** — the demo works identically whether devices are on university Wi-Fi, a home network, or a mobile hotspot.
- **Scales to more devices trivially** — adding a fourth client is a single `tailscale up` command.

**Usage in this project:**

```bash
# On each device (one-time setup)
tailscale up

# Confirm each device's stable IP
tailscale ip -4

# FL client then connects using the server's Tailscale IP
python client.py --server_address 100.x.x.x:50181 --client_id hospital_a
```

---

## 2. Demo Plan — Federated Pneumonia Detection

### 2.1 The Narrative

> *"Three hospitals want to collaboratively train an AI to detect pneumonia from chest X-rays. Patient data is legally and ethically protected — no hospital can send its scans to a central server. With FedLearn, they never have to. Each hospital trains locally on its own patients. Only the learned knowledge — the model weights — is shared. The raw images never leave the device."*

This framing makes the value of federated learning **immediately intuitive** to a non-technical audience. The privacy constraint is real, recognisable, and emotionally resonant. Every technical component — the gRPC communication, the aggregation server, the real-time dashboard — maps cleanly onto something the audience already understands.

---

### 2.2 Dataset

**HuggingFace Dataset:** [`keremberke/chest-xray-classification`](https://huggingface.co/datasets/keremberke/chest-xray-classification)

| Property | Detail |
|---|---|
| Task | Binary image classification |
| Classes | `NORMAL` / `PNEUMONIA` |
| Total Images | ~5,800 (train + val + test splits included) |
| Source | Kermany et al. (Cell, 2018) — widely cited medical imaging benchmark |
| License | CC BY 4.0 |

**Loading the dataset:**

```python
from datasets import load_dataset

ds = load_dataset("keremberke/chest-xray-classification", name="full")
# Splits: ds["train"], ds["validation"], ds["test"]
```

The test split is **held only on the server** and used exclusively for global model evaluation after each round. Neither client ever sees the test data.

---

### 2.3 Device Role Assignment

| Device | Demo Role | Dataset Partition | Data Stays On Device? |
|---|---|---|---|
| **NVIDIA Jetson AGX Orin** | "Hospital A — Pediatric Ward" | Training indices 0 – 1,900 | ✅ Yes |
| **Apple M-series Mac** | "Hospital B — General Medicine" | Training indices 1,900 – 3,800 | ✅ Yes |
| **ROG Zephyrus G14 (Server)** | "Central Aggregator + Dashboard" | Test set only (no training data) | N/A |

The visual arrangement of three physically separate machines, each labelled with its hospital role, is the demo's most important prop. The audience can see that the Jetson never sends data to the laptop.

---

### 2.4 CNN Architecture

A lightweight but purposeful CNN — fast enough for live demo rounds (< 2 minutes per round on the Jetson), credible enough for a medical imaging context.

```python
from __future__ import annotations

import torch
import torch.nn as nn

class PneumoniaCNN(nn.Module):
    """Lightweight CNN for binary chest X-ray classification (Normal / Pneumonia).

    Input:  (B, 1, 224, 224)  — grayscale, normalised to [-1, 1]
    Output: (B, 2)            — logits for [NORMAL, PNEUMONIA]
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
```

**Preprocessing (identical on both clients):**

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
```

**Model size:** ~10–15 MB — well within the unary gRPC threshold. No chunked streaming required; rounds complete quickly.

---

### 2.5 Demo Flow (What the Audience Sees)

#### Phase 0 — Setup (before audience arrives)
- [ ] All three devices enrolled in Tailscale and confirmed reachable via stable IPs.
- [ ] Docker container running on Jetson; dataset partition pre-downloaded and cached locally on each device.
- [ ] Spring Boot backend and React dashboard running on the ROG. Experiment config pre-filled (5 rounds, FedAvg, 2 clients min).
- [ ] Screen layout set: dashboard visible to audience on an external monitor or projected display.

#### Phase 1 — The Problem (30 seconds)
- Point to the two client machines. *"Each of these holds patient X-rays. The data cannot leave."*
- Optional: show a quick standalone single-device training run (e.g., on the Jetson partition alone) to demonstrate limited accuracy (~65–70%). This is your **"before"** baseline.

#### Phase 2 — Start the Federated Experiment (live, ~1 minute)
- Click **"Start Server"** on the React dashboard.
  - Python gRPC server spawns. Port appears in the real-time log panel.
- Start client on the Jetson. *"Hospital A — Pediatric Ward — is now training."*
  - Dashboard shows: 1 client connected.
- Start client on the Mac. *"Hospital B — General Medicine — joins."*
  - Dashboard shows: 2 clients connected. Round 1 begins.

#### Phase 3 — Watch the Rounds (the "magic" moment, ~5–8 minutes for 5 rounds)
- Each round:
  - Both hospitals train **locally on their own images only**.
  - Only model weights travel over the network (via Tailscale-encrypted gRPC).
  - Server aggregates using FedAvg. Global model improves.
- Dashboard shows per-round accuracy climbing in real-time:

  | Round | Expected Accuracy |
  |---|---|
  | 1 | ~72% |
  | 2 | ~78% |
  | 3 | ~83% |
  | 4 | ~86% |
  | 5 | ~88–90% |

- Narrate: *"At no point have Hospital A's images left that Jetson. The server has never seen a single chest scan."*

#### Phase 4 — The Result
- Final global model evaluated on the held-out test set on the ROG server.
- Display the final accuracy metric and (optionally) a confusion matrix on the dashboard.
- Compare to the single-device baseline from Phase 1.
- Optional live inference: feed a single X-ray into the trained global model and display the `NORMAL` / `PNEUMONIA` prediction.

---

### 2.6 Pre-Demo Preparation Checklist

- [ ] **Data partitioning script** written and run on each device — saves local partitions as `.pt` files so no internet download is needed during the demo.
- [ ] **Demo experiment config** saved in the UI (5 rounds, FedAvg, 2 clients, `PneumoniaCNN`) — zero live configuration required.
- [ ] **Tailscale IPs noted** — server IP hardcoded into client launch scripts as `--server_address <tailscale_ip>:<port>`.
- [ ] **Baseline accuracy recorded** — single-device run done in advance so the "before" number is ready to cite without consuming demo time.
- [ ] **Screen recording fallback** — a pre-recorded successful run saved locally in case of unexpected network or hardware issues on the day.
- [ ] **Labelled device placards** — physical labels ("Hospital A", "Hospital B", "Aggregation Server") placed in front of each machine for the audience.

---

### The One-Sentence Demo Pitch

> *"Two hospitals train a pneumonia detector together — without sharing a single patient image — and the combined model outperforms either hospital working alone."*
