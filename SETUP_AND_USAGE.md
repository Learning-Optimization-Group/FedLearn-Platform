# FedLearn Platform — Setup and Usage Guide

This guide covers everything needed to install the FedMob Android app, deploy model files to the phone, run federated training over WiFi, and test trained models on-device.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [App Installation](#2-app-installation)
3. [Model Files — Export and Deploy](#3-model-files--export-and-deploy)
4. [Run Federated Training](#4-run-federated-training)
5. [Model Library and On-Device Testing](#5-model-library-and-on-device-testing)
6. [Architecture Reference](#6-architecture-reference)
7. [Troubleshooting](#7-troubleshooting)
8. [Platform Support](#8-platform-support)

---

## 1. Prerequisites

### Mac (Development Machine)

| Requirement | Version / Notes |
|---|---|
| Xcode Command Line Tools | `xcode-select --install` |
| Android Studio | Latest stable |
| Android NDK | **27.1.12297006** (must match — set in `android/build.gradle`) |
| Android SDK Platform | API 34+ |
| Node.js | 18+ |
| Python | 3.9+ |
| ADB | Comes with Android SDK (`~/Library/Android/sdk/platform-tools/`) |

**Python Virtual Environment (macOS/Linux):**

```bash
cd "/path/to/FedLearn-Platform"
python3 -m venv venv
source venv/bin/activate
pip install -r framework/requirements.txt
```

### Windows (Development Machine)

| Requirement | Version / Notes |
|---|---|
| Android Studio | Latest stable |
| Android NDK | **27.1.12297006** (SDK Manager → SDK Tools → NDK) |
| Android SDK Platform | API 34+ |
| Node.js | 18+ |
| Python | 3.9+ |
| ADB | Comes with Android SDK (`%LOCALAPPDATA%\Android\Sdk\platform-tools\`) |

**Python Virtual Environment (PowerShell):**

```powershell
cd "C:\path\to\FedLearn-Platform"
python -m venv venv
venv\Scripts\activate
pip install -r framework\requirements.txt
```

**Find WiFi IP (Windows):**

```powershell
ipconfig
# Look for "IPv4 Address" under your Wi-Fi adapter
```

All `adb` commands in this guide work identically on Windows. Use `python` instead of `python3` if needed. Replace forward slashes in file paths with backslashes.

### Phone

- Android 10+ (API 29+)
- USB Debugging enabled: Settings → Developer Options → USB Debugging
- Connected via USB **or** on the same WiFi network as your Mac

---

## 2. App Installation

The APK contains the native C++ TurboModule (`libfedlearn_core.so`) compiled for ARM64 with PyTorch Mobile (libtorch) and gRPC. No model files are bundled in the APK — they are deployed separately (see Section 3).

### Build and Install (USB required, one-time)

```bash
# Connect phone via USB and authorize the debugging prompt on your phone
adb devices          # should show your device as "device" (not "unauthorized")

cd mobile_client

# Start Metro bundler (keep this running in a separate terminal)
npx react-native start --reset-cache

# In a second terminal: build and install the APK
npx react-native run-android
```

This compiles the C++ native library, packages the APK, and installs it directly to the phone. After this step, the phone no longer needs to be on USB for daily use.

> **Important — re-push model after every `run-android`**: Each `npx react-native run-android` reinstalls the APK and wipes the app's private storage, deleting any previously pushed `.pt` model files. Always re-push the model after a reinstall before launching the app (see Section 3).

### Verify Installation

```bash
adb shell pm list packages | grep mobileclient
# Expected: package:com.mobileclientnew
```

---

## 3. Model Files — Export and Deploy

| Model | Architecture | Parameters | `.pt` file size | Port |
|---|---|---|---|---|
| 1M | 4-layer CNN | ~1.05M | ~4 MB | 50062 |
| 10M | ResNet-style (6 blocks) | ~11M | ~42 MB | 50063 |
| 100M | Wide MLP | ~93M | ~355 MB | 50064 |

Each model is deployed independently. Run only the block for the model you want to test.

---

### 1M Parameter Model

```bash
# 1. Export TorchScript (Mac, run once)
cd "/path/to/FedLearn-Platform/mobile_client"
source ../venv/bin/activate
python scripts/export_model_1m.py          # → assets/model_1m.pt (~4 MB)

# 2. Push to phone (USB connected)
adb push assets/model_1m.pt /data/local/tmp/model_1m.pt
adb shell run-as com.mobileclientnew cp /data/local/tmp/model_1m.pt files/model_1m.pt

# 3. Select in app — edit this line in mobile_client/src/utils/nativeModelPath.js:
#    const MODEL_FILE_NAME = 'model_1m.pt';

# 4. Reload the app (no APK rebuild needed)
adb shell am broadcast -a com.facebook.react.RELOAD
```

---

### 10M Parameter Model

```bash
# 1. Export TorchScript (Mac, run once)
cd "/path/to/FedLearn-Platform/mobile_client"
source ../venv/bin/activate
python scripts/export_model_10m.py         # → assets/model_10m.pt (~42 MB)

# 2. Push to phone (USB connected)
adb push assets/model_10m.pt /data/local/tmp/model_10m.pt
adb shell run-as com.mobileclientnew cp /data/local/tmp/model_10m.pt files/model_10m.pt

# 3. Select in app — edit this line in mobile_client/src/utils/nativeModelPath.js:
#    const MODEL_FILE_NAME = 'model_10m.pt';

# 4. Reload the app (no APK rebuild needed)
adb shell am broadcast -a com.facebook.react.RELOAD
```

---

### 100M Parameter Model

```bash
# 1. Export TorchScript (Mac, run once)
cd "/path/to/FedLearn-Platform/mobile_client"
source ../venv/bin/activate
python scripts/export_model_100m.py        # → assets/model_100m.pt (~355 MB)

# 2. Push to phone (USB connected — push may take several minutes)
adb push assets/model_100m.pt /data/local/tmp/model_100m.pt
adb shell run-as com.mobileclientnew cp /data/local/tmp/model_100m.pt files/model_100m.pt

# 3. Select in app — edit this line in mobile_client/src/utils/nativeModelPath.js:
#    const MODEL_FILE_NAME = 'model_100m.pt';

# 4. Reload the app (no APK rebuild needed)
adb shell am broadcast -a com.facebook.react.RELOAD
```

---

### Verify files on the phone (any model)

```bash
adb shell run-as com.mobileclientnew ls -lh files/
# Expected output:
# -rw------- 1 u0_a123 u0_a123  4.1M 2026-02-23 model_1m.pt
# -rw------- 1 u0_a123 u0_a123   42M 2026-02-23 model_10m.pt
# -rw------- 1 u0_a123 u0_a123  355M 2026-02-23 model_100m.pt
```

Android blocks direct writes to an app's private directory. The push uses `/data/local/tmp` (world-writable) as a staging area, then `run-as` copies the file into the app's private storage.

---

## 4. Run Federated Training

Both clients train on the **MNIST handwritten digit dataset**:
- **Python client**: real MNIST images (60,000 training samples, partitioned by client ID)
- **Phone client**: synthetic MNIST-shaped tensors (10 samples/round, kept minimal to avoid OOM on-device)

Only **FedAvg** is currently supported end-to-end. The ZO-FL and DeComFL buttons in the app are disabled — the `simple_federation` server implements FedAvg only.

```
[ Mac ]                                      [ Phone ]
run_server.py  <──── gRPC (WiFi/USB) ────>   FedMob App (C++ TurboModule)
run_client.py  <──── gRPC (localhost) ────>  (Python client, runs on Mac)
```

The server waits for `--min_clients` clients (default: 2) before aggregating and advancing each round.

---

### Step 1: Find your Mac's WiFi IP

```bash
ipconfig getifaddr en0
# Example: 192.168.1.50
```

### Step 2: Start the FL server  [Terminal 1]

```bash
cd "/path/to/FedLearn-Platform/framework/examples/simple_federation"
source ../../../venv/bin/activate

# 1M model (2 clients: desktop + mobile)
python run_server.py --model 1m --port 50062 --num_rounds 5 --min_clients 2

# 10M model (2 clients: desktop + mobile)
python run_server.py --model 10m --port 50063 --num_rounds 5 --min_clients 2

# 100M model (2 clients: desktop + mobile)
python run_server.py --model 100m --port 50064 --num_rounds 5 --min_clients 2

# To test with only the desktop client (no phone needed):
# python run_server.py --model 10m --port 50063 --num_rounds 5 --min_clients 1
```

The server binds to `0.0.0.0` and is reachable from both the phone (over WiFi) and the Mac Python client.

### Step 3: Start the Python client  [Terminal 2]

The Python client runs on the **same Mac** as the server — use `localhost`, not the WiFi IP. The WiFi IP is only needed by the phone because it is a separate device on the network.

```bash
cd "/path/to/FedLearn-Platform/framework/examples/simple_federation"
source ../../../venv/bin/activate

# 1M model
python run_client.py --model 1m --server_address localhost:50062 --id 0 --max_samples 500

# 10M model
python run_client.py --model 10m --server_address localhost:50063 --id 0 --max_samples 200

# 100M model
python run_client.py --model 100m --server_address localhost:50064 --id 0 --max_samples 100
```

`--max_samples` limits the training set size per round for faster iteration. Remove it to use the full MNIST partition.

| Client | Address to use |
|---|---|
| Python client (Terminal on Mac) | `localhost:<port>` |
| Phone app (over WiFi) | `<Mac WiFi IP>:<port>` e.g. `192.168.1.50:50063` |
| Phone app (over USB tunnel) | `localhost:<port>` after `adb reverse tcp:<port> tcp:<port>` |

### Step 4: Connect the phone

**Over WiFi** (phone and Mac on same network):
1. Open the FedMob app → **Training** tab
2. Set server address to your Mac's IP and the matching port (e.g. `192.168.1.50:50063`)
3. Tap **Connect (gRPC)** — the app will verify the connection and register with the server. Status should show "Connected". If connection fails, the app will show the native error in the logs section and in the Metro terminal.

**Over USB** (USB connected, using port tunnel):
```bash
adb reverse tcp:50063 tcp:50063   # repeat with the port matching your model
```
Then set server address in the app to `localhost:50063`.

### Step 5: Start training on the phone

Tap **Start FedAvg (SGD)** in the Training tab. The server now has 2 clients (Python + phone) and Round 1 begins.

Keep the app in the foreground — Android kills background processes.

### Step 6: Expected results

**1M model, 5 rounds (~3–4 min total)**

| Round | Phone train | Upload | Server accuracy |
|---|---|---|---|
| 1 | ~25 sec | ~15 sec (4MB) | ~11% |
| 2 | ~25 sec | ~15 sec | ~22–28% |
| 3 | ~25 sec | ~15 sec | ~30–35% |
| 4 | ~25 sec | ~15 sec | ~34–38% |
| 5 | ~25 sec | ~15 sec | ~36–40% |

**10M model, 5 rounds (~20 min total)**

| Round | Phone train | Upload | Server accuracy |
|---|---|---|---|
| 1 | ~3–4 min | ~90 sec (42MB) | ~11% |
| 2 | ~3–4 min | ~90 sec | ~18–25% |
| 3 | ~3–4 min | ~90 sec | ~28–33% |
| 4 | ~3–4 min | ~90 sec | ~32–36% |
| 5 | ~3–4 min | ~90 sec | ~36–40% |

After each completed round, the model entry in the Library tab updates with the latest accuracy and loss.

---

## 5. Model Library and On-Device Testing

### Model Library (Library Tab)

After the model loads at startup, a baseline entry is automatically written to the Library. After each completed FL round, the entry is updated with the latest accuracy and loss.

The Library stores JSON metadata files in the app's private documents folder:
```
/data/user/0/com.mobileclientnew/files/model_trained_model_10m.json
```

To view the Library: tap the **Library** tab in the app. Pull down to refresh the list.

Each entry shows:
- Model name and round number
- Accuracy (if training has been done)
- File size (parameter count × 4 bytes)
- Date saved

### Running Model Testing (Model Testing Tab)

1. In the **Library** tab, tap a model entry
2. Tap **"Test Model"** — this navigates to the Model Testing tab and reloads the `.pt` file into C++
3. In the **Model Testing** tab:
   - MNIST test images are displayed (20 samples, 2 per digit class, loaded from bundled JSON)
   - Use **← Previous** / **Next →** to navigate images
   - Tap **Run Inference** to get a model fitness score

### Inference result interpretation

The inference runs a forward pass on an internal synthetic data batch (C++ layer) and returns a loss value:

| Inference Loss | Color | Meaning |
|---|---|---|
| < 1.0 | Green | Model is well-trained |
| 1.0 – 2.0 | Yellow | Model is partially trained |
| > 2.0 | Red | Model needs more training rounds |

The per-class bar chart shows a proxy confidence score (`exp(-loss)`) with the ground-truth label's bar highlighted in green.

> **Note**: The displayed MNIST digit image is for visual reference. The actual inference score reflects the model's quality on its internal synthetic data batch, not the specific pixel values shown. A full per-image prediction API would require a native `predict(imageData)` method added to the C++ TurboModule.

---

## 6. Architecture Reference

### Component Map

```
FedLearn-Platform/
├── framework/                        # Python FL framework
│   ├── src/fedlearn/
│   │   ├── server/strategy.py        # FedAvg strategy
│   │   ├── server/decomfl_strategy.py# DeComFL strategy
│   │   └── client/decomfl_client.py  # DeComFL client
│   └── examples/simple_federation/
│       ├── run_server.py             # Server launcher (FedAvg only)
│       ├── run_client.py             # Python client
│       ├── model_1m.py               # ~1M param CNN
│       ├── model_10m.py              # ~10M param ResNet
│       └── model_100m.py             # ~93M param MLP
│
└── mobile_client/                    # React Native + C++ app
    ├── scripts/
    │   ├── export_model_1m.py        # TorchScript export
    │   ├── export_model_10m.py
    │   └── export_model_100m.py
    ├── assets/                       # Exported .pt files go here
    ├── src/
    │   ├── screens/
    │   │   ├── TrainingScreen.jsx    # FL training UI
    │   │   ├── ModelLibraryScreen.jsx# Saved model list
    │   │   └── InferenceScreen.jsx   # Model testing UI
    │   ├── services/
    │   │   └── PlatformStorageService.js  # JSON metadata storage
    │   └── utils/
    │       └── nativeModelPath.js    # Model file path resolver
    └── shared/src/                   # C++ TurboModule
        ├── NativeFedLearnCore.cpp    # JS-to-C++ bridge
        ├── ModelManager.cpp          # TorchScript load/train
        ├── FederatedLoop.cpp         # FedAvg / DeComFL loop
        └── FedLearnClient.cpp        # gRPC client
```

### JS ↔ C++ Method Map

| JavaScript call | C++ function | What it does |
|---|---|---|
| `NativeFedLearnCore.loadModel(path)` | `ModelManager::loadScriptModel()` | Loads `.pt` via `torch::jit::load` |
| `NativeFedLearnCore.getModelInfo()` | Returns JSON `{numParams, sizeBytes}` | Reports model size |
| `NativeFedLearnCore.connect(addr, id)` | Creates `FedLearnClient`, waits for channel, registers | Validates connection and registers with server |
| `NativeFedLearnCore.startFedAvgLoop(config)` | `FederatedLoop::fedAvgLoop()` in background thread | Runs FL training loop |
| `NativeFedLearnCore.getStatus()` | Returns JSON `{phase, round, loss, accuracy}` | Polled every 2s by JS |
| `NativeFedLearnCore.getRecentLogs()` | Returns JSON array of native log entries | Drains C++ log buffer for Metro display |
| `NativeFedLearnCore.stopTraining()` | `FederatedLoop::stop()` | Stops background thread |
| `NativeFedLearnCore.trainStep(path, epochs, lr)` | `ModelManager::trainStep()` | Local train step (demo / inference) |

---

## 7. Troubleshooting

### Server stuck on round 1 / "Server is still in round X. Waiting..."

The server requires `--min_clients` updates per round before aggregating. If one client submits but the other never does, the round never completes.

**Common causes:**
- The phone never connected or its connection silently failed. Check Metro logs for `[Native]` prefixed messages showing gRPC activity.
- Only one client is running. Either start a second client or lower `--min_clients 1`.

### Phone shows "Connection failed" when tapping Connect

- Verify the server is running and the port matches.
- Verify the phone and Mac are on the **same WiFi network**.
- Run `ipconfig getifaddr en0` (Mac) or `ipconfig` (Windows) to confirm the correct IP.
- Check that no firewall is blocking the gRPC port.
- If using USB, run `adb reverse tcp:<port> tcp:<port>` and use `localhost:<port>`.

### No native logs visible in Metro terminal

Native C++ logs (gRPC registration, model download, training, upload) are forwarded to Metro via `getRecentLogs()` polling. They appear as `[Native] [FedLearnClient] ...` entries in Metro and in the app's Logs section during active training.

If you do not see them, the polling only runs while the status poller is active (after tapping Start FedAvg). For connection issues, logs are drained immediately after `connect()`.

For raw native logs outside of the JS bridge, use:

```bash
adb logcat -s FedLearnClient:I FederatedLoop:I NativeFedLearnCore:I FedLearnNative:I
```

### Port already in use (EADDRINUSE)

Kill existing processes on the port:

```bash
lsof -ti:<port> | xargs kill -9
```

---

## 8. Platform Support

| Platform | Dev Machine | Mobile Target | Status |
|---|---|---|---|
| macOS | Full support | Android arm64 | Tested and working |
| Windows | Supported (see Section 1) | Android arm64 | Documentation provided; not yet tested end-to-end |
| Linux | Should work (same as macOS — replace `ipconfig getifaddr en0` with `hostname -I`) | Android arm64 | Untested |
| iOS (iPhone) | macOS only | arm64-apple-ios | Not yet implemented |

### Future Work

- **Windows**: end-to-end test and CI validation
- **iOS**: recompile libtorch and gRPC for `arm64-apple-ios`, replace JNI bridge with Objective-C++ TurboModule, add CocoaPods integration
