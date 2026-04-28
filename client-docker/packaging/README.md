# FedLearn Native Client — Packaging

This directory builds standalone executables of the FedLearn training client
using PyInstaller. The resulting bundles are shipped inside the Electron
desktop app's resources so end users don't need Docker, Python, or a repo
checkout — install the desktop app and training just works.

## Targets

| Target | Host needed to build | Torch wheel | Output dir |
|---|---|---|---|
| `macOS arm64` | Apple Silicon Mac, Python 3.11+ | MPS-enabled (default index) | `dist/fedlearn-client/` |
| `macOS x64` | Intel Mac, Python 3.11+ | CPU-only (default index) | `dist/fedlearn-client/` |
| `Windows x64 CPU` | Windows 10/11 x64, Python 3.11+ | CPU-only | `dist/fedlearn-client/` |
| `Windows x64 CUDA 12.4` | Windows 10/11 x64 + NVIDIA driver >= 550, Python 3.11+ | CUDA 12.4 | `dist/fedlearn-client/` |
| `Linux x86_64` | x86_64 Linux, Python 3.11+ | CPU-only (pytorch.org/whl/cpu) | `dist/fedlearn-client/` |
| `Linux aarch64` | aarch64 Linux, Python 3.11+ | CPU-only (default PyPI manylinux_aarch64) | `dist/fedlearn-client/` |

**PyInstaller does not cross-compile.** Each OS/arch combo needs a native
host. Jetson is intentionally not supported as a native build — its L4T
torch wheel is pinned to a specific JetPack firmware stack; Docker
(`nvcr.io/nvidia/l4t-pytorch`) is the correct deployment path there.

Linux GPU is also intentionally out of scope. x86_64+NVIDIA users can use
the Docker path (`fedlearn-client:latest` image); the GA release pipeline
ships CPU-only native bundles for Linux.

## Build commands

From this directory:

```bash
# macOS (arch-aware — runs on both arm64 and x86_64 hosts)
./build-mac.sh

# Linux (arch-aware — runs on both x86_64 and aarch64 hosts)
./build-linux.sh

# Windows x64 CPU (PowerShell)
.\build-win-cpu.ps1

# Windows x64 CUDA (PowerShell)
.\build-win-cuda.ps1
```

Each script creates a local venv, installs pinned deps + the chosen torch
wheel, installs the `fedlearn` framework with `--no-deps` (to preserve our
pins), runs PyInstaller, and smoke-tests the resulting binary with `--help`.

## Expected output

```
dist/fedlearn-client/
├── fedlearn-client          # the entry-point executable (or .exe on Windows)
├── _internal/               # PyInstaller runtime libs
│   ├── torch/
│   ├── transformers/
│   └── ...
└── ...
```

Typical bundle sizes: **~770 MB** Mac arm64 (MPS), **~600 MB** Mac x64 (CPU),
**~500 MB** Linux x86_64/aarch64 (CPU), **~300 MB** Windows x64 CPU, **~2.5 GB**
Windows x64 CUDA. Most of this is `torch` + `transformers` model conversion
scripts.

## Packaging into the Electron app

After building the bundle, from `fedlearn-desktop/`:

```bash
# Mac
npm run package:mac

# Windows CPU (produces FedLearn-Desktop-Setup-X.Y.Z-cpu.exe)
npm run package:win:cpu

# Windows CUDA (produces FedLearn-Desktop-Setup-X.Y.Z-cuda.exe)
npm run package:win:cuda
```

`electron-builder.yml` is configured with `extraResources` pointing at
`client-docker/packaging/dist/fedlearn-client/`, so the bundle is copied
verbatim into the installer. At runtime `DockerService.resolveNativeInvocation`
looks it up via `process.resourcesPath` + `/fedlearn-client/fedlearn-client[.exe]`.

**Only one Windows variant can exist at `dist/fedlearn-client/` at a time** —
pick CPU or CUDA before running the corresponding `package:win:*` script.

## Pinning

`requirements-client.txt` pins every runtime dependency. The primary pin
that matters: `torch==2.5.1` — has official wheels across Mac arm64 (MPS),
Win x64 CPU, and Win x64 CUDA 12.4. PyInstaller handles it cleanly with the
`collect_all('transformers')` pattern in the spec.

To upgrade torch, change the three wheel-install lines in `build-mac.sh` /
`build-win-cpu.ps1` / `build-win-cuda.ps1` together. Version drift across
platforms will produce inconsistent training results.

## Troubleshooting

**`ResolutionImpossible` during pip install**: a transitive dep is pulling
a conflicting version. First check you're installing the framework with
`--no-deps` — that's the most common cause (the framework's
`requirements.txt` pulls in `flwr` which cascades down). If the conflict is
genuine, relax the offending pin in `requirements-client.txt`.

**`Expected client entry at …`**: the spec couldn't find `client.py`.
`SPECPATH` inside the spec file is the directory containing the spec file,
not a file path — don't `dirname` it.

**Silent `.exe` crash on launch**: run the `.exe` directly from `cmd.exe`
(not via a double-click) to see the real traceback. Most common root causes
are missing `collect_all` for a dynamic-import library — add it to the
`FULL_COLLECT` tuple in the spec.

**Bundle too large**: we deliberately leave transformers' `convert_*` scripts
in because stripping them is whack-a-mole across versions. If size becomes a
real problem, add an `excludes` entry for `transformers.models.<family>.convert_*`
patterns you know you don't need.

## What this replaces

Before: the Electron app spawned a Docker container running a Python client
that users had to build separately with `docker build -t fedlearn-client:latest`.
End users needed Docker Desktop installed, the image pulled or built, and
(on Windows) WSL2 configured.

After: the Electron installer includes the native client directly. Docker is
still used for the Jetson profile, but Mac and Windows users see zero Docker.
