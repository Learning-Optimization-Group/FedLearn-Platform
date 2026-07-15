# -*- mode: python ; coding: utf-8 -*-
# =============================================================================
# FedLearn Native Client — PyInstaller Spec
# =============================================================================
# Produces a single onedir bundle `dist/fedlearn-client/` containing the
# executable plus all native libs. The Electron app ships this directory
# inside its resources/ and spawns the binary at runtime — no system Python
# or repo checkout required on the end-user's machine.
#
# Build invocation (per-platform wrapper scripts do the venv + torch install):
#   pyinstaller --clean --noconfirm fedlearn-client.spec
# =============================================================================

import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

# SPECPATH is the directory containing this spec file, not a file path.
SPEC_DIR = os.path.abspath(SPECPATH)
RUNTIME_DIR = os.path.abspath(os.path.join(SPEC_DIR, '..', '..', 'fl-runtime'))
CLIENT_ENTRY = os.path.join(RUNTIME_DIR, 'client.py')

if not os.path.isfile(CLIENT_ENTRY):
    raise SystemExit(f'Expected client entry at {CLIENT_ENTRY}')

datas = []
binaries = []
hiddenimports = []

# Libraries with heavy dynamic dispatch — collect everything (data files,
# hidden submodules, native binaries). Omitting these causes runtime
# ModuleNotFoundError / missing-config errors that don't surface at build time.
FULL_COLLECT = (
    'transformers',
    'tokenizers',
    'datasets',
    'flwr_datasets',
    'huggingface_hub',
    'safetensors',
    'accelerate',
    # scipy is pulled in by sklearn. Its Cython-compiled _cyutility /
    # _ccallback_c submodules are missed by PyInstaller's default analysis,
    # producing "scipy install seems to be broken" at runtime without this.
    'scipy',
    'sklearn',
    'fedlearn',        # installed via `pip install -e ../../framework`
)

for pkg in FULL_COLLECT:
    try:
        d, b, h = collect_all(pkg)
        datas.extend(d)
        binaries.extend(b)
        hiddenimports.extend(h)
    except Exception as exc:
        print(f'[spec] collect_all({pkg}) failed: {exc}', file=sys.stderr)
        raise

# sklearn uses string-based submodule dispatch in several places
hiddenimports.extend(collect_submodules('sklearn'))

# Import ordering safety net — sklearn before torch on ARM64 resolves
# libgomp static-TLS allocation issues (see client.py boot sequence).
hiddenimports.extend([
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.tree._utils',
])

# Bundle the sibling modules that client.py imports by name (directly or
# transitively). These live in fl-runtime/ next to client.py. Kept in sync with
# the transitive-import audit in the DA-5 plan — a missing entry surfaces only as
# a runtime ModuleNotFoundError in the frozen binary, not at build time.
LOCAL_SIBLINGS = ['config.py', 'data.py', 'recipes.py', 'model_utils.py',
                  'device.py', 'models', 'data_loaders', 'architecture']
for name in LOCAL_SIBLINGS:
    src = os.path.join(RUNTIME_DIR, name)
    if os.path.isdir(src):
        datas.append((src, name))
    elif os.path.isfile(src):
        datas.append((src, '.'))

a = Analysis(
    [CLIENT_ENTRY],
    pathex=[RUNTIME_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Interactive-dev tools — saves ~200MB and none are reachable from the
        # training codepath.
        # NOTE: matplotlib cannot be excluded — flwr_datasets/__init__.py
        # eagerly imports its visualization submodule at top level.
        'PIL.ImageTk',
        'tkinter',
        'IPython',
        'jedi',
        'parso',
        'pytest',
        'notebook',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='fedlearn-client',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX compression breaks torch dylib loading on macOS and produces
    # AV false-positives on Windows. Disabled deliberately.
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='fedlearn-client',
)
