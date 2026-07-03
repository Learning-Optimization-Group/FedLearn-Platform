import os
import sys

# The FL trainer scripts (fl_server.py, client.py, …) `import fedlearn` — the custom framework
# that, in a real deploy, is pip-installed alongside them. The backend-scripts CI job and a bare
# local checkout don't install it, so any test that imports one of those scripts would fail at
# collection with ModuleNotFoundError. Put the in-repo framework source on sys.path here (walking
# up to the repo root that carries framework/src) so `import fedlearn` resolves to the checked-out
# framework without requiring a separate editable install.
_here = os.path.dirname(os.path.abspath(__file__))
_dir = _here
for _ in range(12):
    _candidate = os.path.join(_dir, "framework", "src")
    if os.path.isdir(_candidate):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break
    _parent = os.path.dirname(_dir)
    if _parent == _dir:
        break
    _dir = _parent
