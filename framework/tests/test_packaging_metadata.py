"""FR-9 done-when #3: the trove classifiers must not advertise a Python version
below the ``python_requires`` floor.

The package declares ``python_requires='>=3.10'`` but historically still shipped a
``Programming Language :: Python :: 3.9`` classifier — a contradiction that tells
PyPI/pip the package supports a version it explicitly refuses to install on. This
test pins the classifier set to the supported range and keys the lower bound off
``python_requires`` so a future floor bump can't silently leave a stale classifier
behind.
"""

import re
from pathlib import Path

SETUP_PY = Path(__file__).resolve().parent.parent / "setup.py"

_PY_CLASSIFIER = re.compile(r"Programming Language :: Python :: (\d+)\.(\d+)")
_REQUIRES = re.compile(r"python_requires\s*=\s*['\"]>=\s*(\d+)\.(\d+)")


def _read_setup() -> str:
    return SETUP_PY.read_text(encoding="utf-8")


def _python_requires_floor(text: str) -> tuple[int, int]:
    m = _REQUIRES.search(text)
    assert m, "setup.py must declare a python_requires floor of the form >=X.Y"
    return int(m.group(1)), int(m.group(2))


def _versioned_classifiers(text: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in _PY_CLASSIFIER.findall(text)]


def test_no_python_classifier_below_the_python_requires_floor():
    text = _read_setup()
    floor = _python_requires_floor(text)
    below = [f"{maj}.{minor}" for maj, minor in _versioned_classifiers(text) if (maj, minor) < floor]
    assert not below, (
        f"setup.py classifiers advertise Python {below} but python_requires is "
        f">={floor[0]}.{floor[1]}; drop the sub-floor classifier(s)."
    )


def test_python_39_classifier_is_gone():
    # The specific regression FR-9 #3 reopened on.
    assert "Programming Language :: Python :: 3.9" not in _read_setup()


def test_versioned_classifiers_are_exactly_the_supported_range():
    # Done-when #3: classifiers name 3.10-3.12 only.
    assert set(_versioned_classifiers(_read_setup())) == {(3, 10), (3, 11), (3, 12)}
