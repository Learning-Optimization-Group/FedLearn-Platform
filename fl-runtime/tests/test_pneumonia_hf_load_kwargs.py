# SCRIPTS/tests/test_pneumonia_hf_load_kwargs.py
"""SE-19: the PNEUMONIA_CNN HuggingFace-dataset fallback must not auto-execute remote code.

datasets 5 no longer supports script-backed dataset loaders. Reject the former remote-code opt-in
with migration guidance rather than forwarding an unsupported flag to datasets.load_dataset.

Two contracts:
  1. The pure kwargs seam `_hf_load_kwargs` defaults to no `trust_remote_code`, pins `revision` when
     configured, and rejects the obsolete remote-code opt-in.
  2. A wiring guard: `_full_dataset` actually routes its HF load through that seam (bytecode
     introspection — robust to comments/docstrings, and RED against the old inline dict literal).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402


def test_hf_load_kwargs_does_not_enable_remote_code_by_default():
    kw = recipes._hf_load_kwargs("train", env={})
    assert kw.get("trust_remote_code") is not True, "SE-19: HF loader must not execute remote code by default"
    assert kw["split"] == "train"
    assert "revision" not in kw  # nothing pinned unless configured


def test_hf_load_kwargs_pins_revision_when_configured():
    kw = recipes._hf_load_kwargs("test", env={"FEDLEARN_PNEUMONIA_REVISION": "abc123def"})
    assert kw["revision"] == "abc123def"
    assert kw.get("trust_remote_code") is not True  # pinning a commit is not opting into code exec


@pytest.mark.parametrize("value", ["1", " 1 "])
def test_hf_load_kwargs_rejects_obsolete_remote_code_opt_in_with_migration_guidance(value):
    with pytest.raises(ValueError, match="datasets 5.*script-backed") as error:
        recipes._hf_load_kwargs("train", env={"FEDLEARN_PNEUMONIA_TRUST_REMOTE_CODE": value})
    assert "FEDLEARN_PNEUMONIA_DIR" in str(error.value)
    assert "Parquet" in str(error.value)


def test_hf_load_kwargs_ignores_non_1_opt_in_values():
    for val in ("0", "", "true", "yes", "  "):
        kw = recipes._hf_load_kwargs("train", env={"FEDLEARN_PNEUMONIA_TRUST_REMOTE_CODE": val})
        assert kw.get("trust_remote_code") is not True, f"only the literal '1' should opt in, not {val!r}"


def test_full_dataset_routes_through_the_safe_kwargs_seam():
    """The HF fallback must build its kwargs via `_hf_load_kwargs`, not an inline literal.

    On the pre-fix code `_full_dataset` built `{"split": ..., "trust_remote_code": True}` inline and
    never referenced the seam, so this is RED before the fix and GREEN after — and unlike a source
    regex it cannot be tripped by a docstring that merely mentions the literal.
    """
    assert "_hf_load_kwargs" in recipes._full_dataset.__code__.co_names, \
        "SE-19: _full_dataset must obtain load_dataset kwargs from the safe _hf_load_kwargs seam"
