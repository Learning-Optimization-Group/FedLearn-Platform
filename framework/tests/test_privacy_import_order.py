"""Public privacy/backbone entry points must work in a fresh interpreter."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("first", ["fedlearn.privacy.dp_mechanism", "benchmarks.dp_on_head_xray",
                                   "fedlearn.server.strategy_factory"])
def test_privacy_and_factory_entry_points_do_not_depend_on_import_order(first):
    code = f"""
import importlib
importlib.import_module({first!r})
from fedlearn.server import create_strategy, STRATEGY_REGISTRY
from fedlearn.privacy.dp_mechanism import clipped_deltas, replace_one_sensitivity
assert callable(create_strategy) and isinstance(STRATEGY_REGISTRY, dict)
assert callable(clipped_deltas)
assert replace_one_sensitivity('mean', 10, 1.0) == 0.2
namespace = {{}}
exec('from fedlearn.server import *', namespace)
assert namespace['create_strategy'] is create_strategy
assert namespace['STRATEGY_REGISTRY'] is STRATEGY_REGISTRY
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
