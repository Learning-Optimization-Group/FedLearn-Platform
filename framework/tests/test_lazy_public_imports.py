import subprocess
import sys


def test_tree_package_does_not_import_torch():
    result = subprocess.run([sys.executable, "-c",
                             "import fedlearn.tree.model,sys; assert 'torch' not in sys.modules"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_neural_exports_remain_available():
    import fedlearn
    from fedlearn.client.client import Client
    from fedlearn.server.strategy import FedAvg
    assert fedlearn.Client is Client
    assert fedlearn.FedAvg is FedAvg
