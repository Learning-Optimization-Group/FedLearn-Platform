"""The bundled environment must load the checked-in protocol before launching a client."""
from pathlib import Path
import ast
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]


def test_packaged_runtime_meets_generated_grpc_and_protobuf_minimums():
    pins = dict(line.strip().split('==',1) for line in (ROOT/'client-docker/packaging/requirements-client.txt').read_text().splitlines()
                if '==' in line and not line.lstrip().startswith('#'))
    generated = ROOT/'framework/src/fedlearn/communication/generated'
    for module in generated.glob('*_pb2_grpc.py'):
        tree = ast.parse(module.read_text())
        required = next(node.value.value for node in tree.body if isinstance(node,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='GRPC_GENERATED_VERSION' for t in node.targets))
        assert Version(pins['grpcio']) >= Version(required), module.name
    for module in generated.glob('*_pb2.py'):
        tree = ast.parse(module.read_text())
        for node in tree.body:
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and getattr(node.value.func,'attr','')=='ValidateProtobufRuntimeVersion':
                required = '.'.join(str(arg.value) for arg in node.value.args[1:4])
                assert Version(pins['protobuf']) >= Version(required), module.name
