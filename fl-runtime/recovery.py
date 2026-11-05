"""Private session binding and single-process ownership for runtime checkpoints."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import secrets
import stat
import sys
import tempfile
from uuid import UUID

import torch

from fedlearn.server.checkpoint import CheckpointIdentity, RoundCheckpointStore


_SESSION_KEYS = {"version", "run_id", "seed", "configuration_sha256", "source_sha256", "initial_model_sha256"}
_LOCATION_ARGUMENTS = {"checkpoint_directory", "resume_from_checkpoint", "port", "model_path", "init_model_path"}
_BOUND_ENVIRONMENT = (
    "FEDLEARN_REQUIRE_CLIENT_AUTH", "FEDLEARN_REQUIRE_TLS", "FEDLEARN_GRPC_USE_TLS",
    "FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH", "FEDLEARN_LLM_LORA_BASE", "FEDLEARN_LLM_LORA_SUBSET",
    "FEDLEARN_ROUND_TIMEOUT_S", "FEDLEARN_PNEUMONIA_DIR", "FEDLEARN_PNEUMONIA_DATASET",
    "FEDLEARN_PNEUMONIA_CONFIG", "FEDLEARN_PNEUMONIA_REVISION", "FEDLEARN_PNEUMONIA_SUBSET",
    "FEDLEARN_NIH_SOURCE", "FEDLEARN_NIH_PREPARED", "FEDLEARN_NIH_EVAL_MAX_IMAGES", "FEDLEARN_NIH_EVAL_SEED",
)


def _digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions():
    versions = {}
    for name in ("torch", "numpy", "safetensors", "torchvision", "transformers", "datasets",
                 "grpcio", "scikit-learn", "scipy", "pandas"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"packages": versions, "python": list(sys.version_info[:3]), "machine": platform.machine(),
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def source_sha256(source_files):
    """Hash named source bytes and runtime versions without embedding local paths."""
    sources = sorted(source_files, key=lambda item: item[0])
    if not sources or len({name for name, _ in sources}) != len(sources):
        raise ValueError("recovery source identity requires unique named source files")
    digest = hashlib.sha256(json.dumps(_versions(), sort_keys=True, allow_nan=False).encode())
    for name, path in sources:
        raw = Path(path).read_bytes()
        digest.update(json.dumps([name, len(raw)], separators=(",", ":")).encode())
        digest.update(raw)
    return digest.hexdigest()


def _runtime_sources():
    import fedlearn
    runtime = Path(__file__).resolve().parent
    framework = Path(fedlearn.__file__).resolve().parent
    return ([("runtime/" + file.name, file) for file in runtime.glob("*.py")]
            + [("runtime/models/" + file.relative_to(runtime / "models").as_posix(), file)
               for file in (runtime / "models").rglob("*.py")]
            + [("fedlearn/" + file.relative_to(framework).as_posix(), file)
               for file in framework.rglob("*.py")])


def _private_file(descriptor):
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ValueError("recovery session files must be private regular files")
    return info


def _read_session(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        if _private_file(source.fileno()).st_size > 64 * 1024:
            raise ValueError("recovery session metadata exceeds supported size")
        try:
            session = json.loads(source.read(64 * 1024 + 1))
        except (ValueError, UnicodeError) as error:
            raise ValueError("recovery session metadata is invalid") from error
    if (not isinstance(session, dict) or set(session) != _SESSION_KEYS
            or type(session["version"]) is not int or session["version"] != 1
            or type(session["seed"]) is not int or not 0 <= session["seed"] < 2**32):
        raise ValueError("recovery session identity/configuration is invalid")
    return session


def _write_session(path, session):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".session-", delete=False) as output:
            temporary = Path(output.name)
            output.write((json.dumps(session, sort_keys=True, allow_nan=False) + "\n").encode())
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _pin_initial(source, destination, expected_sha256):
    if destination.exists():
        descriptor = os.open(destination, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            _private_file(descriptor)
        finally:
            os.close(descriptor)
        if _digest_file(destination) != expected_sha256:
            raise ValueError("recovery pinned initial model differs from its recorded identity")
        return
    temporary = None
    try:
        digest = hashlib.sha256()
        with Path(source).open("rb") as original, tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=".initial-", delete=False) as output:
            temporary = Path(output.name)
            for chunk in iter(lambda: original.read(1024 * 1024), b""):
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != expected_sha256:
            raise ValueError("recovery initial model changed while its private copy was prepared")
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class RuntimeRecovery:
    directory: Path
    identity: CheckpointIdentity
    store: RoundCheckpointStore
    resume: bool
    initial_model: Path
    initial_model_sha256: str
    source_files: tuple
    source_digest: str
    original_initial_model: Path | None

    def server_options(self):
        return {"checkpoint_directory": str(self.directory), "checkpoint_identity": self.identity,
                "resume_from_checkpoint": self.resume}

    def validate_inputs(self):
        if _digest_file(self.initial_model) != self.initial_model_sha256:
            raise ValueError("recovery initial model changed during startup")
        if (self.original_initial_model is not None
                and _digest_file(self.original_initial_model) != self.initial_model_sha256):
            raise ValueError("recovery initial model changed during startup")
        if source_sha256(self.source_files) != self.source_digest:
            raise ValueError("recovery source/runtime changed during startup")


@contextmanager
def runtime_recovery(args, *, env=None, source_files=None):
    """Hold an exclusive private session through training and final callbacks."""
    directory = getattr(args, "checkpoint_directory", None)
    resume = getattr(args, "resume_from_checkpoint", False)
    if directory is None:
        if resume:
            raise ValueError("resume requires --checkpoint-directory")
        yield None
        return
    if os.name != "posix":
        raise ValueError("runtime recovery is currently qualified only on POSIX servers")
    if (args.strategy.lower() not in {"fedavg", "fedprox", "fedopt"}
            or args.secure_aggregation or args.dp_enabled or torch.cuda.is_available()):
        raise ValueError("runtime recovery currently supports qualified CPU FedAvg/FedProx/FedOpt runs only")
    env = os.environ if env is None else env
    run_id = env.get("FEDLEARN_RUN_ID", "")
    try:
        if str(UUID(run_id)) != run_id:
            raise ValueError("not canonical")
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("recovery requires a canonical run UUID in FEDLEARN_RUN_ID") from error
    directory = Path(directory)
    if Path(args.model_path).resolve().is_relative_to(directory.resolve()):
        raise ValueError("model output must remain outside the private recovery directory")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
        raise ValueError("recovery directory must be private and must not be a symbolic link")
    import fcntl
    lease = os.open(directory / ".owner.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        _private_file(lease)
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another training process already owns this recovery directory") from error
        session_path = directory / "session.json"
        saved = _read_session(session_path) if session_path.exists() else None
        if saved is not None and not resume:
            raise ValueError("existing recovery session requires an explicit resume request")
        if saved is None and resume:
            raise ValueError("no runtime session/checkpoint is available to resume")
        seed = args.seed
        if seed is None:
            seed = saved["seed"] if saved is not None else secrets.randbelow(2**31)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("recovery seed must be an integer in [0, 2**32)")
        args.seed = seed
        sources = tuple(_runtime_sources() if source_files is None else source_files)
        original_initial = Path(args.init_model_path or args.model_path)
        initial_model = directory / "initial-model.npz"
        if saved is None:
            initial_digest = _digest_file(original_initial)
        else:
            initial_digest = saved["initial_model_sha256"]
            if not initial_model.exists():
                raise ValueError("recovery pinned initial model is missing")
            if args.init_model_path is not None and _digest_file(original_initial) != initial_digest:
                raise ValueError("recovery explicit initial model differs from its original identity")
        _pin_initial(original_initial, initial_model, initial_digest)
        if Path(args.model_path).exists() and os.path.samefile(args.model_path, initial_model):
            raise ValueError("model output must not alias the private initial model")
        source_digest = source_sha256(sources)
        configuration = {
            "arguments": {key: value for key, value in vars(args).items() if key not in _LOCATION_ARGUMENTS},
            "environment": {key: env.get(key) for key in _BOUND_ENVIRONMENT},
            "source_sha256": source_digest, "initial_model_sha256": initial_digest,
        }
        config_digest = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"),
                                                  allow_nan=False).encode()).hexdigest()
        session = {"version": 1, "run_id": run_id, "seed": seed, "configuration_sha256": config_digest,
                   "initial_model_sha256": initial_digest, "source_sha256": source_digest}
        if saved is not None and saved != session:
            raise ValueError("recovery run identity, seed, configuration, source or initial model differs")
        identity = CheckpointIdentity(run_id, config_digest)
        store = RoundCheckpointStore(directory, identity)
        latest = store.load_latest()
        if resume and latest is None:
            raise ValueError("no completed-round checkpoint is available to resume")
        if saved is None:
            if latest is not None:
                raise ValueError("checkpoint directory has no matching runtime session identity")
            _write_session(session_path, session)
        yield RuntimeRecovery(directory, identity, store, resume, initial_model, initial_digest,
                              sources, source_digest, original_initial if not resume or args.init_model_path else None)
    finally:
        os.close(lease)
