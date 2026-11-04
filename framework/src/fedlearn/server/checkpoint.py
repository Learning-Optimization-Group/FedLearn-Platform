"""Private, immutable and integrity-checked completed-round checkpoints."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import zipfile

import numpy as np
from safetensors.torch import load as load_tensors, save as save_tensors

from .strategy_checkpoint import RoundSnapshot, capture_round, restore_strategy, validate_snapshot

MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_STATE_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class CheckpointIdentity:
    run_id: str
    configuration_sha256: str

    def __post_init__(self):
        if (not isinstance(self.run_id, str) or not self.run_id.strip() or len(self.run_id) > 256
                or not isinstance(self.configuration_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.configuration_sha256) is None):
            raise ValueError("checkpoint identity requires a run ID and SHA-256 configuration digest")


def _private(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or info.st_mode & 0o077:
        raise ValueError("checkpoint paths must be private and must not be symbolic links")
    return info


class RoundCheckpointStore:
    """One trusted server directory per run; never include it in public exports."""

    def __init__(self, directory, identity: CheckpointIdentity):
        self.directory = Path(directory)
        self.identity = identity
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not stat.S_ISDIR(_private(self.directory).st_mode):
            raise ValueError("checkpoint directory must be a private directory")

    def _generations(self):
        _private(self.directory)
        result = []
        for path in self.directory.glob("round-*.ckpt"):
            match = re.fullmatch(r"round-([0-9]{12})\.ckpt", path.name)
            if match is None:
                raise ValueError("checkpoint generation filename is invalid")
            result.append((int(match[1]), path))
        return sorted(result)

    def save(self, snapshot: RoundSnapshot) -> Path:
        validate_snapshot(snapshot)
        generations = self._generations()
        if generations and generations[-1][0] >= snapshot.completed_round:
            raise FileExistsError("checkpoint generations cannot be overwritten or rewound")
        # Check the existing identity/integrity before appending a generation.
        if generations:
            self.load_latest()
        tensors = {"torch_rng": snapshot.torch_rng_state.contiguous()}
        layout = {}
        for group, values in {"parameters": snapshot.parameters, **snapshot.strategy_tensors}.items():
            layout[group] = []
            for index, (name, value) in enumerate(values.items()):
                key = f"{group}/{index}"
                tensors[key] = value.detach().cpu().contiguous()
                layout[group].append([name, key])
        state = save_tensors(tensors)
        numpy_state = snapshot.numpy_rng_state
        manifest = {
            "version": 1,
            "identity": {"run_id": self.identity.run_id,
                         "configuration_sha256": self.identity.configuration_sha256},
            "completed_round": snapshot.completed_round, "total_rounds": snapshot.total_rounds,
            "strategy_type": snapshot.strategy_type, "strategy_config": snapshot.strategy_config,
            "layout": layout, "history": snapshot.history,
            "partition_to_client": [[key, value] for key, value in snapshot.partition_to_client.items()],
            "python_rng_state": snapshot.python_rng_state,
            "numpy_rng_state": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "state_sha256": hashlib.sha256(state).hexdigest(),
        }
        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > MAX_MANIFEST_BYTES or len(state) > MAX_STATE_BYTES:
            raise ValueError("checkpoint archive exceeds supported size")
        target = self.directory / f"round-{snapshot.completed_round:012d}.ckpt"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".checkpoint-", delete=False) as output:
                temporary = Path(output.name)
                with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
                    archive.writestr("manifest.json", raw)
                    archive.writestr("state.safetensors", state)
                output.flush()
                os.fsync(output.fileno())
            # A same-directory hard link publishes the complete file without replacing prior state.
            os.link(temporary, target)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target

    def load_latest(self) -> RoundSnapshot | None:
        generations = self._generations()
        if not generations:
            return None
        round_number, path = generations[-1]
        info = _private(path)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("checkpoint generation must be a private regular file")
        try:
            with zipfile.ZipFile(path) as archive:
                if (len(archive.infolist()) != 2
                        or set(archive.namelist()) != {"manifest.json", "state.safetensors"}):
                    raise ValueError("checkpoint archive members are invalid")
                if (archive.getinfo("manifest.json").file_size > MAX_MANIFEST_BYTES
                        or archive.getinfo("state.safetensors").file_size > MAX_STATE_BYTES):
                    raise ValueError("checkpoint archive exceeds supported size")
                manifest = json.loads(archive.read("manifest.json"))
                state = archive.read("state.safetensors")
            if type(manifest["version"]) is not int or manifest["version"] != 1:
                raise ValueError("checkpoint schema version is unsupported")
            expected_identity = {"run_id": self.identity.run_id,
                                 "configuration_sha256": self.identity.configuration_sha256}
            if manifest["identity"] != expected_identity:
                raise ValueError("checkpoint identity does not match the requested run/configuration")
            if hashlib.sha256(state).hexdigest() != manifest["state_sha256"]:
                raise ValueError("checkpoint tensor checksum does not match")
            if manifest["completed_round"] != round_number:
                raise ValueError("checkpoint filename and completed round disagree")
            tensors = load_tensors(state)
            groups = {}
            consumed = {"torch_rng"}
            for group, layout in manifest["layout"].items():
                values = OrderedDict()
                for name, key in layout:
                    if name in values or key in consumed:
                        raise ValueError("checkpoint tensor layout contains duplicates")
                    values[name] = tensors[key]
                    consumed.add(key)
                groups[group] = values
            if consumed != set(tensors):
                raise ValueError("checkpoint tensor layout has unbound state")
            numpy_state = manifest["numpy_rng_state"]
            bindings = manifest["partition_to_client"]
            if len(dict(bindings)) != len(bindings):
                raise ValueError("checkpoint identity bindings contain duplicates")
            snapshot = RoundSnapshot(
                manifest["completed_round"], manifest["total_rounds"], groups.pop("parameters"),
                manifest["strategy_type"], manifest["strategy_config"], groups,
                [(number, metrics) for number, metrics in manifest["history"]], dict(bindings),
                tensors["torch_rng"], manifest["python_rng_state"],
                (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]),
            )
            validate_snapshot(snapshot)
            return snapshot
        except (zipfile.BadZipFile, KeyError, TypeError, IndexError, UnicodeError,
                json.JSONDecodeError, RuntimeError) as error:
            raise ValueError("checkpoint is malformed or incomplete") from error
