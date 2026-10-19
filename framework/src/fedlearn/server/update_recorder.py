"""Research-only recording of the individual client updates a coordinator accepts.

A multi-device run can otherwise be checked only through its aggregate, where a disagreement between clients'
compute backends can average out unseen. With ``FEDLEARN_RECORD_CLIENT_UPDATES=<path>``, accepted DeComFL updates
and first-order model updates are recorded after ingress clipping. Model records reference content-addressed
float32 safetensors files beside the JSONL log, so each client's update can be replayed independently.

Writing individual updates to disk exposes each client's contribution. It is therefore off unless asked for, and
refused on a run with secure aggregation or central differential privacy, whose purpose is that no individual
update is kept.
"""
from __future__ import annotations

import json
import os
import hashlib
import tempfile
from pathlib import Path
from typing import List, Optional

ENV_VAR = "FEDLEARN_RECORD_CLIENT_UPDATES"


class ClientUpdateRecorder:
    """Appends accepted client updates to a JSONL file, one line per update, flushed immediately."""

    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)

    def record_decomfl(self, round_number: int, client_id: str, gradient_scalars: List[List[float]],
                       num_examples: int) -> None:
        line = {"round": round_number, "client_id": client_id, "num_examples": num_examples,
                "gradient_scalars": gradient_scalars}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")

    def record_model(self, round_number: int, client_id: str, parameters, num_examples: int) -> None:
        from fedlearn.communication.safetensors_codec import save_safetensors

        payload = save_safetensors([(key, value.detach().float().cpu().numpy())
                                   for key, value in parameters.items()])
        digest = hashlib.sha256(payload).hexdigest()
        directory = self.path.with_name(self.path.name + '.states')
        directory.mkdir(exist_ok=True)
        target = directory / (digest + '.safetensors')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as fh:
                temporary = Path(fh.name)
                fh.write(payload)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        line = {'round': round_number, 'client_id': client_id, 'num_examples': num_examples,
                'model_state': {'codec': 'safetensors-f32-v1', 'path': str(target.relative_to(self.path.parent)),
                                'sha256': digest, 'bytes': len(payload)}}
        with self.path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(line, allow_nan=False) + '\n')


def update_recorder_from_env(*, secure_aggregation: bool, dp_enabled: bool) -> Optional[ClientUpdateRecorder]:
    """The recorder ``FEDLEARN_RECORD_CLIENT_UPDATES`` asks for, or None; raises on a run that must not record."""
    path = os.environ.get(ENV_VAR)
    if not path:
        return None
    if secure_aggregation or dp_enabled:
        raise ValueError(
            f"{ENV_VAR} would record individual client updates, which a run with "
            f"{'secure aggregation' if secure_aggregation else 'central differential privacy'} must not keep; "
            f"unset it for this run")
    return ClientUpdateRecorder(path)
