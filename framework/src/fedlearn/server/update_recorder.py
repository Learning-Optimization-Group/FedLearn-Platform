"""Research-only recording of the individual client updates a coordinator accepts.

A multi-device run can otherwise be checked only through its aggregate, where a disagreement between clients'
compute backends can average out unseen. With ``FEDLEARN_RECORD_CLIENT_UPDATES=<path>``, every DeComFL update the
coordinator accepts is appended to ``<path>`` as one JSON line, exactly as it will be aggregated (after the
ingress clamp).

Writing individual updates to disk exposes each client's contribution. It is therefore off unless asked for, and
refused on a run with secure aggregation or central differential privacy, whose purpose is that no individual
update is kept.
"""
from __future__ import annotations

import json
import os
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
