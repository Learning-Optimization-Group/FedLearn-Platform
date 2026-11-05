"""Commit a complete final model before publishing its bytes or successful completion."""
import os
from pathlib import Path
import tempfile

import numpy as np


def save_final_model(path, parameters):
    """Replace the destination only after its new archive has been flushed to disk."""
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            np.savez(output, **parameters)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        temporary = None
        if os.name == "posix":
            directory = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except Exception as error:
        raise RuntimeError("Final model could not be saved durably; publication and completion are refused") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
