"""Bounded image identity reader with a stable import path for spawned processes."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import warnings

from PIL import Image


def read_image_identity(root: Path, row: dict, *, max_file_bytes: int, max_pixels: int) -> dict:
    """Read one bounded PNG and return stable file/RGB identity or a failure category."""
    record = dict(row)
    record.update(file_bytes="", file_sha256="", width="", height="", rgb_sha256="", error="")
    path = root / row["image_path"]
    try:
        if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
            raise ValueError("image escaped dataset root")
        record["file_bytes"] = path.stat().st_size
        if record["file_bytes"] > max_file_bytes:
            record["error"] = "file_too_large"
            return record
        with path.open("rb") as file:
            raw = file.read(max_file_bytes + 1)
        record["file_bytes"] = len(raw)
        if len(raw) > max_file_bytes:
            record["error"] = "file_too_large"
            return record
        record["file_sha256"] = hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError):
        record["error"] = "unreadable_file"
        return record
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                record["width"], record["height"] = image.size
                if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                    record["error"] = "not_single_frame_png"
                    return record
                if image.width * image.height > max_pixels:
                    record["error"] = "pixel_limit"
                    return record
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                rgb = image.convert("RGB")
                digest = hashlib.sha256(b"capstone-nih-rgb-v1\0")
                digest.update(f"{rgb.width}x{rgb.height}\0".encode("ascii"))
                digest.update(rgb.tobytes())
                record["rgb_sha256"] = digest.hexdigest()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        record["error"] = "decode_error"
    return record
