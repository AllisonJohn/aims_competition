"""Pure salted-random label acquisition for exact-root residual calibration."""

from __future__ import annotations

import hashlib


SALT = "7"


def _stable_uniform(*parts: object) -> float:
    text = "\x1f".join(str(part or "") for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def acquisition_function(input: dict) -> float:
    return _stable_uniform(
        SALT,
        input.get("benchmark"),
        input.get("condition"),
        input.get("subject_content"),
        input.get("item_content"),
    )
