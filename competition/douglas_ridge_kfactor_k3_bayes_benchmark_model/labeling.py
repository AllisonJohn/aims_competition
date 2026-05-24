"""Deterministic pseudo-random acquisition.

The challenge chooses the largest scores returned by acquisition_function, so
this stable hash behaves like random sampling while remaining reproducible.
"""

from __future__ import annotations

import hashlib


def _stable_uniform(*parts: object) -> float:
    text = "\x1f".join(str(part or "") for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def acquisition_function(input: dict) -> float:
    return _stable_uniform(
        input.get("benchmark"),
        input.get("condition"),
        input.get("subject_content"),
        input.get("item_content"),
    )
