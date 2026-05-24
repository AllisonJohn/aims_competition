"""Acquire cases where the k3 factor model and base prior disagree."""

from __future__ import annotations

import hashlib
import math
import re

try:
    import model
except Exception:
    model = None


def _stable_uniform(*parts: object) -> float:
    text = "\x1f".join(str(part or "") for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def _heuristic_diversity(item_content: object) -> float:
    text = str(item_content or "")
    lower = text.lower()
    score = 0.0
    if re.search(r"\b(prove|derive|justify|counterexample|theorem)\b", lower):
        score += 0.015
    if re.search(r"\b(def|class|import|function|debug|runtime|algorithm|python|javascript)\b", lower):
        score += 0.015
    if any(symbol in text for symbol in ("∫", "∑", "∂", "√", "≤", "≥", "∈")):
        score += 0.015
    if re.search(r"\b(image|figure|diagram|chart|visual|table)\b", lower):
        score += 0.010
    if 200 <= len(text) <= 4000:
        score += 0.010
    return score


def acquisition_function(input: dict) -> float:
    item_content = input.get("item_content")
    if model is not None and hasattr(model, "_raw_prediction_parts"):
        try:
            p, item_factors, _ = model._raw_prediction_parts(input)
            base = model._base_prediction(input)
            disagreement = abs(p - base)
            norm2 = sum(float(x) * float(x) for x in (item_factors or []))
            return float(disagreement + 0.25 * p * (1.0 - p) * (0.5 + norm2) + _heuristic_diversity(item_content))
        except Exception:
            pass
    jitter = 0.01 * _stable_uniform(input.get("benchmark"), input.get("condition"), item_content)
    return float(0.24 + _heuristic_diversity(item_content) + jitter)
