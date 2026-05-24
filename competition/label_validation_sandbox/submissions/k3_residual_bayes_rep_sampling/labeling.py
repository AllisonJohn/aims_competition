"""Representative-ish teacher sampling for residual benchmark Bayes.

Benchmark residual Bayes wants labels that are not too selection-biased. This
keeps deterministic random sampling as the main signal, with a small preference
for teacher probabilities away from extreme 0/1 tails.
"""

from __future__ import annotations

import hashlib

try:
    import model
except Exception:
    model = None


def _stable_uniform(*parts: object) -> float:
    text = "\x1f".join(str(part or "") for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def _teacher_prediction(input: dict) -> float:
    if model is not None and hasattr(model, "_raw_predict"):
        try:
            return float(model._raw_predict(input))
        except Exception:
            pass
    return 0.5


def acquisition_function(input: dict) -> float:
    p = max(0.001, min(0.999, _teacher_prediction(input)))
    random_score = _stable_uniform(
        input.get("benchmark"),
        input.get("condition"),
        input.get("subject_content"),
        input.get("item_content"),
    )
    tail_penalty = abs(p - 0.5)
    return float(random_score - 0.18 * tail_penalty)
