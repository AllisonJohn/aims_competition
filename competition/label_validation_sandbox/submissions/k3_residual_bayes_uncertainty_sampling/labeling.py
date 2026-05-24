"""Thompson/Fisher-ish teacher acquisition for residual benchmark Bayes.

Scores are high for examples whose labels should be informative about a
benchmark logit offset: high Bernoulli variance under the teacher, plus a small
deterministic random perturbation.
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
    info = p * (1.0 - p)
    noise = _stable_uniform(
        "thompson",
        input.get("benchmark"),
        input.get("condition"),
        input.get("subject_content"),
        input.get("item_content"),
    )
    return float(info + 0.035 * noise)
