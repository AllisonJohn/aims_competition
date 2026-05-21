"""Adaptive label acquisition for K-factor online MAP calibration.

The label budget is most useful on items that are both uncertain for the
current subject and informative about the subject's latent capability vector.
For the k-factor model, the Fisher information for subject factors is

    p * (1 - p) * outer(item_loading, item_loading)

so this acquisition function scores candidates with the scalar trace:

    p * (1 - p) * ||item_loading||^2
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
try:
    from . import model as MODEL
except Exception:
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        import model as MODEL
    except Exception:
        MODEL = None

try:
    with (ROOT / "artifacts" / "baseline_stats.json").open("r", encoding="utf-8") as f:
        STATS = json.load(f)
except Exception:
    STATS = {"global_rate": 0.5, "subject_rates": {}, "condition_rates": {}, "benchmark_rates": {}}

GLOBAL_RATE = float(STATS.get("global_rate", 0.5))
SUBJECT_RATES = STATS.get("subject_rates", {})
CONDITION_RATES = STATS.get("condition_rates", {})
BENCHMARK_RATES = STATS.get("benchmark_rates", {})

SUBJECT_ALPHA = 250.0
CONDITION_ALPHA = 2000.0
BENCHMARK_ALPHA = 5000.0


def _clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
    return max(lo, min(hi, float(value)))


def _logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _smooth(rate_count: list[float] | None, alpha: float) -> tuple[float, float]:
    if not rate_count:
        return GLOBAL_RATE, 0.0
    rate, count = float(rate_count[0]), float(rate_count[1])
    return (rate * count + GLOBAL_RATE * alpha) / (count + alpha), count


def _key(value: object) -> str:
    return str(value or "").strip().lower()


def _parse_subject_name(subject_content: object) -> str:
    text = str(subject_content or "")
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().lower() == "name":
            return value.strip().lower()
    return text.strip().splitlines()[0].lower() if text.strip() else ""


def _item_complexity_score(item_content: object) -> float:
    text = str(item_content or "")
    lower = text.lower()
    length = len(text)
    score = 0.0
    if 120 <= length <= 2500:
        score += 0.02
    if any(symbol in text for symbol in ("∫", "∑", "∂", "√", "≤", "≥", "∈")):
        score += 0.01
    if re.search(r"\b(prove|derive|justify|counterexample|debug|algorithm|function)\b", lower):
        score += 0.01
    if re.search(r"\b(image|figure|diagram|chart|table|visual)\b", lower):
        score += 0.005
    if length > 5000:
        score -= 0.03
    return score


def _jitter(input: dict) -> float:
    text = "|".join(
        [
            _key(input.get("benchmark")),
            _key(input.get("condition")),
            _parse_subject_name(input.get("subject_content")),
            str(input.get("item_content", ""))[:300],
        ]
    )
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=4).digest()
    return int.from_bytes(digest, "big") / 2**32


def _baseline_probability(input: dict) -> tuple[float, float]:
    subject = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    benchmark = _key(input.get("benchmark"))
    subject_rate, subject_count = _smooth(SUBJECT_RATES.get(subject), SUBJECT_ALPHA)
    condition_rate, _ = _smooth(CONDITION_RATES.get(condition), CONDITION_ALPHA)
    benchmark_rate, _ = _smooth(BENCHMARK_RATES.get(benchmark), BENCHMARK_ALPHA)
    z = 0.80 * _logit(subject_rate) + 0.12 * _logit(condition_rate) + 0.08 * _logit(benchmark_rate)
    return _clamp(_sigmoid(z)), subject_count


def _factor_probability_and_loading(input: dict) -> tuple[float, list[float]] | None:
    if MODEL is None:
        return None
    try:
        latents = MODEL._predict_item_latents(input.get("item_content"))
        if latents is None:
            return None
        item_factors, item_bias = latents
        subject_factors = MODEL._subject_factor(input.get("subject_content"))
        logit_cap = float(getattr(MODEL, "LOGIT_CAP", 4.0))
        z = item_bias + sum(u * v for u, v in zip(subject_factors, item_factors))
        z = max(-logit_cap, min(logit_cap, z))
        return _clamp(_sigmoid(z)), [float(v) for v in item_factors]
    except Exception:
        return None


def _loading_entropy(loadings: list[float]) -> float:
    squares = [value * value for value in loadings]
    total = sum(squares)
    if total <= 1e-12 or len(squares) <= 1:
        return 0.0
    probs = [value / total for value in squares if value > 0.0]
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy / math.log(len(squares))


def acquisition_function(input: dict) -> float:
    baseline_p, subject_count = _baseline_probability(input)
    factor = _factor_probability_and_loading(input)

    if factor is None:
        p = baseline_p
        fisher = p * (1.0 - p)
        reliability = math.sqrt(subject_count / (subject_count + SUBJECT_ALPHA)) if subject_count > 0 else 0.25
        return float(fisher * (0.75 + 0.25 * reliability) + _item_complexity_score(input) + 1e-4 * _jitter(input))

    factor_p, item_factors = factor
    # A light blend keeps acquisition from chasing bad item-head extrapolations.
    p = _clamp(0.70 * factor_p + 0.30 * baseline_p)
    fisher = p * (1.0 - p)
    loading_norm_sq = sum(value * value for value in item_factors)
    loading_norm_sq = min(loading_norm_sq, 9.0)

    # Prefer items that teach the vector MAP update without overselecting only
    # one collapsed latent dimension.
    entropy_bonus = 0.80 + 0.20 * _loading_entropy(item_factors)
    subject_reliability = math.sqrt(subject_count / (subject_count + SUBJECT_ALPHA)) if subject_count > 0 else 0.25
    score = fisher * loading_norm_sq * entropy_bonus * (0.80 + 0.20 * subject_reliability)
    score += _item_complexity_score(input)
    score += 1e-4 * _jitter(input)
    return float(score)
