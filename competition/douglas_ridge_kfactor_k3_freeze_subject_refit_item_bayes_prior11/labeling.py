"""Adaptive label acquisition for the k3 10-epoch ridge submission.

The platform reveals the top-K labels per hidden data category. We do not see the
category, so this scorer favors examples that are both informative for log-loss
calibration and diverse across subjects, conditions, and coarse item text.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
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

_SEEN_SUBJECTS: Counter[str] = Counter()
_SEEN_CONDITIONS: Counter[str] = Counter()
_SEEN_BENCHMARKS: Counter[str] = Counter()
_SEEN_ITEM_BUCKETS: Counter[str] = Counter()


def _clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
    return max(lo, min(hi, float(value)))


def _logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
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


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def _item_bucket(item_content: object) -> str:
    text = str(item_content or "")
    lower = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", lower)
    signature = " ".join(tokens[:80])
    length_bucket = min(len(text) // 500, 10)
    flags = [
        "math" if any(s in text for s in ("∫", "∑", "∂", "√", "≤", "≥", "∈")) else "",
        "code" if re.search(r"\b(def|class|import|debug|runtime|algorithm)\b", lower) else "",
        "proof" if re.search(r"\b(prove|derive|justify|counterexample)\b", lower) else "",
        "vision" if re.search(r"\b(image|figure|diagram|chart|visual|table)\b", lower) else "",
        "mcq" if re.search(r"\b(a\)|b\)|c\)|d\)|multiple choice|choose the best)\b", lower) else "",
    ]
    raw = f"{length_bucket}|{'|'.join(flag for flag in flags if flag)}|{signature}"
    return hashlib.blake2b(raw.encode("utf-8", errors="ignore"), digest_size=6).hexdigest()


def _item_complexity(item_content: object) -> float:
    text = str(item_content or "")
    lower = text.lower()
    length = len(text)
    words = len(text.split())
    score = 0.0
    if 80 <= length <= 3500:
        score += 0.05
    if 3500 < length <= 8000:
        score += 0.02
    if words > 80:
        score += 0.025
    if text.count("\n") > 8:
        score += 0.015
    if any(s in text for s in ("∫", "∑", "∂", "√", "≤", "≥", "∈")):
        score += 0.035
    if re.search(r"\b(prove|derive|justify|counterexample)\b", lower):
        score += 0.035
    if re.search(r"\b(def|class|import|debug|runtime|algorithm)\b", lower):
        score += 0.03
    if re.search(r"\b(image|figure|diagram|chart|visual|table)\b", lower):
        score += 0.02
    if length > 12000:
        score -= 0.08
    return score


def _prior_probability(subject: str, condition: str, benchmark: str, item_content: object) -> float:
    subject_rate, subject_count = _smooth(SUBJECT_RATES.get(subject), SUBJECT_ALPHA)
    condition_lookup = CONDITION_RATES.get(condition)
    condition_rate, condition_count = _smooth(condition_lookup, CONDITION_ALPHA)

    # Hidden benchmarks are generally unseen, so benchmark base rates are not
    # reliable acquisition evidence. Use subject ability heavily, and include
    # condition only when it has public support.
    if condition_count > 0:
        z = 0.82 * _logit(subject_rate) + 0.18 * _logit(condition_rate)
    else:
        z = 0.85 * _logit(subject_rate) + 0.15 * _logit(GLOBAL_RATE)
    z -= 0.10 * _item_complexity(item_content)
    if subject_count < 100:
        z = 0.85 * z + 0.15 * _logit(GLOBAL_RATE)
    return _clamp(_sigmoid(z))


def acquisition_function(input: dict) -> float:
    subject = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    benchmark = _key(input.get("benchmark"))
    item_content = input.get("item_content")
    bucket = _item_bucket(item_content)

    p = _prior_probability(subject, condition, benchmark, item_content)
    uncertainty = p * (1.0 - p)

    # Prefer labels that can calibrate the current round broadly. Since the same
    # labeled list is passed to every predict() call, repeated subjects/items have
    # diminishing returns.
    diversity_penalty = (
        0.030 * _SEEN_SUBJECTS[subject]
        + 0.018 * _SEEN_CONDITIONS[condition]
        + 0.012 * _SEEN_BENCHMARKS[benchmark]
        + 0.045 * _SEEN_ITEM_BUCKETS[bucket]
    )

    score = uncertainty
    score += _item_complexity(item_content)
    score -= diversity_penalty
    score += 1e-6 * _stable_unit_interval(
        "|".join([benchmark, condition, subject, bucket, str(item_content)[:200]])
    )

    _SEEN_SUBJECTS[subject] += 1
    _SEEN_CONDITIONS[condition] += 1
    _SEEN_BENCHMARKS[benchmark] += 1
    _SEEN_ITEM_BUCKETS[bucket] += 1

    if not math.isfinite(score):
        return 0.0
    return float(score)
