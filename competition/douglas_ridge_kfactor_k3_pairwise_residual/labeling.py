"""Adaptive label acquisition for online logit-shift calibration."""

from __future__ import annotations

import hashlib
import json
import math
import re
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


def _clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
    return max(lo, min(hi, float(value)))


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
        score += 0.08
    if any(symbol in text for symbol in ("∫", "∑", "∂", "√", "≤", "≥", "∈")):
        score += 0.05
    if re.search(r"\b(prove|derive|justify|counterexample|debug|algorithm|function)\b", lower):
        score += 0.05
    if re.search(r"\b(image|figure|diagram|chart|table|visual)\b", lower):
        score += 0.03
    if length > 5000:
        score -= 0.12
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


def acquisition_function(input: dict) -> float:
    subject = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    benchmark = _key(input.get("benchmark"))

    subject_rate, subject_count = _smooth(SUBJECT_RATES.get(subject), 250.0)
    condition_rate, _ = _smooth(CONDITION_RATES.get(condition), 2000.0)
    benchmark_rate, _ = _smooth(BENCHMARK_RATES.get(benchmark), 5000.0)
    p = _clamp(0.80 * subject_rate + 0.10 * condition_rate + 0.10 * benchmark_rate)
    fisher = p * (1.0 - p)
    reliability = math.sqrt(subject_count / (subject_count + 250.0)) if subject_count > 0 else 0.25
    score = fisher * (0.75 + 0.25 * reliability)
    score += _item_complexity_score(input)
    score += 1e-4 * _jitter(input)
    return float(score)
