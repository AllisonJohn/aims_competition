"""Diversity-friendly acquisition for local residual adaptation.

This intentionally does not use benchmark base rates. Hidden benchmark names are
cold, so the acquisition score focuses on uncertainty under subject/condition
priors plus broad item-family coverage.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
with (ROOT / "artifacts" / "baseline_stats.json").open("r", encoding="utf-8") as f:
    STATS = json.load(f)

GLOBAL_RATE = float(STATS["global_rate"])
SUBJECT_RATES = STATS["subject_rates"]
CONDITION_RATES = STATS["condition_rates"]
SUBJECT_ALPHA = 250.0
CONDITION_ALPHA = 2000.0


def _clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
    return max(lo, min(hi, float(value)))


def _logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _smooth(rate_count: list[float] | None, alpha: float) -> float:
    if not rate_count:
        return GLOBAL_RATE
    rate, count = float(rate_count[0]), float(rate_count[1])
    return (rate * count + GLOBAL_RATE * alpha) / (count + alpha)


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


def _stable_uniform(*parts: object) -> float:
    text = "\x1f".join(str(part or "") for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def _item_flags(item_content: object) -> tuple[str, ...]:
    text = str(item_content or "")
    lower = text.lower()
    flags = []
    if re.search(r"\b(prove|derive|justify|counterexample|theorem)\b", lower):
        flags.append("proof")
    if re.search(r"\b(def|class|import|function|debug|runtime|algorithm|python|javascript)\b", lower):
        flags.append("code")
    if any(symbol in text for symbol in ("∫", "∑", "∂", "√", "≤", "≥", "∈")) or re.search(
        r"\b(equation|calculate|solve|integer|probability)\b", lower
    ):
        flags.append("math")
    if re.search(r"\b(image|figure|diagram|chart|visual|table)\b", lower):
        flags.append("vision")
    if re.search(r"\b(a\)|b\)|c\)|d\)|multiple choice|choose the best)\b", lower):
        flags.append("mcq")
    if len(text) > 2500:
        flags.append("long")
    return tuple(flags or ["plain"])


def _item_family_score(item_content: object) -> float:
    text = str(item_content or "")
    lower = text.lower()
    score = 0.0
    length = len(text)
    if 120 <= length <= 4000:
        score += 0.025
    elif length > 9000:
        score -= 0.060
    flags = _item_flags(text)
    score += 0.018 * min(3, len(flags))
    if "proof" in flags or "code" in flags or "math" in flags:
        score += 0.020
    if lower.count("\n") > 8:
        score += 0.010
    return score


def acquisition_function(input: dict) -> float:
    subject = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    subject_rate = _smooth(SUBJECT_RATES.get(subject), SUBJECT_ALPHA)
    condition_rate = _smooth(CONDITION_RATES.get(condition), CONDITION_ALPHA)
    z = 0.82 * _logit(subject_rate) + 0.18 * _logit(condition_rate)
    p = _sigmoid(z)
    uncertainty = p * (1.0 - p)
    text = str(input.get("item_content") or "")
    flags = "|".join(_item_flags(text))
    jitter = 0.020 * _stable_uniform(flags, input.get("benchmark"), input.get("item_content"))
    return float(uncertainty + _item_family_score(text) + jitter)
