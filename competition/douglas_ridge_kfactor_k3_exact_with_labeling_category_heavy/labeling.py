"""Diversity-aware acquisition for the labeled residual calibrator.

The predictor uses same-benchmark labels to estimate benchmark/category/subject
residual shifts. This acquisition therefore prefers examples that are uncertain
and cover different inferred categories and subjects.
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

_SEEN_CATEGORIES: Counter[str] = Counter()
_SEEN_SUBJECTS: Counter[str] = Counter()
_SEEN_CONDITIONS: Counter[str] = Counter()
_SEEN_BUCKETS: Counter[str] = Counter()


def _clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
    return max(lo, min(hi, float(value)))


def _logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _key(value: object) -> str:
    return str(value or "").strip().lower()


def _smooth(rate_count: list[float] | None, alpha: float) -> float:
    if not rate_count:
        return GLOBAL_RATE
    rate, count = float(rate_count[0]), float(rate_count[1])
    return (rate * count + GLOBAL_RATE * alpha) / (count + alpha)


def _parse_subject_name(subject_content: object) -> str:
    text = str(subject_content or "")
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().lower() == "name":
            return value.strip().lower()
    return text.strip().splitlines()[0].lower() if text.strip() else ""


def _infer_category(benchmark: object, condition: object, item_content: object) -> str:
    b = _key(benchmark)
    c = _key(condition)
    lower = str(item_content or "").lower()
    if b == "cybench" or re.search(r"\b(cve|exploit|vulnerability|pwn|crypto|forensics|ctf|shellcode|xss|sql injection)\b", lower):
        return "cyber"
    if b in {"hle", "mmlupro"}:
        return "knowledge"
    if b in {"mathvista_mini", "matharena"} or re.search(r"\b(prove|derive|integral|theorem|geometry|algebra)\b", lower):
        return "math"
    if b in {"livecodebench", "swebench"} or re.search(r"\b(def|class|import|runtime|debug|algorithm|function|python|javascript)\b", lower):
        return "code"
    if b in {"ai2d_test", "mmbench_v11"} or re.search(r"\b(image|figure|diagram|chart|graph|visual|shown)\b", lower):
        return "visual"
    if b == "afrimedqa" or re.search(r"\b(patient|diagnosis|clinical|symptom|treatment|disease|medical)\b", lower):
        return "medical"
    if b in {"agentdojo", "bfcl", "androidworld"} or re.search(r"\b(tool|api|browser|android|calendar|email|function call|execute)\b", lower):
        return "agent_tool"
    if b in {"rewardbench", "ultrafeedback", "mtbench"} or re.search(r"\b(prefer|preference|better response|rate|rating|judge|assistant response)\b", lower + " " + c):
        return "preference"
    return "general"


def _stable_unit_interval(*parts: object) -> float:
    text = "\x1f".join(str(part or "") for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def _bucket(item_content: object) -> str:
    text = str(item_content or "")
    lower = text.lower()
    length_bucket = min(len(text) // 400, 12)
    flags = [
        "math" if re.search(r"\b(prove|derive|integral|theorem|geometry|algebra)\b", lower) else "",
        "code" if re.search(r"\b(def|class|import|runtime|debug|algorithm|function)\b", lower) else "",
        "visual" if re.search(r"\b(image|figure|diagram|chart|graph|visual|shown)\b", lower) else "",
        "mcq" if re.search(r"\b(a\)|b\)|c\)|d\)|multiple choice|choose the best)\b", lower) else "",
    ]
    raw = f"{length_bucket}|{'|'.join(flag for flag in flags if flag)}|{lower[:180]}"
    return hashlib.blake2b(raw.encode("utf-8", errors="ignore"), digest_size=6).hexdigest()


def _complexity(item_content: object) -> float:
    text = str(item_content or "")
    lower = text.lower()
    score = 0.0
    if 120 <= len(text) <= 3500:
        score += 0.025
    if re.search(r"\b(prove|derive|justify|counterexample)\b", lower):
        score += 0.018
    if re.search(r"\b(def|class|import|runtime|debug|algorithm|function)\b", lower):
        score += 0.016
    if re.search(r"\b(image|figure|diagram|chart|graph|visual|shown)\b", lower):
        score += 0.014
    if len(text) > 10000:
        score -= 0.06
    return score


def _prior_probability(subject: str, condition: str, item_content: object) -> float:
    subject_rate = _smooth(SUBJECT_RATES.get(subject), 250.0)
    condition_rate = _smooth(CONDITION_RATES.get(condition), 2000.0)
    z = 0.78 * _logit(subject_rate) + 0.22 * _logit(condition_rate)
    z -= 0.08 * _complexity(item_content)
    return _clamp(_sigmoid(z))


def acquisition_function(input: dict) -> float:
    subject = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    category = _infer_category(input.get("benchmark"), condition, input.get("item_content"))
    bucket = _bucket(input.get("item_content"))

    p = _prior_probability(subject, condition, input.get("item_content"))
    uncertainty = 4.0 * p * (1.0 - p)
    score = 0.32 * uncertainty + _complexity(input.get("item_content"))
    score -= 0.125 * _SEEN_CATEGORIES[category]
    score -= 0.060 * _SEEN_SUBJECTS[subject]
    score -= 0.020 * _SEEN_CONDITIONS[condition]
    score -= 0.055 * _SEEN_BUCKETS[bucket]
    score += 0.010 * _stable_unit_interval(input.get("benchmark"), category, subject, bucket, input.get("item_content"))

    _SEEN_CATEGORIES[category] += 1
    _SEEN_SUBJECTS[subject] += 1
    _SEEN_CONDITIONS[condition] += 1
    _SEEN_BUCKETS[bucket] += 1
    return float(score) if math.isfinite(score) else 0.0
