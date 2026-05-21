"""Anchored capability interaction model.

This submission estimates model-specific strengths on fixed capability buckets
and combines them with item bucket loadings. It intentionally avoids runtime
text encoders so prediction is artifact-only and robust.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
with (ROOT / "artifacts" / "anchored_capability_artifact.json").open("r", encoding="utf-8") as f:
    ARTIFACT = json.load(f)

GLOBAL_RATE = float(ARTIFACT["global_rate"])
SUBJECT_RATES = ARTIFACT["subject_rates"]
CONDITION_RATES = ARTIFACT["condition_rates"]
BENCHMARK_RATES = ARTIFACT["benchmark_rates"]
BUCKET_NAMES = [str(value) for value in ARTIFACT["bucket_names"]]
BENCHMARK_BUCKETS = ARTIFACT["benchmark_buckets"]
BUCKET_KEYWORDS = ARTIFACT["bucket_keywords"]
CAPABILITY_OFFSETS = {
    str(subject).strip().lower(): {str(bucket): float(value) for bucket, value in offsets.items()}
    for subject, offsets in ARTIFACT["capability_offsets"].items()
}
BUCKET_DIFFICULTY_OFFSETS = {
    str(bucket): float(value) for bucket, value in ARTIFACT["bucket_difficulty_offsets"].items()
}
ALPHAS = ARTIFACT["alphas"]
CALIBRATOR = ARTIFACT["calibrator"]


def _clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
    return max(lo, min(hi, float(value)))


def _logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _key(value: object) -> str:
    return str(value or "").strip().lower()


def _smooth(rate_count: list[float] | None, alpha: float, prior: float | None = None) -> float:
    base = GLOBAL_RATE if prior is None else float(prior)
    if not rate_count:
        return base
    rate, count = float(rate_count[0]), float(rate_count[1])
    return (rate * count + base * alpha) / (count + alpha)


def _parse_subject_name(subject_content: object) -> str:
    text = str(subject_content or "")
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().lower() == "name":
            return value.strip().lower()
    return text.strip().splitlines()[0].lower() if text.strip() else ""


def _bucket_loadings(benchmark: object, item_content: object) -> dict[str, float]:
    lower = str(item_content or "").lower()
    values = {name: 0.0 for name in BUCKET_NAMES}
    benchmark_bucket = BENCHMARK_BUCKETS.get(_key(benchmark))
    if benchmark_bucket in values:
        values[benchmark_bucket] = 1.0
    for bucket, keywords in BUCKET_KEYWORDS.items():
        if bucket not in values:
            continue
        hits = sum(1 for keyword in keywords if keyword in lower)
        if hits:
            values[bucket] = max(values[bucket], min(1.0, 0.25 + 0.15 * hits))
    total = sum(values.values())
    if total <= 0.0:
        values["general_reasoning"] = 0.5
        total = 0.5
    return {key: value / total for key, value in values.items()}


def _item_adjustment(item_content: object) -> float:
    text = str(item_content or "")
    lower = text.lower()
    length = len(text)
    adjustment = 0.0
    if length > 2500:
        adjustment -= 0.07
    elif length > 1000:
        adjustment -= 0.04
    elif 0 < length < 140:
        adjustment += 0.02
    if text.count("\n") > 12 or lower.count("part ") > 1:
        adjustment -= 0.03
    if any(symbol in text for symbol in ("∫", "∑", "∂", "√", "≤", "≥", "∈")):
        adjustment -= 0.04
    if re.search(r"\b(prove|derive|justify|rigorously|counterexample)\b", lower):
        adjustment -= 0.04
    if re.search(r"\b(def|class|import|function|bug|debug|runtime|algorithm)\b", lower):
        adjustment -= 0.03
    if re.search(r"\b(what is|who is|when did|where is)\b", lower) and length < 240:
        adjustment += 0.03
    if re.search(r"\b(a\)|b\)|c\)|d\)|multiple choice|choose the best)\b", lower):
        adjustment += 0.01
    return adjustment


def _features(input: dict) -> list[float]:
    subject = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    benchmark = _key(input.get("benchmark"))
    subject_rate = _smooth(SUBJECT_RATES.get(subject), float(ALPHAS["subject"]))
    condition_rate = _smooth(CONDITION_RATES.get(condition), float(ALPHAS["condition"]))
    benchmark_rate = _smooth(BENCHMARK_RATES.get(benchmark), float(ALPHAS["benchmark"]))
    loadings = _bucket_loadings(benchmark, input.get("item_content"))
    subject_offsets = CAPABILITY_OFFSETS.get(subject, {})
    capability = sum(subject_offsets.get(bucket, 0.0) * value for bucket, value in loadings.items())
    bucket_difficulty = sum(BUCKET_DIFFICULTY_OFFSETS.get(bucket, 0.0) * value for bucket, value in loadings.items())
    values = [
        _logit(subject_rate),
        _logit(condition_rate),
        _logit(benchmark_rate),
        _item_adjustment(input.get("item_content")),
        capability,
        bucket_difficulty,
        capability * bucket_difficulty,
        abs(capability),
    ]
    values += [float(loadings[name]) for name in BUCKET_NAMES]
    mean = [float(value) for value in CALIBRATOR["scaler_mean"]]
    scale = [float(value) if float(value) != 0.0 else 1.0 for value in CALIBRATOR["scaler_scale"]]
    return [(value - mean[index]) / scale[index] for index, value in enumerate(values)]


def _uncalibrated_prediction(input: dict) -> float:
    features = _features(input)
    z = float(CALIBRATOR["intercept"])
    for value, weight in zip(features, CALIBRATOR["coef"]):
        z += float(weight) * value
    return _clamp(_sigmoid(z))


def _online_logit_shift(input: dict, labeled: list[dict] | None) -> float:
    if not labeled:
        return 0.0
    subject = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    observations = []
    for example in labeled:
        if "label" not in example:
            continue
        label = _clamp(float(example["label"]), 0.0, 1.0)
        probability = _uncalibrated_prediction(example)
        weight = 1.0
        if _parse_subject_name(example.get("subject_content")) == subject:
            weight += 0.5
        if _key(example.get("condition") or "none") == condition:
            weight += 0.25
        observations.append((_logit(probability), label, weight))
    if not observations:
        return 0.0

    delta = 0.0
    prior_variance = 0.7 * 0.7
    for _ in range(6):
        grad = delta / prior_variance
        hess = 1.0 / prior_variance
        for logit_value, label, weight in observations:
            shifted = _sigmoid(logit_value + delta)
            grad += weight * (shifted - label)
            hess += weight * shifted * (1.0 - shifted)
        if hess <= 1e-9:
            break
        delta -= grad / hess
        delta = max(-1.5, min(1.5, delta))
    return delta


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    prediction = _uncalibrated_prediction(input)
    delta = _online_logit_shift(input, labeled)
    if delta:
        prediction = _sigmoid(_logit(prediction) + delta)
    return float(_clamp(prediction))
