"""Learned hierarchical-logistic prior submission."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
with (ROOT / "artifacts" / "hierarchical_logistic_artifact.json").open("r", encoding="utf-8") as f:
    ARTIFACT = json.load(f)

GLOBAL_RATE = float(ARTIFACT["global_rate"])
SUBJECT_RATES = ARTIFACT["subject_rates"]
CONDITION_RATES = ARTIFACT["condition_rates"]
BENCHMARK_RATES = ARTIFACT["benchmark_rates"]
BUCKET_RATES = ARTIFACT["bucket_rates"]
SUBJECT_BUCKET_RATES = ARTIFACT["subject_bucket_rates"]
BENCHMARK_BUCKETS = ARTIFACT["benchmark_buckets"]
BUCKET_KEYWORDS = ARTIFACT["bucket_keywords"]
BUCKET_NAMES = ARTIFACT["bucket_names"]
ALPHAS = ARTIFACT["alphas"]
SCALER_MEAN = [float(x) for x in ARTIFACT["scaler_mean"]]
SCALER_SCALE = [float(x) if float(x) != 0.0 else 1.0 for x in ARTIFACT["scaler_scale"]]
COEF = [float(x) for x in ARTIFACT["coef"]]
INTERCEPT = float(ARTIFACT["intercept"])


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


def _bucket_for(benchmark: object, item_content: object) -> str:
    benchmark_key = _key(benchmark)
    if benchmark_key in BENCHMARK_BUCKETS:
        return BENCHMARK_BUCKETS[benchmark_key]
    lower = str(item_content or "").lower()
    scores = {
        name: sum(1 for keyword in keywords if keyword in lower)
        for name, keywords in BUCKET_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general_reasoning"


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


def _text_features(item_content: object) -> list[float]:
    text = str(item_content or "")
    lower = text.lower()
    words = re.findall(r"[A-Za-z0-9_]+", lower)
    char_count = len(text)
    word_count = len(words)

    def bounded(value: float, scale: float) -> float:
        return min(float(value), scale) / scale

    def has_any(values: tuple[str, ...]) -> float:
        return float(any(value in lower for value in values))

    return [
        bounded(char_count, 5000),
        math.log1p(char_count) / math.log1p(10000),
        bounded(word_count, 1000),
        math.log1p(word_count) / math.log1p(2000),
        bounded(text.count("\n"), 80),
        bounded(text.count("?"), 20),
        bounded(sum(ch.isdigit() for ch in text), 200),
        bounded(text.count("(") + text.count(")"), 80),
        bounded(text.count("`"), 80),
        bounded(text.count("="), 80),
        float(any(symbol in text for symbol in ("∫", "∑", "∂", "√", "≤", "≥", "∈"))),
        has_any(("prove", "proof", "derive", "justify", "counterexample", "reason")),
        has_any(("calculate", "compute", "solve", "evaluate", "simplify")),
        has_any(("def ", "class ", "import ", "function", "algorithm", "pseudocode")),
        has_any(("bug", "debug", "error", "exception", "runtime", "traceback")),
        has_any(("api", "http", "request", "server", "endpoint")),
        has_any(("security", "attack", "injection", "malicious", "vulnerability")),
        has_any(("tool", "agent", "browser", "environment", "action")),
        has_any(("patient", "diagnosis", "treatment", "clinical", "medical")),
        has_any(("image", "figure", "diagram", "visual", "picture")),
        has_any(("chart", "graph", "plot", "table", "axis")),
        has_any(("a)", "b)", "c)", "d)", "multiple choice", "choose the best")),
        has_any(("preference", "ranking", "better response", "judge", "rubric")),
        has_any(("repository", "issue", "pull request", "patch", "test file")),
    ]


def _features(input: dict) -> list[float]:
    subject_name = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    benchmark = _key(input.get("benchmark"))
    bucket = _bucket_for(benchmark, input.get("item_content"))
    subject_rate = _smooth(SUBJECT_RATES.get(subject_name), float(ALPHAS["subject"]))
    condition_rate = _smooth(CONDITION_RATES.get(condition), float(ALPHAS["condition"]))
    benchmark_rate = _smooth(BENCHMARK_RATES.get(benchmark), float(ALPHAS["benchmark"]))
    bucket_rate = _smooth(BUCKET_RATES.get(bucket), float(ALPHAS["bucket"]))
    subject_bucket_rate = _smooth(
        SUBJECT_BUCKET_RATES.get(f"{subject_name}||{bucket}"),
        float(ALPHAS["bucket"]),
        prior=subject_rate,
    )
    values = [
        _logit(subject_rate),
        _logit(condition_rate),
        _logit(benchmark_rate),
        _logit(bucket_rate),
        _logit(subject_bucket_rate) - _logit(bucket_rate),
        _item_adjustment(input.get("item_content")),
    ]
    values += [float(bucket == name) for name in BUCKET_NAMES]
    values += _text_features(input.get("item_content"))
    return [(value - SCALER_MEAN[i]) / SCALER_SCALE[i] for i, value in enumerate(values)]


def _calibrate_with_labeled(prediction: float, input: dict, labeled: list[dict] | None) -> float:
    if not labeled:
        return prediction
    subject_name = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    same_subject = []
    same_condition = []
    all_labels = []
    for example in labeled:
        if "label" not in example:
            continue
        label = _clamp(float(example["label"]), 0.0, 1.0)
        all_labels.append(label)
        if _parse_subject_name(example.get("subject_content")) == subject_name:
            same_subject.append(label)
        if _key(example.get("condition") or "none") == condition:
            same_condition.append(label)
    if same_subject:
        labels = same_subject
        weight = min(0.35, 0.12 + 0.06 * len(labels))
    elif same_condition:
        labels = same_condition
        weight = min(0.25, 0.08 + 0.04 * len(labels))
    elif all_labels:
        labels = all_labels
        weight = min(0.18, 0.06 + 0.02 * len(labels))
    else:
        return prediction
    observed = sum(labels) / len(labels)
    return _clamp((1.0 - weight) * prediction + weight * observed)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    x = _features(input)
    z = INTERCEPT + sum(weight * value for weight, value in zip(COEF, x))
    prediction = _clamp(_sigmoid(z))
    return float(_calibrate_with_labeled(prediction, input, labeled))
