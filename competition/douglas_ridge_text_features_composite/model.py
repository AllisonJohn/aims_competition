"""Composite handcrafted text features + centered IRT difficulty correction submission.

Base predictor: smoothed subject/condition baseline.
Text correction: 73 deterministic item text features plus pairwise interactions
-> centered IRT item difficulty ridge model.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
with (ROOT / "artifacts" / "baseline_stats.json").open("r", encoding="utf-8") as f:
    STATS = json.load(f)
with (ROOT / "artifacts" / "bge_irt_ridge_artifact.json").open("r", encoding="utf-8") as f:
    ARTIFACT = json.load(f)

GLOBAL_RATE = float(STATS["global_rate"])
SUBJECT_RATES = STATS["subject_rates"]
CONDITION_RATES = STATS["condition_rates"]
BENCHMARK_RATES = STATS["benchmark_rates"]

SUBJECT_ALPHA = 250.0
CONDITION_ALPHA = 2000.0
BENCHMARK_ALPHA = 5000.0

MODEL_ID = ARTIFACT["encoder_id"]
BLEND_WEIGHT = float(ARTIFACT["blend_weight"])
DIFFICULTY_CAP = float(ARTIFACT.get("difficulty_cap", 2.5))
RIDGE_HEADS = ARTIFACT.get("ridge_heads")
if RIDGE_HEADS:
    RIDGE_HEADS = [
        {
            "intercept": float(head["ridge_intercept"]),
            "coef": [float(x) for x in head["ridge_coef"]],
            "alpha": float(head.get("ridge_alpha", 0.0)),
            "seed": int(head.get("seed", -1)),
        }
        for head in RIDGE_HEADS
    ]
else:
    RIDGE_HEADS = [
        {
            "intercept": float(ARTIFACT["ridge_intercept"]),
            "coef": [float(x) for x in ARTIFACT["ridge_coef"]],
            "alpha": float(ARTIFACT.get("ridge_alpha", 0.0)),
            "seed": int(ARTIFACT.get("seed", -1)),
        }
    ]
SCALER_MEAN = [float(x) for x in ARTIFACT["scaler_mean"]]
SCALER_SCALE = [float(x) if float(x) != 0.0 else 1.0 for x in ARTIFACT["scaler_scale"]]
BASE_FEATURE_DIM = int(ARTIFACT.get("base_feature_dim", 73))
FEATURE_DIM = int(ARTIFACT.get("feature_dim", BASE_FEATURE_DIM))
USE_PAIRWISE_INTERACTIONS = bool(ARTIFACT.get("pairwise_interactions", False))


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


def _base_prediction(input: dict) -> float:
    subject_name = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    benchmark = _key(input.get("benchmark"))
    subject_rate = _smooth(SUBJECT_RATES.get(subject_name), SUBJECT_ALPHA)
    condition_rate = _smooth(CONDITION_RATES.get(condition), CONDITION_ALPHA)
    benchmark_rate = _smooth(BENCHMARK_RATES.get(benchmark), BENCHMARK_ALPHA)
    z = (
        0.68 * _logit(subject_rate)
        + 0.22 * _logit(condition_rate)
        + 0.10 * _logit(benchmark_rate)
        + _item_adjustment(input.get("item_content"))
    )
    return _clamp(_sigmoid(z))


def _text_features(item_content: object) -> list[float]:
    text = str(item_content or "")
    lower = text.lower()
    words = re.findall(r"[A-Za-z0-9_]+", lower)
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    unique_words = set(words)
    char_count = len(text)
    word_count = len(words)

    def count_any(values: tuple[str, ...]) -> int:
        return sum(lower.count(value) for value in values)

    def has_any(values: tuple[str, ...]) -> float:
        return float(any(value in lower for value in values))

    def bounded(value: float, scale: float) -> float:
        return min(float(value), scale) / scale

    math_symbols = ("∫", "∑", "∂", "√", "≤", "≥", "∈", "\\frac", "\\sum", "\\int")
    operator_count = sum(text.count(symbol) for symbol in "+-*/=<>")
    variable_pattern_count = len(re.findall(r"\b[a-z]\d*\b", lower))
    option_count = len(re.findall(r"\b[a-d][\).]", lower))
    long_words = [word for word in words if len(word) >= 12]
    avg_word_len = (sum(len(word) for word in words) / word_count) if word_count else 0.0
    avg_sentence_words = word_count / max(1, len(sentences))
    upper_count = sum(1 for ch in text if ch.isupper())

    features = [
        bounded(char_count, 5000),
        math.log1p(char_count) / math.log1p(10000),
        bounded(word_count, 1000),
        math.log1p(word_count) / math.log1p(2000),
        bounded(avg_word_len, 20),
        bounded(text.count("\n"), 80),
        bounded(len([p for p in text.split("\n\n") if p.strip()]), 30),
        bounded(len(sentences), 100),
        bounded(text.count("?"), 20),
        bounded(text.count("!"), 20),
        bounded(sum(ch.isdigit() for ch in text), 200),
        upper_count / max(1, char_count),
        bounded(text.count(","), 80),
        bounded(text.count("."), 80),
        bounded(text.count(":"), 60),
        bounded(text.count(";"), 40),
        bounded(text.count('"') + text.count("'"), 80),
        bounded(text.count("(") + text.count(")"), 80),
        bounded(text.count("[") + text.count("]"), 50),
        bounded(text.count("{") + text.count("}"), 50),
        bounded(text.count("/") + text.count("\\"), 80),
        bounded(text.count("`"), 80),
        bounded(text.count("$"), 80),
        bounded(text.count("%"), 40),
        bounded(text.count("="), 80),
        bounded(text.count("<") + text.count(">"), 80),
        float(any(symbol in text or symbol in lower for symbol in math_symbols)),
        has_any(("prove", "proof", "derive", "justify", "counterexample", "reason")),
        has_any(("calculate", "compute", "solve", "evaluate", "simplify")),
        has_any(("theorem", "lemma", "corollary", "conjecture")),
        has_any(("triangle", "circle", "angle", "geometry", "polygon")),
        has_any(("probability", "expected", "random", "distribution", "variance")),
        has_any(("def ", "class ", "import ", "function", "algorithm", "pseudocode")),
        has_any(("bug", "debug", "error", "exception", "runtime", "traceback")),
        has_any(("sql", "database", "query", "schema", "table")),
        has_any(("bash", "shell", "terminal", "git ", "command line")),
        has_any(("api", "http", "request", "server", "endpoint")),
        has_any(("security", "attack", "injection", "malicious", "vulnerability")),
        has_any(("tool", "agent", "browser", "environment", "action")),
        has_any(("patient", "diagnosis", "treatment", "clinical", "medical")),
        has_any(("biology", "cell", "protein", "gene", "organism")),
        has_any(("chemistry", "molecule", "reaction", "compound", "acid")),
        has_any(("physics", "force", "energy", "velocity", "quantum")),
        has_any(("image", "figure", "diagram", "visual", "picture")),
        has_any(("chart", "graph", "plot", "table", "axis")),
        has_any(("a)", "b)", "c)", "d)", "multiple choice", "choose the best")),
        has_any(("preference", "ranking", "better response", "judge", "rubric")),
        has_any(("safe", "unsafe", "refuse", "harmful", "policy")),
        has_any(("reward", "feedback", "helpfulness", "honesty", "truthfulness")),
        has_any(("repository", "issue", "pull request", "patch", "test file")),
        has_any(("instruction", "follow", "constraint", "must", "should")),
        float("```" in text or "`" in text),
        float(bool(re.search(r"(^|\n)\s*[-*•]\s+", text))),
        float(bool(re.search(r"(^|\n)\s*\d+[\).]\s+", text))),
        float("```" in text),
        float("$" in text or "\\(" in text or "\\[" in text),
        float("http://" in lower or "https://" in lower or "www." in lower),
        float(any(marker in lower for marker in ("{", "}", "<xml", "</", "json"))),
        has_any(("traceback", "stack trace", "line ", "syntaxerror", "typeerror")),
        float(bool(re.search(r"[/\\][\w.-]+[/\\]", text))),
        float(bool(re.search(r"\w+\([^)]*\)", text))),
        float(char_count > 2500),
        float(count_any(("part a", "part b", "subproblem", "step 1", "step 2")) > 0),
        len(unique_words) / max(1, word_count),
        len(long_words) / max(1, word_count),
        bounded(max([len(word) for word in words], default=0), 50),
        bounded(avg_sentence_words, 80),
        sum(ch.isdigit() for ch in text) / max(1, char_count),
        bounded(operator_count, 100),
        bounded(variable_pattern_count, 100),
        bounded(option_count, 10),
        text.count("\n") / max(1, char_count),
        bounded(count_any(("because", "therefore", "however", "although", "unless")), 30),
    ]
    assert len(features) == BASE_FEATURE_DIM
    return features


def _composite_features(base_features: list[float]) -> list[float]:
    if not USE_PAIRWISE_INTERACTIONS:
        return base_features
    interactions = []
    for i, left in enumerate(base_features):
        for right in base_features[i:]:
            interactions.append(left * right)
    return base_features + interactions


def _predict_centered_difficulty(item_content: object) -> float | None:
    features = _text_features(item_content)
    if len(features) != FEATURE_DIM:
        return None
    features = [
        (value - SCALER_MEAN[i]) / SCALER_SCALE[i]
        for i, value in enumerate(features)
    ]
    features = _composite_features(features)
    if len(features) != FEATURE_DIM:
        return None
    predictions = [
        head["intercept"] + sum(coef * value for coef, value in zip(head["coef"], features))
        for head in RIDGE_HEADS
    ]
    return sum(predictions) / len(predictions)


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
    base = _base_prediction(input)
    difficulty = _predict_centered_difficulty(input.get("item_content"))
    if difficulty is None:
        prediction = base
    else:
        difficulty = max(-DIFFICULTY_CAP, min(DIFFICULTY_CAP, difficulty))
        prediction = _sigmoid(_logit(base) - BLEND_WEIGHT * difficulty)
    prediction = _calibrate_with_labeled(prediction, input, labeled)
    return float(_clamp(prediction))
