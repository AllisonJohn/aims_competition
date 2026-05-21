"""BGE-large ridge with structured handcrafted feature augmentation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
with (ROOT / "artifacts" / "baseline_stats.json").open("r", encoding="utf-8") as f:
    STATS = json.load(f)
with (ROOT / "artifacts" / "bge_augmented_ridge_artifact.json").open("r", encoding="utf-8") as f:
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
CALIBRATOR = ARTIFACT.get("calibrator")
RIDGE_INTERCEPT = float(ARTIFACT["ridge_intercept"])
RIDGE_COEF = [float(x) for x in ARTIFACT["ridge_coef"]]
BASE_FEATURE_MEAN = [float(x) for x in ARTIFACT["base_feature_mean"]]
BASE_FEATURE_SCALE = [float(x) if float(x) != 0.0 else 1.0 for x in ARTIFACT["base_feature_scale"]]
STRUCTURED_FEATURE_MEAN = [float(x) for x in ARTIFACT["structured_feature_mean"]]
STRUCTURED_FEATURE_SCALE = [
    float(x) if float(x) != 0.0 else 1.0 for x in ARTIFACT["structured_feature_scale"]
]
PCA_MEAN = [float(x) for x in ARTIFACT.get("pca_mean", [])]
PCA_COMPONENTS = [[float(x) for x in row] for row in ARTIFACT.get("pca_components", [])]
PCA_EXPLAINED_VARIANCE = [float(x) for x in ARTIFACT.get("pca_explained_variance", [])]
PCA_WHITEN = bool(ARTIFACT.get("pca_whiten", False))
AUGMENTED_SCALE = float(ARTIFACT.get("augmented_scale", 1.0))
EMBEDDING_DIM = int(ARTIFACT["embedding_dim"])
MAX_LENGTH = int(ARTIFACT.get("max_length", 256))

TOKENIZER = None
ENCODER = None
TORCH = None
DEVICE = "cpu"
EMBED_CACHE: dict[str, list[float]] = {}


CONTINUOUS_INDICES = list(range(0, 26)) + list(range(62, 73))
COMPLEXITY_INDICES = [0, 1, 2, 3, 7, 8, 10, 62, 63, 66, 67, 68, 69, 70, 72]
DOMAIN_INDICES = list(range(26, 51))
MATH_INDICES = [26, 27, 28, 29, 30, 31, 54]
CODE_INDICES = [32, 33, 34, 35, 36, 49, 51, 53, 56, 57, 58, 59, 60]
VISUAL_INDICES = [43, 44]
CHOICE_INDICES = [45, 70]
REASONING_INDICES = [27, 28, 29, 50, 63, 72]
FORMAT_INDICES = list(range(51, 61))


def _unique_pairs(left: list[int], right: list[int]) -> list[tuple[int, int]]:
    pairs = set()
    for i in left:
        for j in right:
            if i != j:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


STRUCTURED_INTERACTION_PAIRS = sorted(
    set(
        _unique_pairs(COMPLEXITY_INDICES, DOMAIN_INDICES)
        + _unique_pairs(MATH_INDICES, REASONING_INDICES + [0, 1, 2, 3, 10])
        + _unique_pairs(CODE_INDICES, FORMAT_INDICES + [0, 1, 2, 3, 10])
        + _unique_pairs(VISUAL_INDICES, FORMAT_INDICES + [0, 1, 2, 3])
        + _unique_pairs(CHOICE_INDICES, [0, 1, 2, 3, 7, 8, 69])
        + _unique_pairs([37, 46, 47, 48, 50], [0, 1, 2, 3, 7, 38, 50])
    )
)


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


def _bounded(value: float, scale: float) -> float:
    return min(float(value), scale) / scale


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

    math_symbols = ("∫", "∑", "∂", "√", "≤", "≥", "∈", "\\frac", "\\sum", "\\int")
    operator_count = sum(text.count(symbol) for symbol in "+-*/=<>")
    variable_pattern_count = len(re.findall(r"\b[a-z]\d*\b", lower))
    option_count = len(re.findall(r"\b[a-d][\).]", lower))
    long_words = [word for word in words if len(word) >= 12]
    avg_word_len = (sum(len(word) for word in words) / word_count) if word_count else 0.0
    avg_sentence_words = word_count / max(1, len(sentences))
    upper_count = sum(1 for ch in text if ch.isupper())

    features = [
        _bounded(char_count, 5000),
        math.log1p(char_count) / math.log1p(10000),
        _bounded(word_count, 1000),
        math.log1p(word_count) / math.log1p(2000),
        _bounded(avg_word_len, 20),
        _bounded(text.count("\n"), 80),
        _bounded(len([p for p in text.split("\n\n") if p.strip()]), 30),
        _bounded(len(sentences), 100),
        _bounded(text.count("?"), 20),
        _bounded(text.count("!"), 20),
        _bounded(sum(ch.isdigit() for ch in text), 200),
        upper_count / max(1, char_count),
        _bounded(text.count(","), 80),
        _bounded(text.count("."), 80),
        _bounded(text.count(":"), 60),
        _bounded(text.count(";"), 40),
        _bounded(text.count('"') + text.count("'"), 80),
        _bounded(text.count("(") + text.count(")"), 80),
        _bounded(text.count("[") + text.count("]"), 50),
        _bounded(text.count("{") + text.count("}"), 50),
        _bounded(text.count("/") + text.count("\\"), 80),
        _bounded(text.count("`"), 80),
        _bounded(text.count("$"), 80),
        _bounded(text.count("%"), 40),
        _bounded(text.count("="), 80),
        _bounded(text.count("<") + text.count(">"), 80),
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
        _bounded(max([len(word) for word in words], default=0), 50),
        _bounded(avg_sentence_words, 80),
        sum(ch.isdigit() for ch in text) / max(1, char_count),
        _bounded(operator_count, 100),
        _bounded(variable_pattern_count, 100),
        _bounded(option_count, 10),
        text.count("\n") / max(1, char_count),
        _bounded(count_any(("because", "therefore", "however", "although", "unless")), 30),
    ]
    assert len(features) == 73
    return features


def _expanded_features(raw: list[float], scaled: list[float]) -> list[float]:
    transformed = []
    for index in CONTINUOUS_INDICES:
        raw_value = max(0.0, raw[index])
        scaled_value = scaled[index]
        transformed.append(scaled_value * scaled_value)
        transformed.append(math.sqrt(raw_value))
        transformed.append(math.log1p(10.0 * raw_value) / math.log1p(10.0))
    interactions = [scaled[i] * scaled[j] for i, j in STRUCTURED_INTERACTION_PAIRS]
    return scaled + transformed + interactions


def _ensure_encoder() -> bool:
    global TOKENIZER, ENCODER, TORCH, DEVICE
    if TOKENIZER is not None and ENCODER is not None:
        return True
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        TORCH = torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        ENCODER = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).to(DEVICE)
        ENCODER.eval()
        return True
    except Exception as exc:
        print(f"[douglas_ridge_data_augmentation] Could not load encoder: {exc}", flush=True)
        TOKENIZER = None
        ENCODER = None
        return False


def _embed_item(item_content: object) -> list[float] | None:
    text = str(item_content or "")
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]
    if not _ensure_encoder():
        return None
    prompt = f"Represent this evaluation question for difficulty prediction: {text}"
    try:
        with TORCH.no_grad():
            encoded = TOKENIZER(
                [prompt],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            output = ENCODER(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = TORCH.nn.functional.normalize(pooled, p=2, dim=1)
            vector = pooled[0].detach().cpu().tolist()
    except Exception as exc:
        print(f"[douglas_ridge_data_augmentation] Embedding failed: {exc}", flush=True)
        return None
    EMBED_CACHE[text] = vector
    return vector


def _augmented_features(item_content: object) -> list[float]:
    raw = _text_features(item_content)
    base_scaled = [
        (value - BASE_FEATURE_MEAN[i]) / BASE_FEATURE_SCALE[i]
        for i, value in enumerate(raw)
    ]
    structured = _expanded_features(raw, base_scaled)
    structured_scaled = [
        (value - STRUCTURED_FEATURE_MEAN[i]) / STRUCTURED_FEATURE_SCALE[i]
        for i, value in enumerate(structured)
    ]
    if PCA_COMPONENTS:
        centered = [value - PCA_MEAN[i] for i, value in enumerate(structured_scaled)]
        projected = []
        for component_index, component in enumerate(PCA_COMPONENTS):
            value = sum(weight * feature for weight, feature in zip(component, centered))
            if PCA_WHITEN:
                variance = max(PCA_EXPLAINED_VARIANCE[component_index], 1e-12)
                value /= math.sqrt(variance)
            projected.append(value)
        structured_scaled = projected
    return [AUGMENTED_SCALE * value for value in structured_scaled]


def _predict_centered_difficulty(item_content: object) -> float | None:
    embedding = _embed_item(item_content)
    if embedding is None or len(embedding) != EMBEDDING_DIM:
        return None
    features = embedding + _augmented_features(item_content)
    return RIDGE_INTERCEPT + sum(coef * value for coef, value in zip(RIDGE_COEF, features))


def _calibrator_features(input: dict, difficulty: float) -> list[float]:
    subject_name = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    benchmark = _key(input.get("benchmark"))
    subject_rate = _smooth(SUBJECT_RATES.get(subject_name), SUBJECT_ALPHA)
    condition_rate = _smooth(CONDITION_RATES.get(condition), CONDITION_ALPHA)
    benchmark_rate = _smooth(BENCHMARK_RATES.get(benchmark), BENCHMARK_ALPHA)
    subject_logit = _logit(subject_rate)
    condition_logit = _logit(condition_rate)
    benchmark_logit = _logit(benchmark_rate)
    adjustment = _item_adjustment(input.get("item_content"))
    return [
        subject_logit,
        condition_logit,
        benchmark_logit,
        adjustment,
        difficulty,
        subject_logit * difficulty,
        condition_logit * difficulty,
        benchmark_logit * difficulty,
        abs(difficulty),
    ]


def _learned_skip_prediction(input: dict, difficulty: float) -> float | None:
    if not CALIBRATOR:
        return None
    features = _calibrator_features(input, difficulty)
    mean = [float(x) for x in CALIBRATOR["scaler_mean"]]
    scale = [float(x) if float(x) != 0.0 else 1.0 for x in CALIBRATOR["scaler_scale"]]
    coef = [float(x) for x in CALIBRATOR["coef"]]
    z = float(CALIBRATOR["intercept"])
    for value, mu, sigma, weight in zip(features, mean, scale, coef):
        z += weight * ((value - mu) / sigma)
    return _sigmoid(z)


def _uncalibrated_prediction(input: dict) -> float:
    base = _base_prediction(input)
    difficulty = _predict_centered_difficulty(input.get("item_content"))
    if difficulty is None:
        return base
    difficulty = max(-DIFFICULTY_CAP, min(DIFFICULTY_CAP, difficulty))
    learned_prediction = _learned_skip_prediction(input, difficulty)
    if learned_prediction is None:
        return _sigmoid(_logit(base) - BLEND_WEIGHT * difficulty)
    return learned_prediction


def _online_logit_shift(input: dict, labeled: list[dict] | None) -> float:
    if not labeled:
        return 0.0
    subject_name = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    observations = []
    for example in labeled:
        if "label" not in example:
            continue
        label = _clamp(float(example["label"]), 0.0, 1.0)
        probability = _uncalibrated_prediction(example)
        weight = 1.0
        if _parse_subject_name(example.get("subject_content")) == subject_name:
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


def _calibrate_with_labeled(prediction: float, input: dict, labeled: list[dict] | None) -> float:
    delta = _online_logit_shift(input, labeled)
    if delta == 0.0:
        return prediction
    return _clamp(_sigmoid(_logit(prediction) + delta))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    prediction = _uncalibrated_prediction(input)
    prediction = _calibrate_with_labeled(prediction, input, labeled)
    return float(_clamp(prediction))
