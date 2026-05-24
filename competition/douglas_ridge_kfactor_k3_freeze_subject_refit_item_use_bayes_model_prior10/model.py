"""BGE-large + K-factor IRT ridge submission.

Base predictor: smoothed subject/condition/benchmark baseline.
K-factor correction: BGE-large item text -> predicted item factor vector/bias,
combined with the learned subject/model factor vector from the K-dimensional IRT fit.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
with (ROOT / "artifacts" / "baseline_stats.json").open("r", encoding="utf-8") as f:
    STATS = json.load(f)
with (ROOT / "artifacts" / "bge_kfactor_ridge_artifact.json").open("r", encoding="utf-8") as f:
    ARTIFACT = json.load(f)

GLOBAL_RATE = float(STATS["global_rate"])
SUBJECT_RATES = STATS["subject_rates"]
CONDITION_RATES = STATS["condition_rates"]
BENCHMARK_RATES = STATS["benchmark_rates"]

SUBJECT_ALPHA = 250.0
CONDITION_ALPHA = 2000.0
BENCHMARK_ALPHA = 5000.0

MODEL_ID = ARTIFACT["encoder_id"]
LATENT_DIM = int(ARTIFACT["latent_dim"])
BLEND_WEIGHT = float(ARTIFACT["blend_weight"])
LOGIT_CAP = float(ARTIFACT.get("logit_cap", 4.0))
CALIBRATOR = ARTIFACT.get("calibrator")
BENCHMARK_BAYES_PRIOR_COUNT = 10.0
SUBJECT_FACTORS = {
    str(key).strip().lower(): [float(x) for x in value]
    for key, value in ARTIFACT["subject_factors"].items()
}
GLOBAL_SUBJECT_FACTOR = [float(x) for x in ARTIFACT.get("global_subject_factor", [0.0] * LATENT_DIM)]
RIDGE_HEADS = ARTIFACT["ridge_heads"]
SCALER_MEAN = [float(x) for x in ARTIFACT["scaler_mean"]]
SCALER_SCALE = [float(x) if float(x) != 0.0 else 1.0 for x in ARTIFACT["scaler_scale"]]
EMBEDDING_DIM = int(ARTIFACT["embedding_dim"])
MAX_LENGTH = int(ARTIFACT.get("max_length", 256))

TOKENIZER = None
ENCODER = None
TORCH = None
DEVICE = "cpu"
EMBED_CACHE: dict[str, list[float]] = {}


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


def _numeric_features(item_content: object) -> list[float]:
    text = str(item_content or "")
    lower = text.lower()
    return [
        min(len(text), 5000) / 5000.0,
        min(len(text.split()), 1000) / 1000.0,
        min(text.count("\n"), 40) / 40.0,
        float(any(s in text for s in ("∫", "∑", "∂", "√", "≤", "≥", "∈"))),
        float(any(w in lower for w in ("prove", "derive", "justify", "counterexample"))),
        float(any(w in lower for w in ("def ", "class ", "import ", "function", "debug", "algorithm"))),
        float(any(w in lower for w in ("image", "figure", "diagram", "chart", "visual"))),
        float(any(w in lower for w in ("a)", "b)", "c)", "d)", "multiple choice", "choose the best"))),
    ]


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
        print(f"[douglas_ridge_kfactor] Could not load encoder: {exc}", flush=True)
        TOKENIZER = None
        ENCODER = None
        return False


def _embed_item(item_content: object) -> list[float] | None:
    text = str(item_content or "")
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]
    if not _ensure_encoder():
        return None
    prompt = f"Represent this evaluation question for capability-factor prediction: {text}"
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
        print(f"[douglas_ridge_kfactor] Embedding failed: {exc}", flush=True)
        return None
    EMBED_CACHE[text] = vector
    return vector


def _feature_vector(item_content: object) -> list[float] | None:
    embedding = _embed_item(item_content)
    if embedding is None or len(embedding) != EMBEDDING_DIM:
        return None
    numeric = _numeric_features(item_content)
    numeric = [
        (value - SCALER_MEAN[i]) / SCALER_SCALE[i]
        for i, value in enumerate(numeric)
    ]
    return embedding + numeric


def _predict_item_latents(item_content: object) -> tuple[list[float], float] | None:
    features = _feature_vector(item_content)
    if features is None:
        return None
    values = {}
    for head in RIDGE_HEADS:
        prediction = float(head["ridge_intercept"])
        prediction += sum(float(coef) * value for coef, value in zip(head["ridge_coef"], features))
        if head.get("target_standardized"):
            prediction = float(head.get("target_mean", 0.0)) + float(head.get("target_std", 1.0)) * prediction
        values[str(head["name"])] = prediction
    item_factors = [values.get(f"factor_{k}", 0.0) for k in range(LATENT_DIM)]
    item_bias = values.get("item_bias", 0.0)
    return item_factors, item_bias


def _subject_factor(subject_content: object) -> list[float]:
    subject_name = _parse_subject_name(subject_content)
    vector = SUBJECT_FACTORS.get(subject_name)
    if vector is None:
        return GLOBAL_SUBJECT_FACTOR
    return vector


def _calibrator_features(input: dict, factor_logit: float) -> list[float]:
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
        factor_logit,
        subject_logit * factor_logit,
        condition_logit * factor_logit,
        benchmark_logit * factor_logit,
        abs(factor_logit),
    ]


def _learned_factor_prediction(input: dict, factor_logit: float) -> float | None:
    if not CALIBRATOR:
        return None
    features = _calibrator_features(input, factor_logit)
    mean = [float(x) for x in CALIBRATOR["scaler_mean"]]
    scale = [float(x) if float(x) != 0.0 else 1.0 for x in CALIBRATOR["scaler_scale"]]
    coef = [float(x) for x in CALIBRATOR["coef"]]
    z = float(CALIBRATOR["intercept"])
    for value, mu, sigma, weight in zip(features, mean, scale, coef):
        z += weight * ((value - mu) / sigma)
    return _sigmoid(z)


def _raw_predict(input: dict) -> float:
    base = _base_prediction(input)
    latents = _predict_item_latents(input.get("item_content"))
    if latents is None:
        prediction = base
    else:
        item_factors, item_bias = latents
        subject_factors = _subject_factor(input.get("subject_content"))
        factor_logit = item_bias + sum(u * v for u, v in zip(subject_factors, item_factors))
        factor_logit = max(-LOGIT_CAP, min(LOGIT_CAP, factor_logit))
        learned_prediction = _learned_factor_prediction(input, factor_logit)
        if learned_prediction is None:
            blended_logit = (1.0 - BLEND_WEIGHT) * _logit(base) + BLEND_WEIGHT * factor_logit
            prediction = _sigmoid(blended_logit)
        else:
            prediction = learned_prediction
    return float(_clamp(prediction))


def _calibrate_with_labeled(prediction: float, input: dict, labeled: list[dict] | None) -> float:
    if not labeled:
        return prediction
    benchmark = _key(input.get("benchmark"))
    if not benchmark:
        return prediction
    labels = []
    expected = []
    for example in labeled:
        if "label" not in example:
            continue
        if _key(example.get("benchmark")) != benchmark:
            continue
        label = _clamp(float(example["label"]), 0.0, 1.0)
        labels.append(label)
        expected.append(_raw_predict(example))
    if not labels:
        return prediction

    prior_rate = _clamp(sum(expected) / len(expected))
    observed_rate = _clamp(sum(labels) / len(labels))
    prior_count = float(BENCHMARK_BAYES_PRIOR_COUNT)
    posterior_rate = _clamp(
        (prior_rate * prior_count + observed_rate * len(labels))
        / (prior_count + len(labels))
    )
    delta = _logit(posterior_rate) - _logit(prior_rate)
    delta = max(-0.35, min(0.35, delta))
    return _clamp(_sigmoid(_logit(prediction) + delta))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    prediction = _raw_predict(input)
    prediction = _calibrate_with_labeled(prediction, input, labeled)
    return float(_clamp(prediction))
