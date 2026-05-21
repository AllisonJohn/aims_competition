"""Direct BGE + L2 logistic submission.

This variant gives up the intermediate IRT/Rasch difficulty target. It embeds
the item text, appends subject/condition priors and simple text features, then
uses one regularized linear logistic model to predict correctness directly.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
with (ROOT / "artifacts" / "baseline_stats.json").open("r", encoding="utf-8") as f:
    STATS = json.load(f)
with (ROOT / "artifacts" / "bge_direct_l2_artifact.json").open("r", encoding="utf-8") as f:
    ARTIFACT = json.load(f)

GLOBAL_RATE = float(STATS["global_rate"])
SUBJECT_RATES = STATS["subject_rates"]
CONDITION_RATES = STATS["condition_rates"]

SUBJECT_ALPHA = 250.0
CONDITION_ALPHA = 2000.0

MODEL_ID = ARTIFACT["encoder_id"]
INTERCEPT = float(ARTIFACT["intercept"])
COEF = [float(x) for x in ARTIFACT["coef"]]
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
        print(f"[direct_l2] Could not load encoder: {exc}", flush=True)
        TOKENIZER = None
        ENCODER = None
        return False


def _embed_item(item_content: object) -> list[float] | None:
    text = str(item_content or "")
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]
    if not _ensure_encoder():
        return None
    prompt = f"Represent this evaluation question for correctness prediction: {text}"
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
        print(f"[direct_l2] Embedding failed: {exc}", flush=True)
        return None
    EMBED_CACHE[text] = vector
    return vector


def _linear_features(input: dict) -> list[float] | None:
    subject_name = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    subject_rate = _smooth(SUBJECT_RATES.get(subject_name), SUBJECT_ALPHA)
    condition_rate = _smooth(CONDITION_RATES.get(condition), CONDITION_ALPHA)
    embedding = _embed_item(input.get("item_content"))
    if embedding is None or len(embedding) != EMBEDDING_DIM:
        return None
    return [
        _logit(subject_rate),
        _logit(condition_rate),
        _item_adjustment(input.get("item_content")),
        *embedding,
        *_numeric_features(input.get("item_content")),
    ]


def _base_fallback(input: dict) -> float:
    subject_name = _parse_subject_name(input.get("subject_content"))
    condition = _key(input.get("condition") or "none")
    subject_rate = _smooth(SUBJECT_RATES.get(subject_name), SUBJECT_ALPHA)
    condition_rate = _smooth(CONDITION_RATES.get(condition), CONDITION_ALPHA)
    z = 0.85 * _logit(subject_rate) + 0.15 * _logit(condition_rate)
    return _sigmoid(z)


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
    features = _linear_features(input)
    if features is None or not (len(features) == len(SCALER_MEAN) == len(SCALER_SCALE) == len(COEF)):
        prediction = _base_fallback(input)
    else:
        z = INTERCEPT
        for value, mu, sigma, weight in zip(features, SCALER_MEAN, SCALER_SCALE, COEF):
            z += weight * ((value - mu) / sigma)
        prediction = _sigmoid(z)
    prediction = _calibrate_with_labeled(prediction, input, labeled)
    return float(_clamp(prediction))
