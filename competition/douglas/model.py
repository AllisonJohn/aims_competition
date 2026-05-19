"""Self-contained Douglas submission model.

This file intentionally includes the lightweight prediction code needed to
load ``artifacts/douglas_submit_features.pt``. Do not rely on ``modeling.py``
being present in the submitted ZIP.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import torch
from torch import nn


LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "douglas_model.pt"
SUBMIT_ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "douglas_submit_features.pt"

MISSING_TOKEN = "__missing__"
UNKNOWN_MODEL_TOKEN = "__unknown_model__"
MODEL_EMBED_DIM = 8
METADATA_FIELDS = ("model_id", "name", "organization", "size_params", "release_date", "family")
NUMERIC_MODEL_FEATURES = ("log_params", "release_date", "frontier_developer")
MODEL_FEATURE_DIM = len(METADATA_FIELDS) + len(NUMERIC_MODEL_FEATURES)
MODEL_VECTOR_DIM = 4
ITEM_FEATURE_DIM = 73
ITEM_HEAD_HIDDEN_DIM = 128
CLIP_LO = 1e-7
CLIP_HI = 1.0 - 1e-7

LEFT_MODEL_ID_KEYS = ("model_a_id", "model_id_a", "left_model_id", "subject_a_id", "subject_id_a")
RIGHT_MODEL_ID_KEYS = ("model_b_id", "model_id_b", "right_model_id", "subject_b_id", "subject_id_b")
LEFT_SUBJECT_CONTENT_KEYS = (
    "subject_a_content",
    "model_a_content",
    "left_subject_content",
    "subject_content_a",
)
RIGHT_SUBJECT_CONTENT_KEYS = (
    "subject_b_content",
    "model_b_content",
    "right_subject_content",
    "subject_content_b",
)


def _local_smoke_test_enabled() -> bool:
    value = os.environ.get(LOCAL_SMOKE_TEST_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def first_present(mapping: dict, keys: tuple[str, ...]):
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def extract_model_metadata(subject_content: str | None, model_id: str | None = None) -> dict:
    metadata = {
        "model_id": model_id,
        "name": None,
        "organization": None,
        "size_params": None,
        "release_date": None,
        "family": None,
    }
    label_to_key = {
        "name": "name",
        "organization": "organization",
        "parameters": "size_params",
        "released": "release_date",
        "family": "family",
    }
    if not subject_content:
        return metadata
    for line in subject_content.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = label_to_key.get(label.strip().lower())
        if key:
            metadata[key] = value.strip() or None
    return metadata


def is_pairwise_input(input: dict) -> bool:
    has_left = (
        first_present(input, LEFT_MODEL_ID_KEYS) is not None
        or first_present(input, LEFT_SUBJECT_CONTENT_KEYS) is not None
    )
    has_right = (
        first_present(input, RIGHT_MODEL_ID_KEYS) is not None
        or first_present(input, RIGHT_SUBJECT_CONTENT_KEYS) is not None
    )
    return has_left and has_right


def extract_left_model_metadata(input: dict) -> dict:
    subject_content = first_present(input, LEFT_SUBJECT_CONTENT_KEYS)
    model_id = first_present(input, LEFT_MODEL_ID_KEYS)
    if subject_content is None and model_id is None:
        subject_content = input.get("subject_content")
        model_id = input.get("model_id")
    return extract_model_metadata(subject_content, model_id=model_id)


def extract_right_model_metadata(input: dict) -> dict:
    return extract_model_metadata(
        first_present(input, RIGHT_SUBJECT_CONTENT_KEYS),
        model_id=first_present(input, RIGHT_MODEL_ID_KEYS),
    )


class ModelSideEncoder(nn.Module):
    def __init__(
        self,
        field_value_means: dict[str, dict[str, float]] | None = None,
        field_default_means: dict[str, float] | None = None,
        global_mean: float = 0.5,
        p: int = MODEL_EMBED_DIM,
        output_dim: int = MODEL_VECTOR_DIM,
    ) -> None:
        super().__init__()
        self.p = p
        self.output_dim = output_dim
        self.global_mean = float(global_mean)
        self.field_value_means = {
            field: dict((field_value_means or {}).get(field, {}))
            for field in METADATA_FIELDS
        }
        self.field_default_means = {
            field: float((field_default_means or {}).get(field, self.global_mean))
            for field in METADATA_FIELDS
        }
        self.model_to_index = {UNKNOWN_MODEL_TOKEN: 0}
        self.organization_to_index = {MISSING_TOKEN: 0}
        self.family_to_index = {MISSING_TOKEN: 0}
        self.name_token_to_index = {MISSING_TOKEN: 0}
        self.num_models = len(self.field_value_means["model_id"])
        self.output_projection = nn.Linear(MODEL_FEATURE_DIM, output_dim)
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    def forward(self, metadata: dict) -> torch.Tensor:
        values = [
            self.metadata_mean(field, metadata.get(field))
            for field in METADATA_FIELDS
        ]
        values.extend(numeric_model_features(metadata))
        features = torch.tensor(values, dtype=torch.float32, device=self._device_anchor.device)
        return self.output_projection(features)

    def metadata_mean(self, field: str, value) -> float:
        key = normalize_metadata_value(value)
        return self.field_value_means[field].get(key, self.field_default_means[field])


class HandcraftedItemQuestionEncoder(nn.Module):
    item_encoder_type = "handcrafted"

    def __init__(
        self,
        encoder_name: str = "handcrafted-item-features",
        loading_dim: int = MODEL_VECTOR_DIM,
        max_length: int = 256,
        hidden_dim: int = ITEM_HEAD_HIDDEN_DIM,
        freeze_backbone: bool = True,
        local_files_only: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.loading_dim = loading_dim
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.representation_dim = ITEM_FEATURE_DIM
        self.loading_head = build_item_head(self.representation_dim, loading_dim, hidden_dim=hidden_dim)
        self.bias_head = build_item_head(self.representation_dim, 1, hidden_dim=hidden_dim)

    def encode_sentence_representations(self, item_side_info: dict) -> torch.Tensor:
        return item_hardness_features(item_side_info).to(self.head_device()).unsqueeze(0)

    def forward_from_representations(
        self,
        sentence_representations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loading = torch.softmax(self.loading_head(sentence_representations), dim=-1)
        bias = self.bias_head(sentence_representations).squeeze(-1)
        return loading.mean(dim=0), bias.mean()

    def head_device(self) -> torch.device:
        return next(self.loading_head.parameters()).device


class DouglasScorer(nn.Module):
    def __init__(
        self,
        model_encoder: ModelSideEncoder,
        item_encoder: HandcraftedItemQuestionEncoder,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.model_encoder = model_encoder
        self.item_encoder = item_encoder
        self.temperature = temperature
        self.item_representation_cache: dict[str, torch.Tensor] = {}

    def score_logit(self, input: dict) -> torch.Tensor:
        item_representations = self.cached_item_representations(input)
        V_j, z_j = self.item_encoder.forward_from_representations(item_representations)
        U_left = self.model_encoder(extract_left_model_metadata(input))
        if is_pairwise_input(input):
            U_right = self.model_encoder(extract_right_model_metadata(input))
            return torch.dot(U_left - U_right, V_j) + z_j
        return torch.dot(U_left, V_j) + z_j

    def forward(self, input: dict) -> torch.Tensor:
        return self.score_logit(input) / self.temperature

    def predict_probability(self, input: dict) -> float:
        self.eval()
        with torch.no_grad():
            probability = torch.sigmoid(self.forward(input))
        return clip_probability(float(probability.detach().cpu()))

    def cached_item_representations(self, input: dict) -> torch.Tensor:
        key = "::".join(
            str(input.get(field) or "")
            for field in ("benchmark", "condition", "item_content")
        )
        if key not in self.item_representation_cache:
            with torch.no_grad():
                self.item_representation_cache[key] = (
                    self.item_encoder.encode_sentence_representations(input).detach()
                )
        return self.item_representation_cache[key]


def load_checkpoint(path: Path, map_location: str = "cpu") -> DouglasScorer:
    data = torch.load(path, map_location=map_location, weights_only=False)
    config = data.get("config", {})
    item_encoder_type = config.get("item_encoder_type", "handcrafted")
    if item_encoder_type != "handcrafted":
        raise ValueError("This self-contained model.py only supports the handcrafted features artifact.")

    k = int(config.get("k", MODEL_VECTOR_DIM))
    p = int(config.get("p", MODEL_EMBED_DIM))
    item_head_hidden_dim = int(config.get("item_head_hidden_dim", ITEM_HEAD_HIDDEN_DIM))

    model_encoder = ModelSideEncoder(
        field_value_means=data.get("field_value_means"),
        field_default_means=data.get("field_default_means"),
        global_mean=float(data.get("global_mean", 0.5)),
        p=p,
        output_dim=k,
    )
    model_encoder.load_state_dict(data["model_encoder_state_dict"])

    item_encoder = HandcraftedItemQuestionEncoder(
        loading_dim=k,
        hidden_dim=item_head_hidden_dim,
    )
    item_encoder.loading_head.load_state_dict(data["item_heads_state_dict"]["loading_head"])
    item_encoder.bias_head.load_state_dict(data["item_heads_state_dict"]["bias_head"])

    scorer = DouglasScorer(
        model_encoder=model_encoder,
        item_encoder=item_encoder,
        temperature=float(data.get("temperature", config.get("temperature", 1.0))),
    )
    scorer.eval()
    return scorer


def build_item_head(input_dim: int, output_dim: int, hidden_dim: int = ITEM_HEAD_HIDDEN_DIM) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(hidden_dim, output_dim),
    )


def item_hardness_features(item_side_info: dict) -> torch.Tensor:
    text = item_side_info.get("item_content") or ""
    lower = text.lower()
    sentences = split_sentences(text)
    words = re.findall(r"[A-Za-z0-9_]+", text)
    unique_words = {word.lower() for word in words}

    char_count = len(text)
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_len = sum(len(word) for word in words) / word_count if word_count else 0.0
    avg_sentence_words = word_count / max(sentence_count, 1)
    unique_ratio = len(unique_words) / max(word_count, 1)

    digit_count = sum(char.isdigit() for char in text)
    punctuation_count = sum(char in ".,;:!?()[]{}" for char in text)
    uppercase_count = sum(char.isupper() for char in text)
    math_symbols = "\u222b\u2211\u2202\u221a\u2264\u2265\u2260\u2248\u221e\u03c0\u03b8\u03bb\u03bc+-*/=<>^"
    math_symbol_count = sum(char in math_symbols for char in text)
    newline_count = text.count("\n")
    choice_count = count_choice_markers(text)
    code_marker_count = count_code_markers(text)
    latex_marker_count = count_latex_markers(text)
    comparison_count = len(re.findall(r"[A-Za-z0-9]\s*(=|<|>|\u2264|\u2265|\u2248|\u2260)\s*[A-Za-z0-9]", text))
    constraint_count = len(
        re.findall(
            r"\b(if|unless|except|exactly|at least|at most|minimum|maximum|must|cannot|not)\b",
            lower,
        )
    )

    features = [
        clamp01(math.log1p(char_count) / 10.0),
        clamp01(math.log1p(word_count) / 8.0),
        clamp01(math.log1p(sentence_count) / 5.0),
        clamp01(avg_word_len / 20.0),
        clamp01(avg_sentence_words / 80.0),
        clamp01(unique_ratio),
        safe_density(digit_count, char_count),
        safe_density(punctuation_count, char_count),
        safe_density(uppercase_count, char_count),
        safe_density(math_symbol_count, char_count),
        safe_density(newline_count, max(char_count / 80.0, 1.0)),
        clamp01(choice_count / 10.0),
        clamp01(code_marker_count / 20.0),
        clamp01(latex_marker_count / 20.0),
        clamp01(comparison_count / 20.0),
        clamp01(constraint_count / 20.0),
    ]
    features.extend(one_hot_bin(char_count, [128, 512, 2048, 8192]))
    features.extend(one_hot_bin(word_count, [25, 100, 400, 1000]))
    features.extend(one_hot_bin(sentence_count, [1, 3, 8, 20]))
    features.extend(one_hot_bin(newline_count, [0, 2, 8, 25]))
    features.extend(one_hot_bin(choice_count, [0, 2, 4, 6]))
    features.extend(one_hot_bin(math_symbol_count, [0, 2, 8, 25]))
    features.extend(one_hot_bin(code_marker_count, [0, 1, 4, 12]))
    features.extend(one_hot_bin(latex_marker_count, [0, 1, 4, 12]))
    features.extend(one_hot_bin(safe_density(punctuation_count, char_count), [0.02, 0.06, 0.12, 0.20]))
    features.extend(
        [
            float("?" in text),
            float(math_symbol_count > 0),
            float(latex_marker_count > 0),
            float(comparison_count > 0),
            float(code_marker_count > 0),
            float(choice_count > 0),
            float(has_table_shape(text)),
            float(any(marker in lower for marker in ("image", "figure", "<img", ".png", ".jpg"))),
            float(any(word in lower for word in ("prove", "derive", "proof", "theorem"))),
            float(any(word in lower for word in ("explain", "justify", "reason", "infer"))),
            float(any(word in lower for word in ("not", "except", "false", "incorrect"))),
            float(constraint_count >= 2),
        ]
    )
    return torch.tensor(features, dtype=torch.float32)


def split_sentences(text: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    return sentences or [""]


def one_hot_bin(value: float, thresholds: list[float]) -> list[float]:
    index = 0
    while index < len(thresholds) and value > thresholds[index]:
        index += 1
    return [float(position == index) for position in range(len(thresholds) + 1)]


def safe_density(count: float, total: float) -> float:
    return clamp01(count / max(total, 1.0))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def count_choice_markers(text: str) -> int:
    return len(re.findall(r"(?im)(?:^|\s)(?:\(?[A-H]\)|[A-H][\).:])\s+", text))


def count_code_markers(text: str) -> int:
    lower = text.lower()
    markers = [
        "```",
        "def ",
        "class ",
        "import ",
        "return ",
        "for ",
        "while ",
        "function",
        "traceback",
        "error:",
        "{",
        "}",
        "=>",
        "::",
    ]
    return sum(lower.count(marker) for marker in markers)


def count_latex_markers(text: str) -> int:
    return len(re.findall(r"\$|\\\(|\\\[|\\frac|\\sum|\\int|\\sqrt|\\log|\\mathbb|\\begin", text))


def has_table_shape(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    table_like_lines = sum("|" in line or "\t" in line for line in lines)
    return table_like_lines >= 2


def numeric_model_features(metadata: dict) -> list[float]:
    params_b = parse_params_billions(metadata.get("size_params"), metadata.get("name"))
    log_params = 0.0 if params_b is None else clamp01(math.log1p(params_b) / math.log1p(1000.0))
    return [
        log_params,
        normalized_release_date(metadata.get("release_date")),
        frontier_developer_feature(metadata),
    ]


def normalized_release_date(release_date: str | None) -> float:
    year, month = parse_release_year_month(release_date)
    if year is None:
        return 0.0
    decimal_year = year + (month - 1.0) / 12.0
    return clamp01((decimal_year - 2020.0) / 8.0)


def frontier_developer_feature(metadata: dict) -> float:
    text = " ".join(
        str(metadata.get(field) or "")
        for field in ("organization", "name", "model_id", "family")
    ).lower()
    markers = ("anthropic", "claude", "openai", "gpt", "google", "deepmind", "gemini")
    return float(any(marker in text for marker in markers))


def parse_params_billions(size_params: str | None, name: str | None) -> float | None:
    text = " ".join(value for value in (size_params, name) if value)
    if not text:
        return None
    mixture_match = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*([bBmM])", text)
    if mixture_match:
        count = float(mixture_match.group(1))
        size = float(mixture_match.group(2))
        unit = mixture_match.group(3).lower()
        value_b = count * size
        return value_b if unit == "b" else value_b / 1000.0
    values = []
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([bBmM])", text):
        value = float(number)
        values.append(value if unit.lower() == "b" else value / 1000.0)
    return max(values) if values else None


def parse_release_year_month(release_date: str | None) -> tuple[float | None, float]:
    if not release_date:
        return None, 0.0
    match = re.search(r"(20\d{2}|19\d{2})(?:[-/](\d{1,2}))?", str(release_date))
    if not match:
        return None, 0.0
    year = float(match.group(1))
    month = float(match.group(2) or 1.0)
    return year, max(1.0, min(12.0, month))


def normalize_metadata_value(value) -> str:
    if value is None:
        return MISSING_TOKEN
    text = str(value).strip().lower()
    return text or MISSING_TOKEN


def clip_probability(probability: float) -> float:
    return max(CLIP_LO, min(CLIP_HI, probability))


SCORER = None
LOAD_ERROR = None
LOAD_ERRORS = []

try:
    for artifact_path in (SUBMIT_ARTIFACT_PATH, ARTIFACT_PATH):
        if SCORER is not None:
            break
        try:
            if not artifact_path.exists():
                raise FileNotFoundError(f"Missing model artifact: {artifact_path}")
            SCORER = load_checkpoint(artifact_path)
        except Exception as exc:
            LOAD_ERRORS.append((artifact_path, exc))
    if SCORER is None:
        if LOAD_ERRORS:
            raise LOAD_ERRORS[0][1]
        raise FileNotFoundError("No Douglas model artifacts found.")
except Exception as exc:
    LOAD_ERROR = exc
    if not _local_smoke_test_enabled():
        print(f"[douglas/model.py] WARNING: using fallback predictor ({exc!r})", flush=True)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    if SCORER is None:
        return 0.5
    return SCORER.predict_probability(input)
