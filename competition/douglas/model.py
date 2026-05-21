"""Douglas BGE neural submit model.

This file is self-contained for submission. It loads the standard BGE-large
checkpoint trained by:

    modal run competition/douglas/modal_train_submit.py \
      --item-encoder bge-large \
      --epochs 3 \
      --batch-size 512 \
      --learning-rate 3e-4 \
      --irt-l2 1e-3 \
      --temperature 1 \
      --weight-decay 1e-4 \
      --item-head-hidden-dim 64 \
      --limit 0 \
      --artifact-suffix _3epoch

Expected submitted files:
- model.py
- models.txt
- requirements.txt
- artifacts/douglas_submit_bge_large_3epoch.pt
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import torch
from torch import nn


LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
SUBMIT_ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "douglas_submit_bge_large_3epoch.pt"

MISSING_TOKEN = "__missing__"
MODEL_EMBED_DIM = 8
METADATA_FIELDS = ("model_id", "name", "organization", "size_params", "release_date", "family")
NUMERIC_MODEL_FEATURES = ("log_params", "release_date", "frontier_developer")
MODEL_FEATURE_DIM = len(METADATA_FIELDS) + len(NUMERIC_MODEL_FEATURES)
MODEL_VECTOR_DIM = 4
QUESTION_ENCODER_NAME = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this evaluation question for difficulty prediction: "
MAX_QUESTION_TOKENS = 256
SENTENCE_COMPLEXITY_FEATURE_DIM = 16
ITEM_HEAD_HIDDEN_DIM = 64
ITEM_HEAD_RESIDUAL = True
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


def _resolve_cache_dir() -> str | None:
    candidates = [
        os.environ.get("HF_HOME", "").strip(),
        "/app/hf_cache",
        str(Path(__file__).with_name(".hf_cache")),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(path, os.W_OK):
            return str(path)
    return None


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
    for line in str(subject_content).splitlines():
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
        output_dim: int = MODEL_VECTOR_DIM,
    ) -> None:
        super().__init__()
        self.global_mean = float(global_mean)
        self.field_value_means = {
            field: dict((field_value_means or {}).get(field, {}))
            for field in METADATA_FIELDS
        }
        self.field_default_means = {
            field: float((field_default_means or {}).get(field, self.global_mean))
            for field in METADATA_FIELDS
        }
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


class DirectModelLookupEncoder(nn.Module):
    def __init__(
        self,
        model_vectors: torch.Tensor,
        model_id_to_index: dict[str, int] | None = None,
        name_to_index: dict[str, int] | None = None,
        fallback_vector: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        vectors = model_vectors.detach().clone().float()
        self.register_buffer("model_vectors", vectors)
        if fallback_vector is None:
            fallback_vector = vectors.mean(dim=0) if len(vectors) else torch.zeros(vectors.shape[1])
        self.register_buffer("fallback_vector", fallback_vector.detach().clone().float())
        self.model_id_to_index = dict(model_id_to_index or {})
        self.name_to_index = dict(name_to_index or {})

    def forward(self, metadata: dict) -> torch.Tensor:
        for value, mapping in (
            (metadata.get("model_id"), self.model_id_to_index),
            (metadata.get("name"), self.name_to_index),
        ):
            key = normalize_metadata_value(value)
            if key in mapping:
                return self.model_vectors[mapping[key]]
        return self.fallback_vector


class ItemQuestionEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str = QUESTION_ENCODER_NAME,
        loading_dim: int = MODEL_VECTOR_DIM,
        max_length: int = MAX_QUESTION_TOKENS,
        hidden_dim: int = ITEM_HEAD_HIDDEN_DIM,
        item_head_residual: bool = ITEM_HEAD_RESIDUAL,
        local_files_only: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.encoder_name = encoder_name
        self.loading_dim = loading_dim
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.item_head_residual = item_head_residual
        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.backbone = AutoModel.from_pretrained(
            encoder_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        backbone_dim = self.backbone.config.hidden_size
        self.representation_dim = backbone_dim + SENTENCE_COMPLEXITY_FEATURE_DIM
        self.loading_head = build_item_head(
            self.representation_dim,
            loading_dim,
            hidden_dim=hidden_dim,
            residual=item_head_residual,
        )
        self.bias_head = build_item_head(
            self.representation_dim,
            1,
            hidden_dim=hidden_dim,
            residual=item_head_residual,
        )

    def encode_sentence_representations(self, item_side_info: dict) -> torch.Tensor:
        sentences = split_sentences(render_item_encoder_text(item_side_info))
        return self.encode_sentences(sentences)

    def encode_sentences(self, sentences: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            [f"{BGE_QUERY_PREFIX}{sentence}" for sentence in sentences],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.backbone.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = self.backbone(**encoded)
        pooled = self.mean_pool(output.last_hidden_state, encoded["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        complexity = sentence_complexity_features(sentences).to(pooled.device)
        return torch.cat([pooled, complexity], dim=-1)

    def forward_from_representations(
        self,
        sentence_representations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loading = torch.softmax(self.loading_head(sentence_representations), dim=-1)
        bias = self.bias_head(sentence_representations).squeeze(-1)
        return loading.mean(dim=0), bias.mean()

    @staticmethod
    def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


class DouglasScorer(nn.Module):
    def __init__(
        self,
        model_encoder: nn.Module,
        item_encoder: ItemQuestionEncoder,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.model_encoder = model_encoder
        self.item_encoder = item_encoder
        self.temperature = float(temperature)
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
        return self.score_logit(input) / max(self.temperature, 1e-8)

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


class ResidualItemHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.residual_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs)
        hidden = self.activation(hidden + self.residual_block(hidden))
        return self.output_projection(hidden)


def build_item_head(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    residual: bool,
) -> nn.Module:
    if residual:
        return ResidualItemHead(input_dim, output_dim, hidden_dim)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(hidden_dim, output_dim),
    )


def load_checkpoint(path: Path) -> DouglasScorer:
    data = torch.load(path, map_location="cpu", weights_only=False)
    config = data.get("config", {})
    k = int(config.get("k", MODEL_VECTOR_DIM))
    encoder_name = config.get("encoder_name", QUESTION_ENCODER_NAME)
    max_length = int(config.get("max_length", MAX_QUESTION_TOKENS))
    item_head_hidden_dim = int(config.get("item_head_hidden_dim", ITEM_HEAD_HIDDEN_DIM))
    item_head_residual = bool(config.get("item_head_residual", ITEM_HEAD_RESIDUAL))

    model_encoder_type = config.get("model_encoder_type", "metadata_projection")
    if model_encoder_type == "direct_lookup":
        state = data["model_encoder_state_dict"]
        model_encoder = DirectModelLookupEncoder(
            model_vectors=state["model_vectors"],
            model_id_to_index=data.get("model_id_to_index", {}),
            name_to_index=data.get("name_to_index", {}),
            fallback_vector=state.get("fallback_vector"),
        )
    else:
        model_encoder = ModelSideEncoder(
            field_value_means=data.get("field_value_means"),
            field_default_means=data.get("field_default_means"),
            global_mean=float(data.get("global_mean", 0.5)),
            output_dim=k,
        )
    model_encoder.load_state_dict(data["model_encoder_state_dict"])

    item_encoder = ItemQuestionEncoder(
        encoder_name=encoder_name,
        loading_dim=k,
        max_length=max_length,
        hidden_dim=item_head_hidden_dim,
        item_head_residual=item_head_residual,
        local_files_only=True,
        cache_dir=_resolve_cache_dir(),
    )
    item_encoder.loading_head.load_state_dict(data["item_heads_state_dict"]["loading_head"])
    item_encoder.bias_head.load_state_dict(data["item_heads_state_dict"]["bias_head"])

    scorer = DouglasScorer(
        model_encoder=model_encoder,
        item_encoder=item_encoder,
        temperature=float(data.get("temperature", config.get("temperature", 1.0))),
    )
    scorer.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return scorer.to(device)


def split_sentences(text: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(text))
        if sentence.strip()
    ]
    return sentences or [""]


def render_item_encoder_text(item_side_info: dict) -> str:
    benchmark = item_side_info.get("benchmark") or "unknown"
    condition = item_side_info.get("condition") or "none"
    item_content = item_side_info.get("item_content") or ""
    return f"benchmark: {benchmark}\ncondition: {condition}\nitem: {item_content}"


def sentence_complexity_features(sentences: list[str]) -> torch.Tensor:
    return torch.tensor(
        [sentence_complexity_feature(sentence) for sentence in sentences],
        dtype=torch.float32,
    )


def sentence_complexity_feature(sentence: str) -> list[float]:
    words = re.findall(r"[A-Za-z0-9_]+", sentence)
    word_count = len(words)
    char_count = len(sentence)
    avg_word_len = sum(len(word) for word in words) / word_count if word_count else 0.0
    digit_count = sum(char.isdigit() for char in sentence)
    punctuation_count = sum(char in ".,;:!?()[]{}" for char in sentence)
    uppercase_count = sum(char.isupper() for char in sentence)
    math_symbols = "∫∑∂√≤≥≠≈∞πθλμ+-*/=<>^"
    math_symbol_count = sum(char in math_symbols for char in sentence)
    lower = sentence.lower()

    return [
        math.log1p(char_count) / 8.0,
        math.log1p(word_count) / 6.0,
        min(avg_word_len, 20.0) / 20.0,
        digit_count / max(char_count, 1),
        punctuation_count / max(char_count, 1),
        uppercase_count / max(char_count, 1),
        math_symbol_count / max(char_count, 1),
        float("?" in sentence),
        float(math_symbol_count > 0),
        float(bool(re.search(r"\$.*?\$|\\[a-zA-Z]+|\\\(|\\\[", sentence))),
        float(bool(re.search(r"[A-Za-z0-9]\s*(=|<|>|≤|≥|≈|≠)\s*[A-Za-z0-9]", sentence))),
        float(bool(re.search(r"\b\d+(?:\.\d+)?\b", sentence))),
        float(any(marker in lower for marker in ("def ", "class ", "function", "return ", "import ", "```"))),
        float(any(marker in lower for marker in (" a)", " b)", " c)", " d)", "(a)", "(b)", "(c)", "(d)"))),
        float(any(word in lower for word in ("prove", "derive", "explain", "justify", "reason", "infer"))),
        float(any(word in lower for word in ("not", "except", "least", "false", "incorrect"))),
    ]


def numeric_model_features(metadata: dict) -> list[float]:
    params_b = parse_params_billions(metadata.get("size_params"), metadata.get("name"))
    log_params = 0.0 if params_b is None else clamp01(math.log1p(params_b) / math.log1p(1000.0))
    return [
        log_params,
        normalized_release_date(metadata.get("release_date")),
        frontier_developer_feature(metadata),
    ]


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


def normalized_release_date(release_date: str | None) -> float:
    year, month = parse_release_year_month(release_date)
    if year is None:
        return 0.0
    decimal_year = year + (month - 1.0) / 12.0
    return clamp01((decimal_year - 2020.0) / 8.0)


def parse_release_year_month(release_date: str | None) -> tuple[float | None, float]:
    if not release_date:
        return None, 0.0
    match = re.search(r"(20\d{2}|19\d{2})(?:[-/](\d{1,2}))?", str(release_date))
    if not match:
        return None, 0.0
    year = float(match.group(1))
    month = float(match.group(2) or 1.0)
    return year, max(1.0, min(12.0, month))


def frontier_developer_feature(metadata: dict) -> float:
    text = " ".join(
        str(metadata.get(field) or "")
        for field in ("organization", "name", "model_id", "family")
    ).lower()
    markers = ("anthropic", "claude", "openai", "gpt", "google", "deepmind", "gemini")
    return float(any(marker in text for marker in markers))


def normalize_metadata_value(value) -> str:
    if value is None:
        return MISSING_TOKEN
    text = str(value).strip().lower()
    return text or MISSING_TOKEN


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def clip_probability(probability: float) -> float:
    return max(CLIP_LO, min(CLIP_HI, float(probability)))


SCORER = None
LOAD_ERROR = None

try:
    if _local_smoke_test_enabled():
        SCORER = None
    else:
        if not SUBMIT_ARTIFACT_PATH.exists():
            raise FileNotFoundError(f"Missing model artifact: {SUBMIT_ARTIFACT_PATH}")
        SCORER = load_checkpoint(SUBMIT_ARTIFACT_PATH)
except Exception as exc:
    LOAD_ERROR = exc
    if _local_smoke_test_enabled():
        print(f"[douglas/model.py] WARNING: using fallback predictor ({exc!r})", flush=True)
    else:
        raise


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    if SCORER is None:
        return 0.5
    return SCORER.predict_probability(input)
