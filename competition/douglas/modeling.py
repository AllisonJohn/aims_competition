from __future__ import annotations

import math
import re
from pathlib import Path

import torch
from torch import nn


MISSING_TOKEN = "__missing__"
UNKNOWN_MODEL_TOKEN = "__unknown_model__"
MODEL_EMBED_DIM = 8
METADATA_FIELDS = ("model_id", "name", "organization", "size_params", "release_date", "family")
NUMERIC_MODEL_FEATURES = ("log_params", "release_date", "frontier_developer")
MODEL_FEATURE_DIM = len(METADATA_FIELDS) + len(NUMERIC_MODEL_FEATURES)
MODEL_VECTOR_DIM = 4
QUESTION_ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_QUESTION_TOKENS = 256
SENTENCE_COMPLEXITY_FEATURE_DIM = 16
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


def first_present(mapping: dict, keys: tuple[str, ...]):
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def is_pairwise_input(input: dict) -> bool:
    """Return True when an input contains two visible model sides."""
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
    """Fixed metadata statistics followed by a model-side linear projection."""

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


class ItemQuestionEncoder(nn.Module):
    """Encode each item sentence into loadings and bias.

    The transformer backbone can be cached. Training usually precomputes
    sentence representations, then updates only the small loading/bias heads.
    """

    item_encoder_type = "transformer"

    def __init__(
        self,
        encoder_name: str = QUESTION_ENCODER_NAME,
        loading_dim: int = MODEL_VECTOR_DIM,
        max_length: int = MAX_QUESTION_TOKENS,
        hidden_dim: int = ITEM_HEAD_HIDDEN_DIM,
        freeze_backbone: bool = True,
        local_files_only: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.encoder_name = encoder_name
        self.loading_dim = loading_dim
        self.max_length = max_length
        self.hidden_dim = hidden_dim
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

        backbone_dim = self.backbone.config.hidden_size
        self.representation_dim = backbone_dim + SENTENCE_COMPLEXITY_FEATURE_DIM
        self.loading_head = build_item_head(self.representation_dim, loading_dim, hidden_dim=hidden_dim)
        self.bias_head = build_item_head(self.representation_dim, 1, hidden_dim=hidden_dim)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, item_side_info: dict) -> tuple[torch.Tensor, torch.Tensor]:
        representations = self.encode_sentence_representations(item_side_info)
        return self.forward_from_representations(representations)

    def encode_sentence_representations(self, item_side_info: dict) -> torch.Tensor:
        text = render_item_encoder_text(item_side_info)
        sentences = split_sentences(text)
        return self.encode_sentences(sentences)

    def encode_sentence_representations_batch(
        self,
        item_side_infos: list[dict],
    ) -> list[torch.Tensor]:
        sentence_groups = [
            split_sentences(render_item_encoder_text(item_side_info))
            for item_side_info in item_side_infos
        ]
        flat_sentences = [
            sentence
            for sentences in sentence_groups
            for sentence in sentences
        ]
        flat_representations = self.encode_sentences(flat_sentences)

        grouped = []
        offset = 0
        for sentences in sentence_groups:
            next_offset = offset + len(sentences)
            grouped.append(flat_representations[offset:next_offset])
            offset = next_offset
        return grouped

    def encode_sentences(self, sentences: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.backbone.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = self.backbone(**encoded)
        pooled = self.mean_pool(
            output.last_hidden_state,
            encoded["attention_mask"],
        )
        complexity = sentence_complexity_features(sentences).to(pooled.device)
        return torch.cat([pooled, complexity], dim=-1)

    def forward_from_representations(
        self,
        sentence_representations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loading = torch.softmax(self.loading_head(sentence_representations), dim=-1)
        bias = self.bias_head(sentence_representations).squeeze(-1)
        return loading.mean(dim=0), bias.mean()

    def mean_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


class HandcraftedItemQuestionEncoder(nn.Module):
    """Encode one-hot item hardness features into loadings and bias."""

    item_encoder_type = "handcrafted"

    def __init__(
        self,
        encoder_name: str = "handcrafted-item-features",
        loading_dim: int = MODEL_VECTOR_DIM,
        max_length: int = MAX_QUESTION_TOKENS,
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

    def forward(self, item_side_info: dict) -> tuple[torch.Tensor, torch.Tensor]:
        representations = self.encode_sentence_representations(item_side_info)
        return self.forward_from_representations(representations)

    def encode_sentence_representations(self, item_side_info: dict) -> torch.Tensor:
        return item_hardness_features(item_side_info).to(self.head_device()).unsqueeze(0)

    def encode_sentence_representations_batch(
        self,
        item_side_infos: list[dict],
    ) -> list[torch.Tensor]:
        device = self.head_device()
        return [
            item_hardness_features(item_side_info).to(device).unsqueeze(0)
            for item_side_info in item_side_infos
        ]

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
    """Compute absolute or pairwise IRT probabilities."""

    def __init__(
        self,
        model_encoder: ModelSideEncoder,
        item_encoder: ItemQuestionEncoder,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.model_encoder = model_encoder
        self.item_encoder = item_encoder
        self.temperature = temperature
        self.item_representation_cache: dict[str, torch.Tensor] = {}

    def encode_pair(
        self,
        input: dict,
        item_representations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_side_info = extract_left_model_metadata(input)
        U_i = self.model_encoder(model_side_info)
        if item_representations is None:
            item_representations = self.cached_item_representations(input)
        V_j, z_j = self.item_encoder.forward_from_representations(item_representations)
        return U_i, V_j, z_j

    def encode_pairwise(
        self,
        input: dict,
        item_representations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        U_a = self.model_encoder(extract_left_model_metadata(input))
        U_b = self.model_encoder(extract_right_model_metadata(input))
        if item_representations is None:
            item_representations = self.cached_item_representations(input)
        V_j, z_j = self.item_encoder.forward_from_representations(item_representations)
        return U_a, U_b, V_j, z_j

    def score_logit(
        self,
        input: dict,
        item_representations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if is_pairwise_input(input):
            return self.score_pairwise_logit(input, item_representations=item_representations)
        U_i, V_j, z_j = self.encode_pair(input, item_representations=item_representations)
        return torch.dot(U_i, V_j) + z_j

    def score_pairwise_logit(
        self,
        input: dict,
        item_representations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        U_a, U_b, V_j, z_j = self.encode_pairwise(
            input,
            item_representations=item_representations,
        )
        return torch.dot(U_a - U_b, V_j) + z_j

    def forward(
        self,
        input: dict,
        item_representations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.score_logit(input, item_representations=item_representations) / self.temperature

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


def build_model_side_encoder(
    examples: list[dict],
    p: int = MODEL_EMBED_DIM,
    output_dim: int = MODEL_VECTOR_DIM,
) -> ModelSideEncoder:
    field_sums = {field: {} for field in METADATA_FIELDS}
    field_counts = {field: {} for field in METADATA_FIELDS}
    global_sum = 0.0
    global_count = 0

    for example in examples:
        label = example.get("label")
        if label not in (0, 1, 0.0, 1.0):
            continue
        label = float(label)
        if is_pairwise_input(example):
            metadata_labels = (
                (extract_left_model_metadata(example), label),
                (extract_right_model_metadata(example), 1.0 - label),
            )
        else:
            metadata_labels = (
                (extract_left_model_metadata(example), label),
            )

        for metadata, metadata_label in metadata_labels:
            global_sum += metadata_label
            global_count += 1

            for field in METADATA_FIELDS:
                key = normalize_metadata_value(metadata.get(field))
                field_sums[field][key] = field_sums[field].get(key, 0.0) + metadata_label
                field_counts[field][key] = field_counts[field].get(key, 0) + 1

    global_mean = global_sum / max(global_count, 1)
    field_value_means = {}
    field_default_means = {}
    for field in METADATA_FIELDS:
        value_means = {
            key: field_sums[field][key] / field_counts[field][key]
            for key in field_sums[field]
        }
        field_value_means[field] = value_means
        field_default_means[field] = (
            sum(value_means.values()) / len(value_means)
            if value_means
            else global_mean
        )

    value_counts = ", ".join(
        f"{field}={len(field_value_means[field])}"
        for field in METADATA_FIELDS
    )
    print(
        f"Built fixed metadata means from {global_count} examples "
        f"(global_mean={global_mean:.4f}; {value_counts}).",
        flush=True,
    )
    return ModelSideEncoder(
        field_value_means=field_value_means,
        field_default_means=field_default_means,
        global_mean=global_mean,
        p=p,
        output_dim=output_dim,
    )


def save_checkpoint(
    path: Path,
    scorer: DouglasScorer,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_config = dict(config)
    saved_config["item_encoder_type"] = getattr(
        scorer.item_encoder,
        "item_encoder_type",
        "transformer",
    )
    saved_config["item_head_hidden_dim"] = getattr(
        scorer.item_encoder,
        "hidden_dim",
        ITEM_HEAD_HIDDEN_DIM,
    )
    torch.save(
        {
            "config": saved_config,
            "temperature": scorer.temperature,
            "model_to_index": scorer.model_encoder.model_to_index,
            "num_models": scorer.model_encoder.num_models,
            "organization_to_index": scorer.model_encoder.organization_to_index,
            "family_to_index": scorer.model_encoder.family_to_index,
            "name_token_to_index": scorer.model_encoder.name_token_to_index,
            "field_value_means": scorer.model_encoder.field_value_means,
            "field_default_means": scorer.model_encoder.field_default_means,
            "global_mean": scorer.model_encoder.global_mean,
            "model_encoder_state_dict": scorer.model_encoder.state_dict(),
            "item_heads_state_dict": {
                "loading_head": scorer.item_encoder.loading_head.state_dict(),
                "bias_head": scorer.item_encoder.bias_head.state_dict(),
            },
        },
        path,
    )


def load_checkpoint(
    path: Path,
    local_files_only: bool = True,
    cache_dir: str | None = None,
    map_location: str = "cpu",
) -> DouglasScorer:
    data = torch.load(path, map_location=map_location, weights_only=False)
    config = data.get("config", {})
    k = int(config.get("k", MODEL_VECTOR_DIM))
    p = int(config.get("p", MODEL_EMBED_DIM))
    encoder_name = config.get("encoder_name", QUESTION_ENCODER_NAME)
    max_length = int(config.get("max_length", MAX_QUESTION_TOKENS))
    item_encoder_type = config.get("item_encoder_type", "transformer")
    item_head_hidden_dim = int(config.get("item_head_hidden_dim", ITEM_HEAD_HIDDEN_DIM))

    model_encoder = ModelSideEncoder(
        field_value_means=data.get("field_value_means"),
        field_default_means=data.get("field_default_means"),
        global_mean=float(data.get("global_mean", 0.5)),
        p=p,
        output_dim=k,
    )
    model_encoder.load_state_dict(data["model_encoder_state_dict"])

    item_encoder_cls = (
        HandcraftedItemQuestionEncoder
        if item_encoder_type == "handcrafted"
        else ItemQuestionEncoder
    )
    item_encoder = item_encoder_cls(
        encoder_name=encoder_name,
        loading_dim=k,
        max_length=max_length,
        hidden_dim=item_head_hidden_dim,
        freeze_backbone=True,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
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


def split_sentences(text: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    return sentences or [""]


def render_item_encoder_text(item_side_info: dict) -> str:
    """Include task context in the text encoder input."""
    benchmark = item_side_info.get("benchmark") or "unknown"
    condition = item_side_info.get("condition") or "none"
    item_content = item_side_info.get("item_content") or ""
    return f"benchmark: {benchmark}\ncondition: {condition}\nitem: {item_content}"


def build_item_head(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = ITEM_HEAD_HIDDEN_DIM,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(hidden_dim, output_dim),
    )


def sentence_complexity_features(sentences: list[str]) -> torch.Tensor:
    return torch.tensor(
        [sentence_complexity_feature(sentence) for sentence in sentences],
        dtype=torch.float32,
    )


def sentence_complexity_feature(sentence: str) -> list[float]:
    words = re.findall(r"[A-Za-z0-9_]+", sentence)
    word_count = len(words)
    char_count = len(sentence)
    avg_word_len = (
        sum(len(word) for word in words) / word_count
        if word_count
        else 0.0
    )
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
    params_b = parse_params_billions(
        metadata.get("size_params"),
        metadata.get("name"),
    )
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
    frontier_markers = (
        "anthropic",
        "claude",
        "openai",
        "gpt",
        "google",
        "deepmind",
        "gemini",
    )
    return float(any(marker in text for marker in frontier_markers))


def item_hardness_features(item_side_info: dict) -> torch.Tensor:
    text = item_side_info.get("item_content") or ""
    lower = text.lower()
    sentences = split_sentences(text)
    words = re.findall(r"[A-Za-z0-9_]+", text)
    unique_words = {word.lower() for word in words}

    char_count = len(text)
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_len = (
        sum(len(word) for word in words) / word_count
        if word_count
        else 0.0
    )
    avg_sentence_words = word_count / max(sentence_count, 1)
    unique_ratio = len(unique_words) / max(word_count, 1)

    digit_count = sum(char.isdigit() for char in text)
    punctuation_count = sum(char in ".,;:!?()[]{}" for char in text)
    uppercase_count = sum(char.isupper() for char in text)
    math_symbols = "∫∑∂√≤≥≠≈∞πθλμ+-*/=<>^"
    math_symbol_count = sum(char in math_symbols for char in text)
    newline_count = text.count("\n")
    choice_count = count_choice_markers(text)
    code_marker_count = count_code_markers(text)
    latex_marker_count = count_latex_markers(text)
    comparison_count = len(re.findall(r"[A-Za-z0-9]\s*(=|<|>|≤|≥|≈|≠)\s*[A-Za-z0-9]", text))
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
    if len(features) != ITEM_FEATURE_DIM:
        raise ValueError(f"Expected {ITEM_FEATURE_DIM} item features, got {len(features)}.")
    return torch.tensor(features, dtype=torch.float32)


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
    return len(
        re.findall(
            r"(?im)(?:^|\s)(?:\(?[A-H]\)|[A-H][\).:])\s+",
            text,
        )
    )


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
    return len(
        re.findall(
            r"\$|\\\(|\\\[|\\frac|\\sum|\\int|\\sqrt|\\log|\\mathbb|\\begin",
            text,
        )
    )


def has_table_shape(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    table_like_lines = sum("|" in line or "\t" in line for line in lines)
    return table_like_lines >= 2


def model_lookup_key(metadata: dict) -> str | None:
    for field in ("model_id", "name"):
        value = metadata.get(field)
        if value:
            return str(value).strip().lower()
    return None


def model_aliases(metadata: dict) -> list[str]:
    aliases = []
    for field in ("model_id", "name"):
        value = metadata.get(field)
        if value:
            aliases.append(str(value).strip().lower())
    return aliases


def normalize_metadata_value(value) -> str:
    if value is None:
        return MISSING_TOKEN
    text = str(value).strip().lower()
    return text or MISSING_TOKEN


def tokenize_name(name: str | None) -> list[str]:
    if not name:
        return []
    return re.findall(r"[a-z0-9]+", name.lower())


def parse_params_billions(size_params: str | None, name: str | None) -> float | None:
    text = " ".join(value for value in (size_params, name) if value)
    if not text:
        return None

    mixture_match = re.search(
        r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*([bBmM])",
        text,
    )
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
    match = re.search(r"\b(19\d{2}|20\d{2})(?:[-/](\d{1,2}))?", release_date)
    if not match:
        return None, 0.0
    year = float(match.group(1))
    month = float(match.group(2)) if match.group(2) else 1.0
    return year, max(1.0, min(month, 12.0))


def build_shared_vocab(values) -> dict[str, int]:
    vocab = {MISSING_TOKEN: 0}
    for value in values:
        key = normalize_metadata_value(value)
        if key not in vocab:
            vocab[key] = len(vocab)
    return vocab


def build_name_token_vocab(names) -> dict[str, int]:
    vocab = {MISSING_TOKEN: 0}
    for name in names:
        for token in tokenize_name(name):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def clip_probability(probability: float) -> float:
    return max(CLIP_LO, min(CLIP_HI, float(probability)))
