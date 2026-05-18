from __future__ import annotations

import math
import re
from pathlib import Path

import torch
from torch import nn


MISSING_TOKEN = "__missing__"
UNKNOWN_MODEL_TOKEN = "__unknown_model__"
MODEL_EMBED_DIM = 8
MODEL_VECTOR_DIM = 5
QUESTION_ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_QUESTION_TOKENS = 256
SENTENCE_COMPLEXITY_FEATURE_DIM = 16
CLIP_LO = 1e-7
CLIP_HI = 1.0 - 1e-7


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


class ModelSideEncoder(nn.Module):
    """Shared model-side encoder with a fallback unknown-model embedding."""

    def __init__(
        self,
        model_to_index: dict[str, int],
        num_models: int,
        organization_to_index: dict[str, int],
        family_to_index: dict[str, int],
        name_token_to_index: dict[str, int],
        p: int = MODEL_EMBED_DIM,
        output_dim: int = MODEL_VECTOR_DIM,
    ) -> None:
        super().__init__()
        if UNKNOWN_MODEL_TOKEN not in model_to_index:
            raise ValueError("model_to_index must contain UNKNOWN_MODEL_TOKEN.")
        if num_models <= 0:
            raise ValueError("num_models must be positive.")

        self.p = p
        self.output_dim = output_dim
        self.model_to_index = dict(model_to_index)
        self.organization_to_index = dict(organization_to_index)
        self.family_to_index = dict(family_to_index)
        self.name_token_to_index = dict(name_token_to_index)
        self.num_models = num_models

        self.model_embeddings = nn.Embedding(num_models, p)
        self.organization_embeddings = nn.Embedding(len(organization_to_index), p)
        self.family_embeddings = nn.Embedding(len(family_to_index), p)
        self.name_token_embeddings = nn.Embedding(len(name_token_to_index), p)
        self.size_projection = nn.Linear(2, p)
        self.release_projection = nn.Linear(3, p)
        self.output_projection = nn.Linear(6 * p, output_dim)

    def forward(self, metadata: dict) -> torch.Tensor:
        model_feature = self.model_embeddings(self.model_index(metadata))
        organization_feature = self.organization_embeddings(
            self.shared_index(metadata.get("organization"), self.organization_to_index)
        )
        family_feature = self.family_embeddings(
            self.shared_index(metadata.get("family"), self.family_to_index)
        )
        name_feature = self.name_feature(metadata.get("name"))
        size_feature = self.size_projection(self.size_features(metadata))
        release_feature = self.release_projection(self.release_features(metadata))

        full_feature = torch.cat(
            [
                model_feature,
                organization_feature,
                family_feature,
                name_feature,
                size_feature,
                release_feature,
            ],
            dim=0,
        )
        return self.output_projection(full_feature)

    def model_index(self, metadata: dict) -> torch.Tensor:
        model_key = model_lookup_key(metadata)
        index = self.model_to_index.get(model_key, self.model_to_index[UNKNOWN_MODEL_TOKEN])
        return torch.tensor(index, dtype=torch.long, device=self.model_embeddings.weight.device)

    def shared_index(self, value, vocab: dict[str, int]) -> torch.Tensor:
        key = normalize_metadata_value(value)
        index = vocab.get(key, vocab[MISSING_TOKEN])
        return torch.tensor(index, dtype=torch.long, device=self.model_embeddings.weight.device)

    def name_feature(self, name: str | None) -> torch.Tensor:
        token_indexes = [
            self.name_token_to_index.get(token, self.name_token_to_index[MISSING_TOKEN])
            for token in tokenize_name(name)
        ]
        if not token_indexes:
            token_indexes = [self.name_token_to_index[MISSING_TOKEN]]
        indexes = torch.tensor(
            token_indexes,
            dtype=torch.long,
            device=self.model_embeddings.weight.device,
        )
        return self.name_token_embeddings(indexes).mean(dim=0)

    def size_features(self, metadata: dict) -> torch.Tensor:
        params_b = parse_params_billions(
            metadata.get("size_params"),
            metadata.get("name"),
        )
        if params_b is None:
            values = [0.0, 1.0]
        else:
            values = [math.log1p(params_b), 0.0]
        return torch.tensor(values, dtype=torch.float32, device=self.model_embeddings.weight.device)

    def release_features(self, metadata: dict) -> torch.Tensor:
        year, month = parse_release_year_month(metadata.get("release_date"))
        if year is None:
            values = [0.0, 0.0, 1.0]
        else:
            values = [(year - 2020.0) / 10.0, month / 12.0, 0.0]
        return torch.tensor(values, dtype=torch.float32, device=self.model_embeddings.weight.device)


class ItemQuestionEncoder(nn.Module):
    """Encode each item sentence into loadings and bias.

    The transformer backbone can be cached. Training usually precomputes
    sentence representations, then updates only the small loading/bias heads.
    """

    def __init__(
        self,
        encoder_name: str = QUESTION_ENCODER_NAME,
        loading_dim: int = MODEL_VECTOR_DIM,
        max_length: int = MAX_QUESTION_TOKENS,
        freeze_backbone: bool = True,
        local_files_only: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.encoder_name = encoder_name
        self.loading_dim = loading_dim
        self.max_length = max_length
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
        self.loading_head = nn.Linear(self.representation_dim, loading_dim)
        self.bias_head = nn.Linear(self.representation_dim, 1)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, item_side_info: dict) -> tuple[torch.Tensor, torch.Tensor]:
        representations = self.encode_sentence_representations(item_side_info)
        return self.forward_from_representations(representations)

    def encode_sentence_representations(self, item_side_info: dict) -> torch.Tensor:
        text = item_side_info.get("item_content") or ""
        sentences = split_sentences(text)
        return self.encode_sentences(sentences)

    def encode_sentence_representations_batch(
        self,
        item_side_infos: list[dict],
    ) -> list[torch.Tensor]:
        sentence_groups = [
            split_sentences(item_side_info.get("item_content") or "")
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


class DouglasScorer(nn.Module):
    """Compute sigmoid(U_i dot V_j + z_j) for a model-item pair."""

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
        model_side_info = extract_model_metadata(
            input.get("subject_content"),
            model_id=input.get("model_id"),
        )
        U_i = self.model_encoder(model_side_info)
        if item_representations is None:
            item_representations = self.cached_item_representations(input)
        V_j, z_j = self.item_encoder.forward_from_representations(item_representations)
        return U_i, V_j, z_j

    def score_logit(
        self,
        input: dict,
        item_representations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        U_i, V_j, z_j = self.encode_pair(input, item_representations=item_representations)
        return torch.dot(U_i, V_j) + z_j

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
        key = input.get("item_content") or ""
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
    model_to_index = {UNKNOWN_MODEL_TOKEN: 0}
    organizations = []
    families = []
    names = []
    next_index = 1

    for example in examples:
        metadata = extract_model_metadata(
            example.get("subject_content"),
            model_id=example.get("model_id"),
        )
        aliases = model_aliases(metadata)
        if not aliases:
            continue

        organizations.append(metadata.get("organization"))
        families.append(metadata.get("family"))
        names.append(metadata.get("name"))

        index = next((model_to_index[alias] for alias in aliases if alias in model_to_index), None)
        if index is None:
            index = next_index
            next_index += 1
        for alias in aliases:
            model_to_index[alias] = index

    num_models = next_index
    print(f"Found {num_models - 1} known fixed models plus unknown fallback.")
    return ModelSideEncoder(
        model_to_index=model_to_index,
        num_models=num_models,
        organization_to_index=build_shared_vocab(organizations),
        family_to_index=build_shared_vocab(families),
        name_token_to_index=build_name_token_vocab(names),
        p=p,
        output_dim=output_dim,
    )


def save_checkpoint(
    path: Path,
    scorer: DouglasScorer,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "temperature": scorer.temperature,
            "model_to_index": scorer.model_encoder.model_to_index,
            "num_models": scorer.model_encoder.num_models,
            "organization_to_index": scorer.model_encoder.organization_to_index,
            "family_to_index": scorer.model_encoder.family_to_index,
            "name_token_to_index": scorer.model_encoder.name_token_to_index,
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

    model_encoder = ModelSideEncoder(
        model_to_index=data["model_to_index"],
        num_models=data["num_models"],
        organization_to_index=data["organization_to_index"],
        family_to_index=data["family_to_index"],
        name_token_to_index=data["name_token_to_index"],
        p=p,
        output_dim=k,
    )
    model_encoder.load_state_dict(data["model_encoder_state_dict"])

    item_encoder = ItemQuestionEncoder(
        encoder_name=encoder_name,
        loading_dim=k,
        max_length=max_length,
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
