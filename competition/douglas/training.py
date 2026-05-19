"""Offline training script for Douglas's competition model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.douglas.modeling import (  # noqa: E402
    MODEL_EMBED_DIM,
    MODEL_VECTOR_DIM,
    QUESTION_ENCODER_NAME,
    ITEM_HEAD_HIDDEN_DIM,
    DouglasScorer,
    ItemQuestionEncoder,
    build_model_side_encoder,
    extract_left_model_metadata,
    extract_right_model_metadata,
    is_pairwise_input,
    save_checkpoint,
)
from competition.utils.load_train_data import evaluate, load_split_data  # noqa: E402


ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "douglas_model.pt"
ITEM_CACHE_PATH = Path(__file__).with_name("artifacts") / "item_representations_context_v2.pt"
LOG_EVERY_EXAMPLES = 100_000
LOG_EVERY_UNIQUE_ITEMS = 1_000
LOG_EVERY_BATCHES = 100


class IRTFactorModel(nn.Module):
    """Direct k-factor IRT lookup model used to create training targets."""

    def __init__(self, num_models: int, num_items: int, k: int) -> None:
        super().__init__()
        self.model_factors = nn.Embedding(num_models, k)
        self.item_loading_logits = nn.Embedding(num_items, k)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.normal_(self.model_factors.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.item_loading_logits.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        model_indexes: torch.Tensor,
        item_indexes: torch.Tensor,
        right_model_indexes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        U_i = self.model_factors(model_indexes)
        if right_model_indexes is not None:
            right_safe = right_model_indexes.clamp(min=0)
            right_mask = (right_model_indexes >= 0).to(U_i.dtype).unsqueeze(-1)
            U_i = U_i - self.model_factors(right_safe) * right_mask
        V_j = torch.softmax(self.item_loading_logits(item_indexes), dim=-1)
        z_j = self.item_bias(item_indexes).squeeze(-1)
        return (U_i * V_j).sum(dim=-1) + z_j


class DouglasModel:
    """Training wrapper around the Douglas scorer."""

    def __init__(
        self,
        k: int = MODEL_VECTOR_DIM,
        p: int = MODEL_EMBED_DIM,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 1,
        batch_size: int = 64,
        encode_batch_size: int = 128,
        item_head_hidden_dim: int = ITEM_HEAD_HIDDEN_DIM,
        temperature: float = 1.0,
        irt_l2: float = 1e-4,
        device: str | None = None,
    ) -> None:
        self.k = k
        self.p = p
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.encode_batch_size = encode_batch_size
        self.item_head_hidden_dim = item_head_hidden_dim
        self.temperature = temperature
        self.irt_l2 = irt_l2
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer: DouglasScorer | None = None

    def train(self, examples) -> None:
        print(
            "DouglasModel config: "
            f"k={self.k} p={self.p} epochs={self.epochs} batch_size={self.batch_size} "
            f"encode_batch_size={self.encode_batch_size} item_head_hidden_dim={self.item_head_hidden_dim} "
            f"lr={self.learning_rate} "
            f"weight_decay={self.weight_decay} irt_l2={self.irt_l2} "
            f"temperature={self.temperature} device={self.device}",
            flush=True,
        )
        print("[stage 1/5] Materializing binary training examples...", flush=True)
        examples = self._materialize_binary_examples(examples)
        print("[stage 2/5] Indexing model/item observations...", flush=True)
        indexed = self._build_indexed_training_data(examples)
        print("[stage 3/5] Fitting latent IRT factors...", flush=True)
        irt_model = self._fit_irt_factors(indexed)

        print("[stage 4/5] Fitting metadata -> U_i predictor...", flush=True)
        model_encoder = build_model_side_encoder(examples, p=self.p, output_dim=self.k).to(self.device)
        self._fit_model_side_encoder(
            model_encoder=model_encoder,
            indexed=indexed,
            irt_model=irt_model,
        )

        print("[stage 5/5] Fitting item content -> V_j,z_j predictor...", flush=True)
        print(
            f"Initializing item encoder {ItemQuestionEncoder.__name__} "
            f"with encoder_name={QUESTION_ENCODER_NAME!r}",
            flush=True,
        )
        item_encoder = ItemQuestionEncoder(
            encoder_name=QUESTION_ENCODER_NAME,
            loading_dim=self.k,
            hidden_dim=self.item_head_hidden_dim,
            freeze_backbone=True,
        ).to(self.device)
        self.scorer = DouglasScorer(
            model_encoder=model_encoder,
            item_encoder=item_encoder,
            temperature=self.temperature,
        ).to(self.device)

        item_cache = self._precompute_item_representations_from_infos(indexed["item_infos"])
        self._fit_item_side_encoder(
            item_encoder=item_encoder,
            indexed=indexed,
            irt_model=irt_model,
            item_cache=item_cache,
        )

    def _fit_irt_factors(self, indexed: dict) -> IRTFactorModel:
        print(
            f"Fitting {self.k}-factor IRT model by joint maximum likelihood "
            f"on {len(indexed['labels'])} observed entries...",
            flush=True,
        )
        irt_model = IRTFactorModel(
            num_models=len(indexed["model_to_index"]),
            num_items=len(indexed["item_to_index"]),
            k=self.k,
        ).to(self.device)
        dataset = TensorDataset(
            indexed["model_indexes"],
            indexed["right_model_indexes"],
            indexed["item_indexes"],
            indexed["labels"],
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        print(
            f"IRT setup: models={len(indexed['model_to_index'])} "
            f"items={len(indexed['item_to_index'])} batches={len(loader)} "
            f"lr={self.learning_rate} l2={self.irt_l2}",
            flush=True,
        )
        optimizer = AdamW(irt_model.parameters(), lr=self.learning_rate, weight_decay=0.0)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            irt_model.train()
            total_loss = 0.0
            total_count = 0

            for batch_number, (
                model_indexes,
                right_model_indexes,
                item_indexes,
                labels,
            ) in enumerate(loader, start=1):
                model_indexes = model_indexes.to(self.device)
                right_model_indexes = right_model_indexes.to(self.device)
                item_indexes = item_indexes.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = irt_model(model_indexes, item_indexes, right_model_indexes)
                bce_loss = criterion(logits, labels)
                U_i = irt_model.model_factors(model_indexes)
                right_safe = right_model_indexes.clamp(min=0)
                right_mask = (right_model_indexes >= 0).to(U_i.dtype).unsqueeze(-1)
                U_i = U_i - irt_model.model_factors(right_safe) * right_mask
                V_j = torch.softmax(irt_model.item_loading_logits(item_indexes), dim=-1)
                l2_penalty = U_i.pow(2).mean() + V_j.pow(2).mean()
                loss = bce_loss + self.irt_l2 * l2_penalty
                loss.backward()
                optimizer.step()

                total_loss += float(bce_loss.detach().cpu()) * len(labels)
                total_count += len(labels)

                if batch_number % LOG_EVERY_BATCHES == 0:
                    running_loss = total_loss / max(total_count, 1)
                    print(
                        f"irt_epoch={epoch + 1} batch={batch_number}/{len(loader)} "
                        f"batch_bce={float(bce_loss.detach().cpu()):.6f} "
                        f"running_bce={running_loss:.6f}",
                        flush=True,
                    )

            mean_loss = total_loss / max(total_count, 1)
            print(
                f"irt_epoch={epoch + 1} train_bce={mean_loss:.6f} examples={total_count}",
                flush=True,
            )
        return irt_model

    def _fit_model_side_encoder(
        self,
        model_encoder,
        indexed: dict,
        irt_model: IRTFactorModel,
    ) -> None:
        print("Training metadata -> U_i predictor with L1 loss...", flush=True)
        targets = irt_model.model_factors.weight.detach()
        model_examples = indexed["model_examples"]
        loader = DataLoader(
            list(range(len(model_examples))),
            batch_size=min(self.batch_size, len(model_examples)),
            shuffle=True,
        )
        print(
            f"Metadata setup: models={len(model_examples)} batches={len(loader)} "
            f"lr={self.learning_rate} weight_decay={self.weight_decay}",
            flush=True,
        )
        optimizer = AdamW(model_encoder.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.L1Loss()

        for epoch in range(self.epochs):
            model_encoder.train()
            total_loss = 0.0
            total_count = 0
            for batch_number, batch_indexes in enumerate(loader, start=1):
                optimizer.zero_grad()
                predictions = []
                batch_targets = []
                for model_index in batch_indexes.tolist():
                    metadata = model_examples[model_index]
                    predictions.append(model_encoder(metadata))
                    batch_targets.append(targets[model_index].to(self.device))

                prediction_tensor = torch.stack(predictions)
                target_tensor = torch.stack(batch_targets)
                loss = criterion(prediction_tensor, target_tensor)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu()) * len(batch_indexes)
                total_count += len(batch_indexes)

            print(
                f"model_side_epoch={epoch + 1} l1={total_loss / max(total_count, 1):.6f} "
                f"models={total_count}",
                flush=True,
            )

    def _fit_item_side_encoder(
        self,
        item_encoder,
        indexed: dict,
        irt_model: IRTFactorModel,
        item_cache: dict[str, torch.Tensor],
    ) -> None:
        print("Training item content -> V_j,z_j with frozen IRT U_i targets...", flush=True)
        U_targets = irt_model.model_factors.weight.detach()
        dataset = TensorDataset(
            indexed["model_indexes"],
            indexed["right_model_indexes"],
            indexed["item_indexes"],
            indexed["labels"],
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        print(
            f"Item-side setup: examples={len(indexed['labels'])} batches={len(loader)} "
            f"cached_items={len(item_cache)} lr={self.learning_rate} "
            f"weight_decay={self.weight_decay}",
            flush=True,
        )
        optimizer = AdamW(
            list(item_encoder.loading_head.parameters()) + list(item_encoder.bias_head.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            item_encoder.train()
            total_loss = 0.0
            total_count = 0
            for batch_number, (
                model_indexes,
                right_model_indexes,
                item_indexes,
                labels,
            ) in enumerate(loader, start=1):
                optimizer.zero_grad()
                logits = []
                for model_index, right_model_index, item_index in zip(
                    model_indexes.tolist(),
                    right_model_indexes.tolist(),
                    item_indexes.tolist(),
                ):
                    item_key = indexed["item_keys"][item_index]
                    U_i = U_targets[model_index].to(self.device)
                    if right_model_index >= 0:
                        U_i = U_i - U_targets[right_model_index].to(self.device)
                    V_j, z_j = item_encoder.forward_from_representations(item_cache[item_key])
                    logits.append(torch.dot(U_i, V_j) + z_j)

                logits_tensor = torch.stack(logits)
                labels_tensor = labels.to(logits_tensor.device)
                loss = criterion(logits_tensor, labels_tensor)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu()) * len(labels)
                total_count += len(labels)

                if batch_number % LOG_EVERY_BATCHES == 0:
                    running_loss = total_loss / max(total_count, 1)
                    print(
                        f"item_epoch={epoch + 1} batch={batch_number}/{len(loader)} "
                        f"batch_bce={float(loss.detach().cpu()):.6f} "
                        f"running_bce={running_loss:.6f}",
                        flush=True,
                    )

            print(
                f"item_epoch={epoch + 1} train_bce={total_loss / max(total_count, 1):.6f} "
                f"examples={total_count}",
                flush=True,
            )

    def _build_indexed_training_data(self, examples: list[dict]) -> dict:
        model_to_index = {}
        item_to_index = {}
        model_examples = []
        item_infos = {}
        model_indexes = []
        right_model_indexes = []
        item_indexes = []
        labels = []

        def ensure_model(metadata: dict) -> int:
            model_key = self._metadata_cache_key(metadata)
            if model_key not in model_to_index:
                model_to_index[model_key] = len(model_examples)
                model_examples.append(metadata)
            return model_to_index[model_key]

        for example in examples:
            pairwise = is_pairwise_input(example)
            model_index = ensure_model(extract_left_model_metadata(example))
            right_model_index = (
                ensure_model(extract_right_model_metadata(example))
                if pairwise
                else -1
            )

            item_key = self._item_cache_key(example)
            if item_key not in item_to_index:
                item_to_index[item_key] = len(item_infos)
                item_infos[item_key] = {
                    "benchmark": example.get("benchmark"),
                    "condition": example.get("condition"),
                    "item_content": example.get("item_content") or "",
                }

            model_indexes.append(model_index)
            right_model_indexes.append(right_model_index)
            item_indexes.append(item_to_index[item_key])
            labels.append(float(example["label"]))

        item_keys = [None] * len(item_infos)
        for key, index in item_to_index.items():
            item_keys[index] = key

        print(
            f"Indexed training data: models={len(model_to_index)} "
            f"items={len(item_to_index)} examples={len(labels)} "
            f"pairwise={sum(index >= 0 for index in right_model_indexes)}",
            flush=True,
        )
        return {
            "model_to_index": model_to_index,
            "item_to_index": item_to_index,
            "model_examples": model_examples,
            "item_infos": item_infos,
            "item_keys": item_keys,
            "model_indexes": torch.tensor(model_indexes, dtype=torch.long),
            "right_model_indexes": torch.tensor(right_model_indexes, dtype=torch.long),
            "item_indexes": torch.tensor(item_indexes, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }

    def predict(self, input: dict, labeled: list[dict] | None = None) -> float:
        if self.scorer is None:
            raise RuntimeError("DouglasModel must be trained before predict().")
        return self.scorer.predict_probability(input)

    def save(self, path: Path = ARTIFACT_PATH) -> None:
        if self.scorer is None:
            raise RuntimeError("DouglasModel must be trained before saving.")
        save_checkpoint(
            path=path,
            scorer=self.scorer,
            config={
                "k": self.k,
                "p": self.p,
                "encoder_name": QUESTION_ENCODER_NAME,
                "item_head_hidden_dim": self.item_head_hidden_dim,
                "temperature": self.temperature,
            },
        )
        print(f"Saved checkpoint to {path}")

    def _precompute_item_representations_from_infos(
        self,
        item_infos: dict[str, dict],
    ) -> dict[str, torch.Tensor]:
        if self.scorer is None:
            raise RuntimeError("DouglasModel must initialize scorer before caching items.")

        if ITEM_CACHE_PATH.exists():
            print(f"Loading cached item representations from {ITEM_CACHE_PATH}", flush=True)
            item_cache = torch.load(ITEM_CACHE_PATH, map_location=self.device, weights_only=False)
            item_cache = {
                key: value.to(self.device)
                for key, value in item_cache.items()
            }
            missing_keys = [key for key in item_infos if key not in item_cache]
            if not missing_keys:
                print(
                    f"Loaded complete item cache: {len(item_cache)} representations.",
                    flush=True,
                )
                return item_cache
            print(
                f"Loaded {len(item_cache)} cached item representations, "
                f"but {len(missing_keys)} required items are missing; recomputing cache.",
                flush=True,
            )

        print(
            f"Encoding {len(item_infos)} unique items in batches of {self.encode_batch_size}...",
            flush=True,
        )

        item_cache = {}
        item_keys = list(item_infos)
        self.scorer.item_encoder.eval()
        for start in range(0, len(item_keys), self.encode_batch_size):
            batch_keys = item_keys[start:start + self.encode_batch_size]
            batch_infos = [item_infos[key] for key in batch_keys]
            with torch.no_grad():
                batch_representations = (
                    self.scorer.item_encoder.encode_sentence_representations_batch(batch_infos)
                )
            for key, representation in zip(batch_keys, batch_representations):
                item_cache[key] = representation.detach()

            if len(item_cache) % LOG_EVERY_UNIQUE_ITEMS < self.encode_batch_size:
                print(
                    f"cached item representations for {len(item_cache)}/{len(item_infos)} unique items",
                    flush=True,
                )

        ITEM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(item_cache, ITEM_CACHE_PATH)
        print(f"Saved item representation cache to {ITEM_CACHE_PATH}", flush=True)
        print(f"Cached {len(item_cache)} unique item representations.", flush=True)
        return item_cache

    def _item_cache_key(self, example: dict) -> str:
        item_key = example.get("item_id") or example.get("item_content") or ""
        benchmark = example.get("benchmark") or ""
        condition = example.get("condition") or ""
        return f"{benchmark}::{condition}::{item_key}"

    def _model_cache_key(self, example: dict) -> str:
        return str(example.get("model_id") or example.get("subject_content") or "")

    def _metadata_cache_key(self, metadata: dict) -> str:
        return str(metadata.get("model_id") or metadata.get("name") or metadata)

    def _trainable_parameters(self):
        if self.scorer is None:
            return []
        return (
            list(self.scorer.model_encoder.parameters())
            + list(self.scorer.item_encoder.loading_head.parameters())
            + list(self.scorer.item_encoder.bias_head.parameters())
        )

    def _materialize_binary_examples(self, examples) -> list[dict]:
        binary = []
        seen = 0
        skipped = 0
        print("Loading and filtering training examples...", flush=True)
        for example in examples:
            seen += 1
            if example.get("label") in (0, 1, 0.0, 1.0):
                binary.append(example)
            else:
                skipped += 1
            if seen % LOG_EVERY_EXAMPLES == 0:
                print(
                    f"loaded={seen} binary={len(binary)} skipped_non_binary={skipped}",
                    flush=True,
                )
        print(
            f"Using {len(binary)}/{seen} binary examples; skipped_non_binary={skipped}.",
            flush=True,
        )
        return binary


def main() -> None:
    train_data, _valid_data, test_data = load_split_data(
        split_ratios=(0.8, 0.0, 0.2),
    )

    print(f"Train benchmarks: {train_data['benchmark_ids']}")
    print(f"Test benchmarks: {test_data['benchmark_ids']}")

    model = DouglasModel(
        k=4,
        p=8,
        learning_rate=1e-4,
        weight_decay=1e-4,
        epochs=1,
        batch_size=64,
        encode_batch_size=128,
        temperature=1.0,
    )

    print("Training DouglasModel...")
    model.train(train_data["examples"])
    model.save()

    print("Evaluating on held-out test benchmarks...")
    test_metrics = evaluate(model.predict, test_data["examples"])
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
