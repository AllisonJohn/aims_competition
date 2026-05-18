"""Offline training script for Douglas's competition model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition.douglas.modeling import (  # noqa: E402
    MODEL_EMBED_DIM,
    MODEL_VECTOR_DIM,
    QUESTION_ENCODER_NAME,
    DouglasScorer,
    ItemQuestionEncoder,
    build_model_side_encoder,
    save_checkpoint,
)
from competition.utils.load_train_data import evaluate, load_split_data  # noqa: E402


ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "douglas_model.pt"
ITEM_CACHE_PATH = Path(__file__).with_name("artifacts") / "item_representations.pt"
LOG_EVERY_EXAMPLES = 100_000
LOG_EVERY_UNIQUE_ITEMS = 1_000
LOG_EVERY_BATCHES = 100


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
        temperature: float = 1.0,
        device: str | None = None,
    ) -> None:
        self.k = k
        self.p = p
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.encode_batch_size = encode_batch_size
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer: DouglasScorer | None = None

    def train(self, examples) -> None:
        examples = self._materialize_binary_examples(examples)
        model_encoder = build_model_side_encoder(examples, p=self.p, output_dim=self.k).to(self.device)
        item_encoder = ItemQuestionEncoder(
            encoder_name=QUESTION_ENCODER_NAME,
            loading_dim=self.k,
            freeze_backbone=True,
        ).to(self.device)
        self.scorer = DouglasScorer(
            model_encoder=model_encoder,
            item_encoder=item_encoder,
            temperature=self.temperature,
        ).to(self.device)

        item_cache = self._precompute_item_representations(examples)
        loader = DataLoader(
            list(range(len(examples))),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = AdamW(self._trainable_parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            self.scorer.train()
            total_loss = 0.0
            total_count = 0

            for batch_number, batch_indexes in enumerate(loader, start=1):
                optimizer.zero_grad()
                logits = []
                labels = []
                for index in batch_indexes.tolist():
                    example = examples[index]
                    item_key = self._item_cache_key(example)
                    logits.append(
                        self.scorer(
                            example,
                            item_representations=item_cache[item_key],
                        )
                    )
                    labels.append(float(example["label"]))

                logits_tensor = torch.stack(logits)
                labels_tensor = torch.tensor(labels, dtype=logits_tensor.dtype, device=logits_tensor.device)
                loss = criterion(logits_tensor, labels_tensor)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu()) * len(labels)
                total_count += len(labels)

                if batch_number % LOG_EVERY_BATCHES == 0:
                    running_loss = total_loss / max(total_count, 1)
                    print(
                        f"epoch={epoch + 1} batch={batch_number}/{len(loader)} "
                        f"running_bce={running_loss:.6f}",
                        flush=True,
                    )

            mean_loss = total_loss / max(total_count, 1)
            print(
                f"epoch={epoch + 1} train_bce={mean_loss:.6f} examples={total_count}",
                flush=True,
            )

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
                "temperature": self.temperature,
            },
        )
        print(f"Saved checkpoint to {path}")

    def _precompute_item_representations(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        if self.scorer is None:
            raise RuntimeError("DouglasModel must initialize scorer before caching items.")

        if ITEM_CACHE_PATH.exists():
            print(f"Loading cached item representations from {ITEM_CACHE_PATH}", flush=True)
            item_cache = torch.load(ITEM_CACHE_PATH, map_location=self.device, weights_only=False)
            item_cache = {
                key: value.to(self.device)
                for key, value in item_cache.items()
            }
            print(f"Loaded {len(item_cache)} cached item representations.", flush=True)
            return item_cache

        item_infos = {}
        for example in examples:
            item_key = self._item_cache_key(example)
            if item_key in item_infos:
                continue
            item_infos[item_key] = {
                "benchmark": example.get("benchmark"),
                "condition": example.get("condition"),
                "item_content": example.get("item_content") or "",
            }

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
        return str(example.get("item_id") or example.get("item_content") or "")

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
        k=5,
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
