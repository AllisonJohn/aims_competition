"""MiniLM-based smoke test for Douglas's light training path."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import competition.douglas.training as full_training  # noqa: E402
from competition.douglas.training import DouglasModel  # noqa: E402
from competition.utils.load_train_data import evaluate, load_split_data  # noqa: E402


MINILM_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MINI_LIGHT_ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "douglas_model_light_mini.pt"
MINI_LIGHT_ITEM_CACHE_PATH = (
    Path(__file__).with_name("artifacts") / "item_representations_light_mini_minilm_v1.pt"
)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def item_subset_key(example: dict) -> str:
    item_key = example.get("item_id") or example.get("item_content") or ""
    benchmark = example.get("benchmark") or ""
    condition = example.get("condition") or ""
    return f"{benchmark}::{condition}::{item_key}"


def take_binary_item_subset(
    examples,
    max_unique_items: int,
    max_rows: int,
) -> list[dict]:
    selected_items: set[str] = set()
    kept: list[dict] = []
    seen = 0
    skipped_non_binary = 0
    skipped_new_items = 0

    for example in examples:
        seen += 1
        label = example.get("label")
        if label not in (0, 1, 0.0, 1.0):
            skipped_non_binary += 1
            continue

        key = item_subset_key(example)
        if key not in selected_items:
            if len(selected_items) >= max_unique_items:
                skipped_new_items += 1
                continue
            selected_items.add(key)

        kept.append(example)
        if len(kept) >= max_rows:
            break

    print(
        f"Subset scan: seen={seen} kept={len(kept)} unique_items={len(selected_items)} "
        f"skipped_non_binary={skipped_non_binary} skipped_new_items={skipped_new_items}",
        flush=True,
    )
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny MiniLM Douglas training/eval smoke test.",
    )
    parser.add_argument("--train-items", type=int, default=env_int("DOUGLAS_MINI_TRAIN_ITEMS", 256))
    parser.add_argument("--test-items", type=int, default=env_int("DOUGLAS_MINI_TEST_ITEMS", 128))
    parser.add_argument("--train-rows", type=int, default=env_int("DOUGLAS_MINI_TRAIN_ROWS", 20_000))
    parser.add_argument("--test-rows", type=int, default=env_int("DOUGLAS_MINI_TEST_ROWS", 5_000))
    parser.add_argument("--epochs", type=int, default=env_int("DOUGLAS_MINI_EPOCHS", 1))
    parser.add_argument("--batch-size", type=int, default=env_int("DOUGLAS_MINI_BATCH_SIZE", 256))
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=env_int("DOUGLAS_MINI_ENCODE_BATCH_SIZE", 256),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--irt-l2", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    full_training.QUESTION_ENCODER_NAME = MINILM_ENCODER_NAME
    full_training.ITEM_CACHE_PATH = MINI_LIGHT_ITEM_CACHE_PATH

    train_data, _valid_data, test_data = load_split_data(
        split_ratios=(0.8, 0.0, 0.2),
    )
    train_examples = take_binary_item_subset(
        train_data["examples"],
        max_unique_items=args.train_items,
        max_rows=args.train_rows,
    )
    test_examples = take_binary_item_subset(
        test_data["examples"],
        max_unique_items=args.test_items,
        max_rows=args.test_rows,
    )

    print("Running Douglas MiniLM mini training check.", flush=True)
    print(f"Using item encoder: {MINILM_ENCODER_NAME}", flush=True)
    print(f"Train benchmarks: {train_data['benchmark_ids']}", flush=True)
    print(f"Test benchmarks: {test_data['benchmark_ids']}", flush=True)
    print(
        f"Mini limits: train_items={args.train_items} train_rows={len(train_examples)} "
        f"test_items={args.test_items} test_rows={len(test_examples)} epochs={args.epochs} "
        f"batch_size={args.batch_size} encode_batch_size={args.encode_batch_size}",
        flush=True,
    )
    print(f"Mini light item cache: {MINI_LIGHT_ITEM_CACHE_PATH}", flush=True)
    print(f"Mini light artifact: {MINI_LIGHT_ARTIFACT_PATH}", flush=True)

    model = DouglasModel(
        k=4,
        p=8,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        temperature=args.temperature,
        irt_l2=args.irt_l2,
    )

    print("Training mini MiniLM DouglasModel...", flush=True)
    model.train(train_examples)
    model.save(MINI_LIGHT_ARTIFACT_PATH)

    print("Evaluating mini MiniLM model on capped held-out test rows...", flush=True)
    test_metrics = evaluate(model.predict, test_examples)
    print(f"Mini MiniLM test metrics: {test_metrics}", flush=True)


if __name__ == "__main__":
    main()
