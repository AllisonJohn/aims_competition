"""Small-subset smoke test for Douglas's light training path."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import competition.douglas.training as full_training  # noqa: E402
from competition.douglas.modeling import ITEM_HEAD_HIDDEN_DIM  # noqa: E402
from competition.douglas.training_light import LIGHT_CONFIGS, LightDouglasModel  # noqa: E402
from competition.utils.load_train_data import evaluate, load_split_data  # noqa: E402


MINI_ARTIFACT_PATHS = {
    "features": Path(__file__).with_name("artifacts") / "douglas_model_light_mini_features.pt",
    "minilm": Path(__file__).with_name("artifacts") / "douglas_model_light_mini_minilm.pt",
}
MINI_ITEM_CACHE_PATHS = {
    "features": Path(__file__).with_name("artifacts") / "item_feature_representations_light_mini_v1.pt",
    "minilm": Path(__file__).with_name("artifacts") / "item_representations_light_mini_minilm_v1.pt",
}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
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
        description="Run a tiny Douglas light training/eval smoke test.",
    )
    parser.add_argument(
        "--item-encoder",
        choices=sorted(LIGHT_CONFIGS),
        default=os.environ.get("LIGHT_ITEM_ENCODER", "features"),
        help="Same encoder switch as training_light.py.",
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
        default=env_optional_int("DOUGLAS_MINI_ENCODE_BATCH_SIZE"),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--irt-l2", type=float, default=1e-4)
    parser.add_argument(
        "--item-head-hidden-dim",
        type=int,
        default=env_int("DOUGLAS_ITEM_HEAD_HIDDEN_DIM", ITEM_HEAD_HIDDEN_DIM),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LIGHT_CONFIGS[args.item_encoder]
    artifact_path = MINI_ARTIFACT_PATHS[args.item_encoder]
    item_cache_path = MINI_ITEM_CACHE_PATHS[args.item_encoder]
    encode_batch_size = args.encode_batch_size or config["encode_batch_size"]

    full_training.QUESTION_ENCODER_NAME = config["encoder_name"]
    full_training.ITEM_CACHE_PATH = item_cache_path
    if config["item_encoder_cls"] is not None:
        full_training.ItemQuestionEncoder = config["item_encoder_cls"]

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

    print("Running Douglas light mini training check.", flush=True)
    print(f"Light mini variant: {args.item_encoder}", flush=True)
    print(f"Using item encoder: {config['encoder_name']}", flush=True)
    print(f"Item encoder class: {full_training.ItemQuestionEncoder.__name__}", flush=True)
    print(
        f"Training args: learning_rate={args.learning_rate} "
        f"weight_decay={args.weight_decay} irt_l2={args.irt_l2} "
        f"temperature={args.temperature} "
        f"item_head_hidden_dim={args.item_head_hidden_dim}",
        flush=True,
    )
    print(f"Train benchmarks: {train_data['benchmark_ids']}", flush=True)
    print(f"Test benchmarks: {test_data['benchmark_ids']}", flush=True)
    print(
        f"Mini limits: train_items={args.train_items} train_rows={len(train_examples)} "
        f"test_items={args.test_items} test_rows={len(test_examples)} epochs={args.epochs} "
        f"batch_size={args.batch_size} encode_batch_size={encode_batch_size}",
        flush=True,
    )
    print(f"Mini light item cache: {item_cache_path}", flush=True)
    print(f"Mini light artifact: {artifact_path}", flush=True)

    model = LightDouglasModel(
        k=4,
        p=8,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encode_batch_size=encode_batch_size,
        item_head_hidden_dim=args.item_head_hidden_dim,
        temperature=args.temperature,
        irt_l2=args.irt_l2,
    )

    print("Training mini light DouglasModel...", flush=True)
    model.train(train_examples)
    model.save(artifact_path)

    print("Evaluating mini light model on capped held-out test rows...", flush=True)
    test_metrics = evaluate(model.predict, test_examples)
    print(f"Mini light {args.item_encoder} test metrics: {test_metrics}", flush=True)


if __name__ == "__main__":
    main()
