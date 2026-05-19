"""Switchable light Douglas training runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import competition.douglas.training as full_training  # noqa: E402
from competition.douglas.modeling import HandcraftedItemQuestionEncoder  # noqa: E402
from competition.douglas.training import DouglasModel  # noqa: E402
from competition.utils.load_train_data import evaluate, load_split_data  # noqa: E402


LIGHT_CONFIGS = {
    "features": {
        "encoder_name": "handcrafted-item-features",
        "artifact_path": Path(__file__).with_name("artifacts") / "douglas_model_light_features.pt",
        "item_cache_path": Path(__file__).with_name("artifacts") / "item_feature_representations_light_v2.pt",
        "item_encoder_cls": HandcraftedItemQuestionEncoder,
        "encode_batch_size": 4096,
    },
    "minilm": {
        "encoder_name": "sentence-transformers/all-MiniLM-L6-v2",
        "artifact_path": Path(__file__).with_name("artifacts") / "douglas_model_light_minilm.pt",
        "item_cache_path": Path(__file__).with_name("artifacts") / "item_representations_light_minilm.pt",
        "item_encoder_cls": None,
        "encode_batch_size": 256,
    },
}


class LightDouglasModel(DouglasModel):
    """Douglas model variant for cheap item features."""

    def _item_cache_key(self, example: dict) -> str:
        item_key = example.get("item_id") or example.get("item_content") or ""
        benchmark = example.get("benchmark") or ""
        condition = example.get("condition") or ""
        return f"{benchmark}::{condition}::{item_key}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a light Douglas model variant.")
    parser.add_argument(
        "--item-encoder",
        choices=sorted(LIGHT_CONFIGS),
        default="features",
        help="Item encoder variant to use.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LIGHT_CONFIGS[args.item_encoder]

    full_training.QUESTION_ENCODER_NAME = config["encoder_name"]
    full_training.ITEM_CACHE_PATH = config["item_cache_path"]
    if config["item_encoder_cls"] is not None:
        full_training.ItemQuestionEncoder = config["item_encoder_cls"]

    train_data, _valid_data, test_data = load_split_data(
        split_ratios=(0.8, 0.0, 0.2),
    )

    print(f"Light variant: {args.item_encoder}", flush=True)
    print(f"Using item encoder: {config['encoder_name']}", flush=True)
    print(f"Item encoder class: {full_training.ItemQuestionEncoder.__name__}", flush=True)
    print(f"Train benchmarks: {train_data['benchmark_ids']}", flush=True)
    print(f"Test benchmarks: {test_data['benchmark_ids']}", flush=True)
    print(f"Light item cache: {config['item_cache_path']}", flush=True)
    print(f"Light artifact: {config['artifact_path']}", flush=True)

    model = LightDouglasModel(
        k=5,
        p=8,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encode_batch_size=config["encode_batch_size"],
        temperature=args.temperature,
    )

    print("Training light DouglasModel...", flush=True)
    model.train(train_data["examples"])
    model.save(config["artifact_path"])

    print("Evaluating light model on held-out test benchmarks...", flush=True)
    test_metrics = evaluate(model.predict, test_data["examples"])
    print(f"Light {args.item_encoder} test metrics: {test_metrics}", flush=True)


if __name__ == "__main__":
    main()
