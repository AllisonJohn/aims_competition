"""Train Douglas submit artifacts on all public training data.

Run once per encoder to produce artifacts that ``model.py`` can ensemble:

    python competition/douglas/train_submit.py --item-encoder features
    python competition/douglas/train_submit.py --item-encoder lm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import competition.douglas.training as full_training  # noqa: E402
from competition.douglas.modeling import ITEM_HEAD_HIDDEN_DIM, HandcraftedItemQuestionEncoder  # noqa: E402
from competition.douglas.training_light import LightDouglasModel  # noqa: E402
from competition.utils.load_train_data import get_training_data  # noqa: E402


SUBMIT_CONFIGS = {
    "features": {
        "encoder_name": "handcrafted-item-features",
        "artifact_path": Path(__file__).with_name("artifacts") / "douglas_submit_features.pt",
        "item_cache_path": Path(__file__).with_name("artifacts") / "item_feature_representations_submit_v1.pt",
        "item_encoder_cls": HandcraftedItemQuestionEncoder,
        "encode_batch_size": 4096,
    },
    "lm": {
        "encoder_name": "sentence-transformers/all-MiniLM-L6-v2",
        "artifact_path": Path(__file__).with_name("artifacts") / "douglas_submit_lm.pt",
        "item_cache_path": Path(__file__).with_name("artifacts") / "item_representations_submit_lm_context_v1.pt",
        "item_encoder_cls": None,
        "encode_batch_size": 256,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Douglas submit artifact on all public data.")
    parser.add_argument(
        "--item-encoder",
        choices=sorted(SUBMIT_CONFIGS),
        required=True,
        help="Submit artifact variant to train.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--irt-l2", type=float, default=1e-3)
    parser.add_argument("--item-head-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row cap for debugging. Use 0 for all rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SUBMIT_CONFIGS[args.item_encoder]

    full_training.QUESTION_ENCODER_NAME = config["encoder_name"]
    full_training.ITEM_CACHE_PATH = config["item_cache_path"]
    if config["item_encoder_cls"] is not None:
        full_training.ItemQuestionEncoder = config["item_encoder_cls"]

    train_data = get_training_data(limit=args.limit or None)

    print("Running Douglas submit training.", flush=True)
    print(f"Submit variant: {args.item_encoder}", flush=True)
    print(f"Using item encoder: {config['encoder_name']}", flush=True)
    print(f"Item encoder class: {full_training.ItemQuestionEncoder.__name__}", flush=True)
    print(f"Training benchmarks: {train_data['benchmark_ids']}", flush=True)
    print(
        f"Training args: epochs={args.epochs} batch_size={args.batch_size} "
        f"learning_rate={args.learning_rate} weight_decay={args.weight_decay} "
        f"irt_l2={args.irt_l2} temperature={args.temperature} "
        f"item_head_hidden_dim={args.item_head_hidden_dim} "
        f"encode_batch_size={config['encode_batch_size']} limit={args.limit or 'all'}",
        flush=True,
    )
    print(f"Submit item cache: {config['item_cache_path']}", flush=True)
    print(f"Submit artifact: {config['artifact_path']}", flush=True)

    model = LightDouglasModel(
        k=4,
        p=8,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encode_batch_size=config["encode_batch_size"],
        item_head_hidden_dim=args.item_head_hidden_dim,
        temperature=args.temperature,
        irt_l2=args.irt_l2,
    )

    print("Training submit DouglasModel...", flush=True)
    model.train(train_data["examples"])
    model.save(config["artifact_path"])
    print(f"Done. Saved submit artifact to {config['artifact_path']}", flush=True)


if __name__ == "__main__":
    main()
