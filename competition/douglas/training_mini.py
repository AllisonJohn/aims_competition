"""Small end-to-end training run for quickly checking Douglas's model path."""

from __future__ import annotations

import os
import sys
from itertools import islice
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import competition.douglas.training as full_training  # noqa: E402
from competition.douglas.training import DouglasModel  # noqa: E402
from competition.utils.load_train_data import evaluate, load_split_data  # noqa: E402


MINI_ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "douglas_model_mini.pt"
MINI_ITEM_CACHE_PATH = Path(__file__).with_name("artifacts") / "item_representations_mini_context_v2.pt"


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def take_examples(data: dict, limit: int) -> dict:
    return {
        **data,
        "examples": list(islice(data["examples"], limit)),
    }


def main() -> None:
    max_train_rows = env_int("DOUGLAS_MINI_TRAIN_ROWS", 20_000)
    max_test_rows = env_int("DOUGLAS_MINI_TEST_ROWS", 5_000)
    epochs = env_int("DOUGLAS_MINI_EPOCHS", 1)
    batch_size = env_int("DOUGLAS_MINI_BATCH_SIZE", 64)
    encode_batch_size = env_int("DOUGLAS_MINI_ENCODE_BATCH_SIZE", 128)

    full_training.ITEM_CACHE_PATH = MINI_ITEM_CACHE_PATH

    train_data, _valid_data, test_data = load_split_data(
        split_ratios=(0.8, 0.0, 0.2),
    )
    train_data = take_examples(train_data, max_train_rows)
    test_data = take_examples(test_data, max_test_rows)

    print("Running Douglas mini training check.", flush=True)
    print(f"Train benchmarks: {train_data['benchmark_ids']}", flush=True)
    print(f"Test benchmarks: {test_data['benchmark_ids']}", flush=True)
    print(
        f"Mini limits: train_rows={len(train_data['examples'])} "
        f"test_rows={len(test_data['examples'])} epochs={epochs} "
        f"batch_size={batch_size} encode_batch_size={encode_batch_size}",
        flush=True,
    )
    print(f"Mini item cache: {MINI_ITEM_CACHE_PATH}", flush=True)
    print(f"Mini artifact: {MINI_ARTIFACT_PATH}", flush=True)

    model = DouglasModel(
        k=4,
        p=8,
        learning_rate=1e-4,
        weight_decay=1e-4,
        epochs=epochs,
        batch_size=batch_size,
        encode_batch_size=encode_batch_size,
        temperature=1.0,
    )

    print("Training mini DouglasModel...", flush=True)
    model.train(train_data["examples"])
    model.save(MINI_ARTIFACT_PATH)

    print("Evaluating mini model on capped held-out test rows...", flush=True)
    test_metrics = evaluate(model.predict, test_data["examples"])
    print(f"Mini test metrics: {test_metrics}", flush=True)


if __name__ == "__main__":
    main()
