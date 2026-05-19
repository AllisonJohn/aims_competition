"""Full Douglas training run with cheap handcrafted item features."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import competition.douglas.training as full_training  # noqa: E402
from competition.douglas.modeling import HandcraftedItemQuestionEncoder  # noqa: E402
from competition.douglas.training import DouglasModel  # noqa: E402
from competition.utils.load_train_data import evaluate, load_split_data  # noqa: E402


LIGHT_QUESTION_ENCODER_NAME = "handcrafted-item-features"
LIGHT_ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "douglas_model_light.pt"
LIGHT_ITEM_CACHE_PATH = Path(__file__).with_name("artifacts") / "item_feature_representations_light.pt"


def main() -> None:
    full_training.QUESTION_ENCODER_NAME = LIGHT_QUESTION_ENCODER_NAME
    full_training.ITEM_CACHE_PATH = LIGHT_ITEM_CACHE_PATH
    full_training.ItemQuestionEncoder = HandcraftedItemQuestionEncoder

    train_data, _valid_data, test_data = load_split_data(
        split_ratios=(0.8, 0.0, 0.2),
    )

    print(f"Using item encoder: {LIGHT_QUESTION_ENCODER_NAME}", flush=True)
    print(f"Train benchmarks: {train_data['benchmark_ids']}", flush=True)
    print(f"Test benchmarks: {test_data['benchmark_ids']}", flush=True)
    print(f"Light item cache: {LIGHT_ITEM_CACHE_PATH}", flush=True)
    print(f"Light artifact: {LIGHT_ARTIFACT_PATH}", flush=True)

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

    print("Training light DouglasModel...", flush=True)
    model.train(train_data["examples"])
    model.save(LIGHT_ARTIFACT_PATH)

    print("Evaluating light model on held-out test benchmarks...", flush=True)
    test_metrics = evaluate(model.predict, test_data["examples"])
    print(f"Light test metrics: {test_metrics}", flush=True)


if __name__ == "__main__":
    main()
