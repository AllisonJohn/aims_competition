"""Run Douglas light-mini training on Modal.

Local setup:
    pip install modal
    modal setup

Default cheap smoke test:
    modal run competition/douglas/modal_train_light_mini.py

MiniLM smoke test:
    modal run competition/douglas/modal_train_light_mini.py --item-encoder minilm

BGE-large smoke test:
    modal run competition/douglas/modal_train_light_mini.py --item-encoder bge-large --use-gpu

K=8 smoke test:
    modal run competition/douglas/modal_train_light_mini.py --latent-dim 8
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "douglas-light-mini-training"
REMOTE_ROOT = Path("/root")
LOCAL_COMPETITION_DIR = Path(__file__).resolve().parents[1]
LOCAL_DOUGLAS_DIR = LOCAL_COMPETITION_DIR / "douglas"
LOCAL_UTILS_DIR = LOCAL_COMPETITION_DIR / "utils"
LOCAL_ARTIFACT_DIR = LOCAL_DOUGLAS_DIR / "artifacts"


app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "huggingface_hub",
        "hf_xet",
        "pyarrow",
        "numpy",
    )
    .add_local_dir(LOCAL_DOUGLAS_DIR, remote_path="/root/competition/douglas")
    .add_local_dir(LOCAL_UTILS_DIR, remote_path="/root/competition/utils")
)

cache_vol = modal.Volume.from_name("predeval-cache", create_if_missing=True)


def _remote_train_light_mini(
    item_encoder: str,
    train_items: int,
    test_items: int,
    train_rows: int,
    test_rows: int,
    epochs: int,
    batch_size: int,
    encode_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    irt_l2: float,
    item_head_hidden_dim: int,
    item_head_residual: bool,
    latent_dim: int,
    artifact_suffix: str,
) -> dict:
    import os
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(REMOTE_ROOT))
    os.environ.setdefault("HF_HOME", "/cache/hf")

    import competition.douglas.training as full_training
    from competition.douglas.training_light import (
        LIGHT_CONFIGS,
        LightDouglasModel,
        artifact_path_for_latent_dim,
    )
    from competition.douglas.training_light_mini import take_binary_item_subset
    from competition.utils.load_train_data import evaluate, load_split_data

    config = LIGHT_CONFIGS[item_encoder]
    artifact_name = artifact_path_for_latent_dim(
        Path(f"douglas_model_light_mini_{item_encoder}{artifact_suffix}.pt"),
        latent_dim,
    ).name
    artifact_path = Path(f"/tmp/{artifact_name}")
    item_cache_path = Path(f"/cache/douglas/item_representations_light_mini_{item_encoder}.pt")
    item_cache_path.parent.mkdir(parents=True, exist_ok=True)

    full_training.QUESTION_ENCODER_NAME = config["encoder_name"]
    full_training.ITEM_CACHE_PATH = item_cache_path
    if config["item_encoder_cls"] is not None:
        full_training.ItemQuestionEncoder = config["item_encoder_cls"]

    train_data, _valid_data, test_data = load_split_data(
        split_ratios=(0.8, 0.0, 0.2),
    )
    train_examples = take_binary_item_subset(
        train_data["examples"],
        max_unique_items=train_items,
        max_rows=train_rows,
    )
    test_examples = take_binary_item_subset(
        test_data["examples"],
        max_unique_items=test_items,
        max_rows=test_rows,
    )

    print("Running Modal Douglas light mini training check.", flush=True)
    print(f"Light mini variant: {item_encoder}", flush=True)
    print(f"Using item encoder: {config['encoder_name']}", flush=True)
    print(f"Item encoder class: {full_training.ItemQuestionEncoder.__name__}", flush=True)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    print(f"Train benchmarks: {train_data['benchmark_ids']}", flush=True)
    print(f"Test benchmarks: {test_data['benchmark_ids']}", flush=True)
    print(
        f"Mini limits: train_items={train_items} train_rows={len(train_examples)} "
        f"test_items={test_items} test_rows={len(test_examples)} epochs={epochs} "
        f"batch_size={batch_size} encode_batch_size={encode_batch_size} "
        f"latent_dim={latent_dim} item_head_hidden_dim={item_head_hidden_dim} "
        f"item_head_residual={item_head_residual} learning_rate={learning_rate} "
        f"irt_l2={irt_l2} weight_decay={weight_decay} temperature={temperature}",
        flush=True,
    )
    print(f"Modal item cache: {item_cache_path}", flush=True)
    print(f"Modal artifact: {artifact_path}", flush=True)

    model = LightDouglasModel(
        k=latent_dim,
        p=8,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        encode_batch_size=encode_batch_size,
        item_head_hidden_dim=item_head_hidden_dim,
        item_head_residual=item_head_residual,
        temperature=temperature,
        irt_l2=irt_l2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("Training Modal mini light DouglasModel...", flush=True)
    model.train(train_examples)
    model.save(artifact_path)

    print("Evaluating Modal mini light model on capped held-out test rows...", flush=True)
    metrics = evaluate(model.predict, test_examples)
    print(f"Modal mini light {item_encoder} test metrics: {metrics}", flush=True)
    cache_vol.commit()

    return {
        "artifact_bytes": artifact_path.read_bytes(),
        "artifact_name": artifact_name,
        "metrics": metrics,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "temperature": temperature,
            "irt_l2": irt_l2,
            "item_head_hidden_dim": item_head_hidden_dim,
            "latent_dim": latent_dim,
        },
        "train_benchmarks": train_data["benchmark_ids"],
        "test_benchmarks": test_data["benchmark_ids"],
    }


@app.function(
    image=image,
    volumes={"/cache": cache_vol},
    cpu=4.0,
    memory=32768,
    timeout=8 * 60 * 60,
)
def train_light_mini_cpu_remote(**kwargs) -> dict:
    return _remote_train_light_mini(**kwargs)


@app.function(
    image=image,
    volumes={"/cache": cache_vol},
    gpu="H100",
    cpu=8.0,
    memory=65536,
    timeout=8 * 60 * 60,
)
def train_light_mini_gpu_remote(**kwargs) -> dict:
    return _remote_train_light_mini(**kwargs)


def _parse_float_grid(value: str, fallback: float) -> list[float]:
    if not value.strip():
        return [fallback]
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_int_grid(value: str, fallback: int) -> list[int]:
    if not value.strip():
        return [fallback]
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _suffix_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


@app.local_entrypoint()
def main(
    item_encoder: str = "features",
    train_items: int = 256,
    test_items: int = 128,
    train_rows: int = 20_000,
    test_rows: int = 5_000,
    epochs: int = 1,
    batch_size: int = 256,
    encode_batch_size: int = 0,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 1.0,
    irt_l2: float = 1e-4,
    item_head_hidden_dim: int = 128,
    item_head_residual: bool = True,
    latent_dim: int = 4,
    use_gpu: bool = False,
    learning_rates: str = "",
    irt_l2s: str = "",
    weight_decays: str = "",
    item_head_hidden_dims: str = "",
) -> None:
    if item_encoder not in {"features", "minilm", "bge-large"}:
        raise ValueError("item_encoder must be one of: features, minilm, bge-large")

    if encode_batch_size <= 0:
        if item_encoder == "features":
            encode_batch_size = 4096
        elif item_encoder == "bge-large":
            encode_batch_size = 96
        else:
            encode_batch_size = 256

    remote_fn = train_light_mini_gpu_remote if use_gpu or item_encoder != "features" else train_light_mini_cpu_remote
    learning_rate_grid = _parse_float_grid(learning_rates, learning_rate)
    irt_l2_grid = _parse_float_grid(irt_l2s, irt_l2)
    weight_decay_grid = _parse_float_grid(weight_decays, weight_decay)
    hidden_dim_grid = _parse_int_grid(item_head_hidden_dims, item_head_hidden_dim)

    results = []
    for lr in learning_rate_grid:
        for l2 in irt_l2_grid:
            for wd in weight_decay_grid:
                for hidden_dim in hidden_dim_grid:
                    artifact_suffix = (
                        f"_lr{_suffix_float(lr)}"
                        f"_l2{_suffix_float(l2)}"
                        f"_wd{_suffix_float(wd)}"
                        f"_h{hidden_dim}"
                    )
                    kwargs = {
                        "item_encoder": item_encoder,
                        "train_items": train_items,
                        "test_items": test_items,
                        "train_rows": train_rows,
                        "test_rows": test_rows,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "encode_batch_size": encode_batch_size,
                        "learning_rate": lr,
                        "weight_decay": wd,
                        "temperature": temperature,
                        "irt_l2": l2,
                        "item_head_hidden_dim": hidden_dim,
                        "item_head_residual": item_head_residual,
                        "latent_dim": latent_dim,
                        "artifact_suffix": artifact_suffix,
                    }
                    out = remote_fn.remote(**kwargs)
                    local_artifact_path = LOCAL_ARTIFACT_DIR / out["artifact_name"]
                    local_artifact_path.parent.mkdir(parents=True, exist_ok=True)
                    local_artifact_path.write_bytes(out["artifact_bytes"])
                    out.pop("artifact_bytes")
                    out["local_artifact_path"] = str(local_artifact_path)
                    results.append(out)
                    print(f"Wrote artifact to {local_artifact_path}")
                    print(f"Test metrics: {out['metrics']}")

    best = max(
        results,
        key=lambda result: result["metrics"].get("negative_log_loss", float("-inf")),
    )
    summary_path = LOCAL_ARTIFACT_DIR / f"douglas_light_mini_{item_encoder}_sweep_summary.json"
    summary_path.write_text(json.dumps({"results": results, "best": best}, indent=2, sort_keys=True))

    print(f"Train benchmarks: {best['train_benchmarks']}")
    print(f"Test benchmarks: {best['test_benchmarks']}")
    print(f"Best artifact: {best['local_artifact_path']}")
    print(f"Best hyperparameters: {best['hyperparameters']}")
    print(f"Best metrics: {best['metrics']}")
    print(f"Wrote sweep summary to {summary_path}")
