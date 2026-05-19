"""Run Douglas submit training on Modal.

Local setup:
    pip install modal
    modal setup

Train both submit artifacts separately:
    modal run competition/douglas/modal_train_submit.py --item-encoder features
    modal run competition/douglas/modal_train_submit.py --item-encoder lm
"""

from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "douglas-submit-training"
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


@app.function(
    image=image,
    volumes={"/cache": cache_vol},
    gpu="H100",
    cpu=8.0,
    memory=65536,
    timeout=8 * 60 * 60,
)
def train_submit_remote(
    item_encoder: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    irt_l2: float,
    item_head_hidden_dim: int,
    limit: int,
) -> dict:
    import os
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_ROOT))
    os.environ.setdefault("HF_HOME", "/cache/hf")

    import competition.douglas.training as full_training
    from competition.douglas.train_submit import SUBMIT_CONFIGS
    from competition.douglas.training_light import LightDouglasModel
    from competition.utils.load_train_data import get_training_data

    config = SUBMIT_CONFIGS[item_encoder]
    remote_artifact_path = Path(f"/tmp/{config['artifact_path'].name}")
    remote_item_cache_path = Path(f"/cache/douglas/{config['item_cache_path'].name}")
    remote_item_cache_path.parent.mkdir(parents=True, exist_ok=True)

    full_training.QUESTION_ENCODER_NAME = config["encoder_name"]
    full_training.ITEM_CACHE_PATH = remote_item_cache_path
    if config["item_encoder_cls"] is not None:
        full_training.ItemQuestionEncoder = config["item_encoder_cls"]

    train_data = get_training_data(limit=limit or None)

    print("Running Modal Douglas submit training.", flush=True)
    print(f"Submit variant: {item_encoder}", flush=True)
    print(f"Using item encoder: {config['encoder_name']}", flush=True)
    print(f"Item encoder class: {full_training.ItemQuestionEncoder.__name__}", flush=True)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    print(f"Training benchmarks: {train_data['benchmark_ids']}", flush=True)
    print(
        f"Training args: epochs={epochs} batch_size={batch_size} "
        f"learning_rate={learning_rate} weight_decay={weight_decay} "
        f"irt_l2={irt_l2} temperature={temperature} "
        f"item_head_hidden_dim={item_head_hidden_dim} "
        f"encode_batch_size={config['encode_batch_size']} limit={limit or 'all'}",
        flush=True,
    )
    print(f"Modal item cache: {remote_item_cache_path}", flush=True)
    print(f"Modal artifact: {remote_artifact_path}", flush=True)

    model = LightDouglasModel(
        k=4,
        p=8,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        encode_batch_size=config["encode_batch_size"],
        item_head_hidden_dim=item_head_hidden_dim,
        temperature=temperature,
        irt_l2=irt_l2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("Training Modal submit DouglasModel...", flush=True)
    model.train(train_data["examples"])
    model.save(remote_artifact_path)
    cache_vol.commit()

    return {
        "artifact_bytes": remote_artifact_path.read_bytes(),
        "artifact_name": config["artifact_path"].name,
        "train_benchmarks": train_data["benchmark_ids"],
    }


@app.local_entrypoint()
def main(
    item_encoder: str = "features",
    epochs: int = 1,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    temperature: float = 0.7,
    irt_l2: float = 1e-3,
    item_head_hidden_dim: int = 64,
    limit: int = 0,
) -> None:
    if item_encoder not in {"features", "lm"}:
        raise ValueError("item_encoder must be one of: features, lm")

    out = train_submit_remote.remote(
        item_encoder=item_encoder,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        temperature=temperature,
        irt_l2=irt_l2,
        item_head_hidden_dim=item_head_hidden_dim,
        limit=limit,
    )

    local_artifact_path = LOCAL_ARTIFACT_DIR / out["artifact_name"]
    local_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    local_artifact_path.write_bytes(out["artifact_bytes"])

    print(f"Wrote artifact to {local_artifact_path}")
    print(f"Train benchmarks: {out['train_benchmarks']}")
