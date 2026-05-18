"""Run Douglas training remotely on Modal.

Local setup:
    pip install modal
    modal setup

Run:
    modal run competition/douglas/modal_train.py
"""

from __future__ import annotations

from pathlib import Path

import modal


APP_NAME = "douglas-predeval-training"
REMOTE_ROOT = Path("/root")
LOCAL_COMPETITION_DIR = Path(__file__).resolve().parents[1]
LOCAL_DOUGLAS_DIR = LOCAL_COMPETITION_DIR / "douglas"
LOCAL_UTILS_DIR = LOCAL_COMPETITION_DIR / "utils"
LOCAL_ARTIFACT_PATH = LOCAL_DOUGLAS_DIR / "artifacts" / "douglas_model.pt"


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
def train_remote(
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    temperature: float = 1.0,
) -> dict:
    import os
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(REMOTE_ROOT))
    os.environ.setdefault("HF_HOME", "/cache/hf")

    from competition.douglas.training import DouglasModel
    from competition.utils.load_train_data import evaluate, load_split_data

    train_data, valid_data, test_data = load_split_data(
        split_ratios=(0.8, 0.0, 0.2),
    )
    print(f"Train benchmarks: {train_data['benchmark_ids']}", flush=True)
    if valid_data is not None:
        print(f"Validation benchmarks: {valid_data['benchmark_ids']}", flush=True)
    print(f"Test benchmarks: {test_data['benchmark_ids']}", flush=True)

    model = DouglasModel(
        k=5,
        p=8,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        temperature=temperature,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.train(train_data["examples"])

    remote_artifact = Path("/tmp/douglas_model.pt")
    model.save(remote_artifact)

    metrics = evaluate(model.predict, test_data["examples"])
    print(f"Test metrics: {metrics}", flush=True)
    cache_vol.commit()

    return {
        "artifact_bytes": remote_artifact.read_bytes(),
        "metrics": metrics,
        "train_benchmarks": train_data["benchmark_ids"],
        "test_benchmarks": test_data["benchmark_ids"],
    }


@app.local_entrypoint()
def main(
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    temperature: float = 1.0,
) -> None:
    out = train_remote.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        temperature=temperature,
    )

    LOCAL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_ARTIFACT_PATH.write_bytes(out["artifact_bytes"])

    print(f"Wrote artifact to {LOCAL_ARTIFACT_PATH}")
    print(f"Train benchmarks: {out['train_benchmarks']}")
    print(f"Test benchmarks: {out['test_benchmarks']}")
    print(f"Test metrics: {out['metrics']}")
