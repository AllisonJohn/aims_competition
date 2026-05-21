"""Validate the handcrafted-text-feature ridge submission with unseen benchmark holdout.

This script trains the same artifact family as ``modal_train_text_features_ridge.py``
on an 80% benchmark split, writes those artifacts beside ``model.py`` inside
the remote container, then imports ``model.py`` and evaluates by calling
``predict()`` on the held-out benchmark rows.

Run:
    modal run competition/douglas_ridge_text_features/modal_validate_text_features_ridge.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "douglas-ridge-text-features-validation"
REMOTE_ROOT = Path("/root")
LOCAL_RIDGE_DIR = Path(__file__).resolve().parent
LOCAL_ARTIFACT_DIR = LOCAL_RIDGE_DIR / "artifacts"

RESPONSE_FILES = [
    "afrimedqa.parquet",
    "agentdojo.parquet",
    "ai2d_test.parquet",
    "androidworld.parquet",
    "bfcl.parquet",
    "cybench.parquet",
    "hle.parquet",
    "livecodebench.parquet",
    "matharena.parquet",
    "mathvista_mini.parquet",
    "mmbench_v11.parquet",
    "mmlupro.parquet",
    "mtbench.parquet",
    "rewardbench.parquet",
    "swebench.parquet",
    "ultrafeedback.parquet",
]


app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "huggingface_hub",
        "hf_xet",
    )
    .add_local_dir(LOCAL_RIDGE_DIR, remote_path="/root/competition/douglas_ridge_text_features")
)

cache_vol = modal.Volume.from_name("predeval-cache", create_if_missing=True)


def _split_response_files(split_ratio: float = 0.8) -> tuple[list[str], list[str]]:
    files = sorted(RESPONSE_FILES)
    train_count = round(len(files) * split_ratio)
    train_count = max(1, min(train_count, len(files) - 1))
    return files[:train_count], files[train_count:]


@app.function(
    image=image,
    volumes={"/cache": cache_vol},
    cpu=8.0,
    memory=65536,
    timeout=8 * 60 * 60,
)
def validate_remote(
    train_rows: int = 1_500_000,
    valid_rows: int = 0,
    item_limit: int = 120_000,
    split_ratio: float = 0.8,
    blend_weights: str = "0,0.1,0.2,0.3,0.4,0.45,0.5,0.6,0.8,1.0",
) -> dict:
    import importlib.util
    import math
    import os
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import Ridge
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    sys.path.insert(0, str(REMOTE_ROOT))
    os.environ.setdefault("HF_HOME", "/cache/hf")

    repo_id = "aims-foundations/measurement-db"
    train_files, valid_files = _split_response_files(split_ratio)

    def parse_blend_weights(value: str) -> list[float]:
        weights = [float(part.strip()) for part in value.split(",") if part.strip()]
        return weights or [0.45]

    def clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
        return max(lo, min(hi, float(value)))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def parse_subject_name(subject_content: str) -> str:
        text = str(subject_content or "")
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip().lower() == "name":
                return value.strip().lower()
        return text.strip().splitlines()[0].lower() if text.strip() else ""

    def binary_label(benchmark: str, response: float) -> int:
        if benchmark == "mtbench":
            y = float(response) / 10.0
        elif benchmark == "ultrafeedback":
            y = (float(response) - 1.0) / 4.0
        else:
            y = float(response)
        return int(clamp(y, 0.0, 1.0) >= 0.5)

    def render_subject_content(subject: dict, fallback_subject_id: str) -> str:
        display_name = subject.get("display_name") or fallback_subject_id
        lines = [f"Name: {display_name}"]
        for key, label in (
            ("provider", "Organization"),
            ("params", "Parameters"),
            ("release_date", "Released"),
            ("family", "Family"),
        ):
            value = subject.get(key)
            if value:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)

    def load_response_frame(response_files: list[str], row_limit: int) -> pd.DataFrame:
        per_file = None if row_limit <= 0 else max(1, row_limit // len(response_files))
        parts = []
        for filename in response_files:
            path = hf_hub_download(repo_id, filename, repo_type="dataset")
            df = pd.read_parquet(
                path,
                columns=["subject_id", "item_id", "benchmark_id", "test_condition", "response"],
            )
            if per_file is not None and len(df) > per_file:
                df = df.sample(n=per_file, random_state=326)
            df["subject_content"] = df["subject_id"].map(subject_lookup).fillna("")
            df["subject_name"] = df["subject_content"].map(parse_subject_name)
            df["item_content"] = df["item_id"].map(item_lookup).fillna("")
            df["item_key"] = df["benchmark_id"].astype(str) + "::" + df["item_id"].astype(str)
            df["condition"] = df["test_condition"].fillna("none").astype(str)
            df["label"] = [
                binary_label(bench, response)
                for bench, response in zip(df["benchmark_id"], df["response"])
            ]
            parts.append(
                df[
                    [
                        "subject_id",
                        "subject_name",
                        "subject_content",
                        "item_key",
                        "benchmark_id",
                        "condition",
                        "item_content",
                        "label",
                    ]
                ]
            )
        out = pd.concat(parts, ignore_index=True)
        if row_limit > 0 and len(out) > row_limit:
            out = out.sample(n=row_limit, random_state=326)
        return out

    def rate_table(df: pd.DataFrame, column: str) -> dict[str, list[float]]:
        grouped = df.groupby(column)["label"].agg(["mean", "count"])
        return {
            str(index).strip().lower(): [float(row["mean"]), float(row["count"])]
            for index, row in grouped.iterrows()
        }

    subjects_path = hf_hub_download(repo_id, "subjects.parquet", repo_type="dataset")
    items_path = hf_hub_download(repo_id, "items.parquet", repo_type="dataset")
    subjects = pd.read_parquet(subjects_path)
    items = pd.read_parquet(items_path, columns=["item_id", "content"])
    subject_lookup = {
        row["subject_id"]: render_subject_content(row.to_dict(), row["subject_id"])
        for _, row in subjects.iterrows()
    }
    item_lookup = dict(zip(items["item_id"], items["content"]))

    print(f"Train files: {train_files}", flush=True)
    print(f"Validation files: {valid_files}", flush=True)
    train_df = load_response_frame(train_files, train_rows)
    valid_df = load_response_frame(valid_files, valid_rows)
    print(f"Loaded train rows={len(train_df)} valid rows={len(valid_df)}", flush=True)

    baseline_stats = {
        "global_rate": float(train_df["label"].mean()),
        "subject_rates": rate_table(train_df, "subject_name"),
        "condition_rates": rate_table(train_df, "condition"),
        "benchmark_rates": rate_table(train_df, "benchmark_id"),
        "row_count": int(len(train_df)),
    }

    subject_codes, subject_uniques = pd.factorize(train_df["subject_name"], sort=True)
    item_codes, item_uniques = pd.factorize(train_df["item_key"], sort=True)
    y = train_df["label"].to_numpy(dtype=np.float64)
    theta = np.zeros(len(subject_uniques), dtype=np.float64)
    beta = np.zeros(len(item_uniques), dtype=np.float64)

    subj_groups = [np.where(subject_codes == i)[0] for i in range(len(subject_uniques))]
    item_groups = [np.where(item_codes == i)[0] for i in range(len(item_uniques))]
    reg = 0.1
    for _ in range(8):
        eta = theta[subject_codes] - beta[item_codes]
        p = sigmoid(eta)
        for i, idx in enumerate(subj_groups):
            if len(idx) == 0:
                continue
            grad = (y[idx] - p[idx]).sum() - reg * theta[i]
            hess = -(p[idx] * (1 - p[idx])).sum() - reg
            theta[i] -= grad / hess
        theta -= theta.mean()

        eta = theta[subject_codes] - beta[item_codes]
        p = sigmoid(eta)
        for j, idx in enumerate(item_groups):
            if len(idx) == 0:
                continue
            grad = -(y[idx] - p[idx]).sum() - reg * beta[j]
            hess = -(p[idx] * (1 - p[idx])).sum() - reg
            beta[j] -= grad / hess

    item_targets = pd.DataFrame({"item_key": item_uniques, "difficulty": beta})
    item_meta = train_df[["item_key", "benchmark_id", "item_content"]].drop_duplicates("item_key")
    item_targets = item_targets.merge(item_meta, on="item_key", how="left")
    item_targets["centered_difficulty"] = item_targets["difficulty"] - item_targets.groupby("benchmark_id")[
        "difficulty"
    ].transform("mean")
    if item_limit > 0 and len(item_targets) > item_limit:
        item_targets = item_targets.sample(n=item_limit, random_state=326)

    remote_submission_dir = Path("/root/competition/douglas_ridge_text_features")
    remote_artifact_dir = remote_submission_dir / "artifacts"
    remote_artifact_dir.mkdir(parents=True, exist_ok=True)
    (remote_artifact_dir / "baseline_stats.json").write_text(json.dumps(baseline_stats), encoding="utf-8")
    dummy_artifact = {
        "encoder_id": "handcrafted-text-features-v1",
        "blend_weight": 0.30,
        "difficulty_cap": 2.5,
        "ridge_intercept": 0.0,
        "ridge_coef": [0.0] * 73,
        "scaler_mean": [0.0] * 73,
        "scaler_scale": [1.0] * 73,
        "feature_dim": 73,
    }
    (remote_artifact_dir / "bge_irt_ridge_artifact.json").write_text(
        json.dumps(dummy_artifact),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "douglas_text_features_validation_model_features",
        remote_submission_dir / "model.py",
    )
    feature_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(feature_module)

    scaler = StandardScaler()
    raw_features = np.asarray(
        [feature_module._text_features(text) for text in item_targets["item_content"].fillna("")],
        dtype=np.float32,
    )
    x_train = scaler.fit_transform(raw_features).astype(np.float32)
    y_train = item_targets["centered_difficulty"].to_numpy(dtype=np.float32)
    ridge = Ridge(alpha=300.0, random_state=326)
    ridge.fit(x_train, y_train)

    artifact = {
        "encoder_id": "handcrafted-text-features-v1",
        "blend_weight": 0.30,
        "difficulty_cap": 2.5,
        "ridge_intercept": float(ridge.intercept_),
        "ridge_coef": [float(x) for x in ridge.coef_],
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) if float(x) != 0.0 else 1.0 for x in scaler.scale_],
        "feature_dim": 73,
    }

    (remote_artifact_dir / "bge_irt_ridge_artifact.json").write_text(json.dumps(artifact), encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        "douglas_text_features_validation_model",
        remote_submission_dir / "model.py",
    )
    model_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(model_module)

    bases = []
    difficulties = []
    labels = []
    for row in valid_df.itertuples(index=False):
        input_row = {
            "benchmark": row.benchmark_id,
            "condition": row.condition,
            "subject_content": row.subject_content,
            "item_content": row.item_content,
        }
        bases.append(float(model_module._base_prediction(input_row)))
        difficulty = model_module._predict_centered_difficulty(input_row.get("item_content"))
        difficulties.append(None if difficulty is None else float(difficulty))
        labels.append(int(row.label))

    labels_array = np.asarray(labels, dtype=np.int64)
    blend_sweep = []
    for weight in parse_blend_weights(blend_weights):
        predictions = []
        for base, difficulty in zip(bases, difficulties):
            if difficulty is None:
                prediction = base
            else:
                clipped = max(-model_module.DIFFICULTY_CAP, min(model_module.DIFFICULTY_CAP, difficulty))
                prediction = model_module._sigmoid(model_module._logit(base) - weight * clipped)
            predictions.append(float(model_module._clamp(prediction)))
        probabilities = np.clip(np.asarray(predictions, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        negative_log_loss = -float(log_loss(labels_array, probabilities, labels=[0, 1]))
        auc = (
            float(roc_auc_score(labels_array, probabilities))
            if len(np.unique(labels_array)) == 2
            else float("nan")
        )
        blend_sweep.append(
            {
                "blend_weight": float(weight),
                "negative_log_loss": negative_log_loss,
                "auc_roc": auc,
            }
        )

    best = max(blend_sweep, key=lambda row: row["negative_log_loss"])
    default = min(blend_sweep, key=lambda row: abs(row["blend_weight"] - 0.45))

    summary = {
        "split": "whole-benchmark 80:20",
        "train_benchmarks": [name.removesuffix(".parquet") for name in train_files],
        "validation_benchmarks": [name.removesuffix(".parquet") for name in valid_files],
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "item_targets": int(len(item_targets)),
        "negative_log_loss": best["negative_log_loss"],
        "auc_roc": best["auc_roc"],
        "encoder": "handcrafted-text-features-v1",
        "ridge_alpha": 300.0,
        "blend_weight": best["blend_weight"],
        "default_blend_weight": 0.45,
        "default_negative_log_loss": default["negative_log_loss"],
        "default_auc_roc": default["auc_roc"],
        "blend_sweep": blend_sweep,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    cache_vol.commit()
    return {"summary": summary, "artifact": artifact, "baseline_stats": baseline_stats}


@app.local_entrypoint()
def main(
    train_rows: int = 1_500_000,
    valid_rows: int = 0,
    item_limit: int = 120_000,
    split_ratio: float = 0.8,
    blend_weights: str = "0,0.1,0.2,0.3,0.4,0.45,0.5,0.6,0.8,1.0",
) -> None:
    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = validate_remote.remote(
        train_rows=train_rows,
        valid_rows=valid_rows,
        item_limit=item_limit,
        split_ratio=split_ratio,
        blend_weights=blend_weights,
    )
    summary_path = LOCAL_ARTIFACT_DIR / "validation_80_20_text_features_summary.json"
    summary_path.write_text(json.dumps(payload["summary"], indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {summary_path}")
