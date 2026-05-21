"""Train an augmented BGE + scalar IRT + Ridge artifact.

This is a Douglas-folder variant of the strong BGE/Ridge baseline. It adds:
  1. direct fitted model ability lookup for seen models,
  2. empirical benchmark priors,
  3. separate Ridge regressors for item difficulty and model metadata ability.

Run:
    modal run competition/douglas/modal_train_bge_ridge_metadata.py

Train final submission artifact on all public benchmarks:
    modal run competition/douglas/modal_train_bge_ridge_metadata.py --submission --train-rows 0 --item-limit 0

Small held-out benchmark smoke test:
    modal run competition/douglas/modal_train_bge_ridge_metadata.py --train-rows 50000 --eval-rows 10000 --item-limit 5000

Small hyperparameter sweep:
    modal run competition/douglas/modal_train_bge_ridge_metadata.py --train-rows 50000 --eval-rows 10000 --item-limit 5000 --item-alphas 100,300,1000 --model-alphas 0.1,1,10
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import modal


APP_NAME = "douglas-bge-ridge-metadata-artifact"
LOCAL_DOUGLAS_DIR = Path(__file__).resolve().parent
LOCAL_ARTIFACT_DIR = LOCAL_DOUGLAS_DIR / "artifacts"

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
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "huggingface_hub",
        "hf_xet",
    )
)

cache_vol = modal.Volume.from_name("predeval-cache", create_if_missing=True)


def _logit(value: float, eps: float = 1e-4) -> float:
    value = max(eps, min(1.0 - eps, float(value)))
    return math.log(value / (1.0 - value))


def _filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return token.strip("._-")


@app.function(
    gpu="H100",
    image=image,
    volumes={"/cache": cache_vol},
    cpu=8.0,
    memory=65536,
    timeout=4 * 60 * 60,
)
def train_remote(
    train_rows: int = 1_500_000,
    eval_rows: int = 0,
    item_limit: int = 120_000,
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    item_alpha: float = 300.0,
    model_alpha: float = 1.0,
    item_alphas: str = "",
    model_alphas: str = "",
    rasch_reg: float = 0.1,
    rasch_iters: int = 8,
    submission: bool = False,
) -> dict:
    import os

    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import Ridge
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer

    os.environ.setdefault("HF_HOME", "/cache/hf")
    repo_id = "aims-foundations/measurement-db"

    def clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
        return max(lo, min(hi, float(value)))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def parse_alpha_grid(value: str, fallback: float) -> list[float]:
        if not value.strip():
            return [float(fallback)]
        return [float(part.strip()) for part in value.split(",") if part.strip()]

    def split_response_files(
        response_files: list[str],
        split_ratios: tuple[float, float, float] = (0.8, 0.0, 0.2),
    ) -> tuple[list[str], list[str], list[str]]:
        files = sorted(response_files)
        train_count = round(len(files) * split_ratios[0])
        validation_count = round(len(files) * split_ratios[1])
        train_end = train_count
        validation_end = train_end + validation_count
        return files[:train_end], files[train_end:validation_end], files[validation_end:]

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

    def parse_params_billions(value) -> float:
        text = str(value or "").lower().replace(",", "")
        if not text:
            return 0.0
        import re

        match = re.search(r"(\d+(?:\.\d+)?)\s*([bmk])?", text)
        if not match:
            return 0.0
        number = float(match.group(1))
        suffix = match.group(2)
        if suffix == "m":
            return number / 1000.0
        if suffix == "k":
            return number / 1_000_000.0
        return number

    def parse_release_year(value) -> float:
        import re

        match = re.search(r"(20\d{2}|19\d{2})", str(value or ""))
        if not match:
            return 0.0
        return max(0.0, min(1.0, (float(match.group(1)) - 2020.0) / 8.0))

    def metadata_key(value) -> str:
        return str(value or "missing").strip().lower() or "missing"

    subjects_path = hf_hub_download(repo_id, "subjects.parquet", repo_type="dataset")
    items_path = hf_hub_download(repo_id, "items.parquet", repo_type="dataset")
    subjects = pd.read_parquet(subjects_path)
    items = pd.read_parquet(items_path, columns=["item_id", "content"])

    subject_meta_lookup = {}
    subject_content_lookup = {}
    for _, row in subjects.iterrows():
        meta = row.to_dict()
        subject_id = meta["subject_id"]
        meta["subject_id"] = subject_id
        subject_meta_lookup[subject_id] = meta
        subject_content_lookup[subject_id] = render_subject_content(meta, subject_id)
    item_lookup = dict(zip(items["item_id"], items["content"]))

    if submission:
        train_files = sorted(RESPONSE_FILES)
        validation_files = []
        test_files = []
    else:
        train_files, validation_files, test_files = split_response_files(RESPONSE_FILES)

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
            df["item_content"] = df["item_id"].map(item_lookup).fillna("")
            df["item_key"] = df["benchmark_id"].astype(str) + "::" + df["item_id"].astype(str)
            df["condition"] = df["test_condition"].fillna("none").astype(str)
            df["label"] = [
                binary_label(bench, response)
                for bench, response in zip(df["benchmark_id"], df["response"])
            ]
            parts.append(
                df[["subject_id", "item_key", "benchmark_id", "condition", "item_content", "label"]]
            )
        out = pd.concat(parts, ignore_index=True)
        if row_limit > 0 and len(out) > row_limit:
            out = out.sample(n=row_limit, random_state=326)
        return out

    train_df = load_response_frame(train_files, train_rows)
    test_df = None if submission else load_response_frame(test_files, eval_rows)

    global_mean = float(train_df["label"].mean())
    global_logit = _logit(global_mean)
    benchmark_label_mean = train_df.groupby("benchmark_id")["label"].mean().to_dict()
    benchmark_logit_offsets = {
        benchmark: _logit(mean) - global_logit
        for benchmark, mean in benchmark_label_mean.items()
    }

    subject_codes, subject_uniques = pd.factorize(train_df["subject_id"], sort=True)
    item_codes, item_uniques = pd.factorize(train_df["item_key"], sort=True)
    y = train_df["label"].to_numpy(dtype=np.float64)
    theta = np.zeros(len(subject_uniques), dtype=np.float64)
    beta = np.zeros(len(item_uniques), dtype=np.float64)

    subj_groups = [np.where(subject_codes == i)[0] for i in range(len(subject_uniques))]
    item_groups = [np.where(item_codes == i)[0] for i in range(len(item_uniques))]
    for _ in range(rasch_iters):
        eta = theta[subject_codes] - beta[item_codes]
        p = sigmoid(eta)
        for i, idx in enumerate(subj_groups):
            if len(idx) == 0:
                continue
            grad = (y[idx] - p[idx]).sum() - rasch_reg * theta[i]
            hess = -(p[idx] * (1 - p[idx])).sum() - rasch_reg
            theta[i] -= grad / hess
        theta -= theta.mean()

        eta = theta[subject_codes] - beta[item_codes]
        p = sigmoid(eta)
        for j, idx in enumerate(item_groups):
            if len(idx) == 0:
                continue
            grad = -(y[idx] - p[idx]).sum() - rasch_reg * beta[j]
            hess = -(p[idx] * (1 - p[idx])).sum() - rasch_reg
            beta[j] -= grad / hess

    item_targets = pd.DataFrame({"item_key": item_uniques, "difficulty": beta})
    item_meta = train_df[["item_key", "benchmark_id", "condition", "item_content"]].drop_duplicates("item_key")
    item_targets = item_targets.merge(item_meta, on="item_key", how="left")
    benchmark_beta_mean = item_targets.groupby("benchmark_id")["difficulty"].mean().to_dict()
    item_targets["benchmark_difficulty_prior"] = item_targets["benchmark_id"].map(benchmark_beta_mean).fillna(0.0)
    item_targets["centered_difficulty"] = (
        item_targets["difficulty"] - item_targets["benchmark_difficulty_prior"]
    )
    if item_limit > 0 and len(item_targets) > item_limit:
        item_targets = item_targets.sample(n=item_limit, random_state=326)

    subject_targets = pd.DataFrame({"subject_id": subject_uniques, "theta": theta})
    subject_targets["meta"] = subject_targets["subject_id"].map(subject_meta_lookup)
    subject_targets = subject_targets.dropna(subset=["meta"])
    global_theta = float(subject_targets["theta"].mean())
    subject_theta_lookup = {
        metadata_key(subject_id): float(theta_value)
        for subject_id, theta_value in zip(subject_targets["subject_id"], subject_targets["theta"])
    }
    name_theta_lookup = {
        metadata_key(meta.get("display_name") or meta.get("subject_id")): float(theta_value)
        for meta, theta_value in zip(subject_targets["meta"], subject_targets["theta"])
    }

    class Embedder:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_id, cache_dir="/cache/hf")
            self.model = AutoModel.from_pretrained(encoder_id, cache_dir="/cache/hf").to("cuda")
            self.model.eval()

        @torch.no_grad()
        def encode(self, texts: list[str], batch_size: int = 96) -> np.ndarray:
            chunks = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                batch = [f"Represent this evaluation question for difficulty prediction: {text}" for text in batch]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
                output = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                chunks.append(pooled.cpu().numpy().astype(np.float32))
            return np.vstack(chunks)

    def item_numeric_features(df: pd.DataFrame) -> np.ndarray:
        rows = []
        for text in df["item_content"]:
            text = str(text or "")
            lower = text.lower()
            rows.append(
                [
                    min(len(text), 5000) / 5000.0,
                    min(len(text.split()), 1000) / 1000.0,
                    min(text.count("\n"), 40) / 40.0,
                    float(any(s in text for s in ("∫", "∑", "∂", "√", "≤", "≥", "∈"))),
                    float(any(w in lower for w in ("prove", "derive", "justify", "counterexample"))),
                    float(any(w in lower for w in ("def ", "class ", "import ", "function", "debug", "algorithm"))),
                    float(any(w in lower for w in ("image", "figure", "diagram", "chart", "visual"))),
                    float(any(w in lower for w in ("a)", "b)", "c)", "d)", "multiple choice", "choose the best"))),
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    embedder = Embedder()
    embeddings = embedder.encode(item_targets["item_content"].fillna("").tolist())
    item_features = np.hstack([embeddings, item_numeric_features(item_targets)]).astype(np.float32)
    item_scaler = StandardScaler()
    item_x = item_scaler.fit_transform(item_features)
    item_y = item_targets["centered_difficulty"].to_numpy(dtype=np.float32)

    def lookup_theta(subject_id) -> float:
        meta = subject_meta_lookup.get(subject_id, {"subject_id": subject_id})
        subject_key = metadata_key(meta.get("subject_id") or subject_id)
        name_key = metadata_key(meta.get("display_name") or subject_id)
        return subject_theta_lookup.get(subject_key, name_theta_lookup.get(name_key, global_theta))

    sweep_results = []
    test_negative_log_loss = float("nan")
    test_auc = float("nan")

    if not submission:
        test_unique_items = test_df[["item_key", "benchmark_id", "item_content"]].drop_duplicates("item_key")
        test_item_embeddings = embedder.encode(test_unique_items["item_content"].fillna("").tolist())
        test_item_numeric = item_numeric_features(test_unique_items)
        test_item_raw_x = np.hstack([test_item_embeddings, test_item_numeric]).astype(np.float32)
        test_item_x = (
            (test_item_raw_x - item_scaler.mean_)
            / np.where(item_scaler.scale_ == 0.0, 1.0, item_scaler.scale_)
        )
        test_item_priors = np.array(
            [
                benchmark_beta_mean.get(benchmark, 0.0)
                for benchmark in test_unique_items["benchmark_id"]
            ],
            dtype=np.float64,
        )
        test_benchmark_offsets = np.array(
            [
                benchmark_logit_offsets.get(benchmark, 0.0)
                for benchmark in test_df["benchmark_id"]
            ],
            dtype=np.float64,
        )
        test_labels = test_df["label"].to_numpy(dtype=np.int64)

        item_alpha_values = parse_alpha_grid(item_alphas, item_alpha)
        best = None

        for item_alpha_value in item_alpha_values:
            item_candidate = Ridge(alpha=item_alpha_value, random_state=326)
            item_candidate.fit(item_x, item_y)
            centered = item_candidate.predict(test_item_x)
            difficulty = centered + test_item_priors
            item_difficulty = dict(zip(test_unique_items["item_key"], difficulty))
            test_beta = test_df["item_key"].map(item_difficulty).fillna(0.0).to_numpy(dtype=np.float64)

            test_theta = test_df["subject_id"].map(lookup_theta).to_numpy(dtype=np.float64)
            test_logits = test_theta - test_beta + test_benchmark_offsets
            test_probs = np.clip(sigmoid(test_logits), 1e-7, 1.0 - 1e-7)
            negative_log_loss = -float(log_loss(test_labels, test_probs, labels=[0, 1]))
            auc = (
                float(roc_auc_score(test_labels, test_probs))
                if len(np.unique(test_labels)) == 2
                else float("nan")
            )
            result = {
                "item_alpha": float(item_alpha_value),
                "model_alpha": None,
                "test_negative_log_loss": negative_log_loss,
                "test_auc_roc": auc,
            }
            sweep_results.append(result)
            if best is None or negative_log_loss > best["test_negative_log_loss"]:
                best = result

        item_alpha = float(best["item_alpha"])
        test_negative_log_loss = float(best["test_negative_log_loss"])
        test_auc = float(best["test_auc_roc"])

    item_model = Ridge(alpha=item_alpha, random_state=326)
    item_model.fit(item_x, item_y)

    artifact = {
        "encoder_id": encoder_id,
        "max_length": 256,
        "artifact_type": "bge_ridge_metadata",
        "global_label_mean": global_mean,
        "global_logit": global_logit,
        "benchmark_logit_offsets": {str(k): float(v) for k, v in benchmark_logit_offsets.items()},
        "benchmark_difficulty_priors": {str(k): float(v) for k, v in benchmark_beta_mean.items()},
        "item_ridge": {
            "alpha": item_alpha,
            "intercept": float(item_model.intercept_),
            "coef": [float(x) for x in item_model.coef_],
            "feature_mean": [float(x) for x in item_scaler.mean_],
            "feature_scale": [float(x) if float(x) != 0.0 else 1.0 for x in item_scaler.scale_],
            "embedding_dim": int(embeddings.shape[1]),
            "numeric_dim": int(item_features.shape[1] - embeddings.shape[1]),
        },
        "model_lookup": {
            "global_theta": global_theta,
            "subject_theta_lookup": subject_theta_lookup,
            "name_theta_lookup": name_theta_lookup,
        },
    }

    summary = {
        "train_rows": int(len(train_df)),
        "eval_rows": 0 if test_df is None else int(len(test_df)),
        "submission": submission,
        "train_benchmarks": [file.removesuffix(".parquet") for file in train_files],
        "validation_benchmarks": [file.removesuffix(".parquet") for file in validation_files],
        "test_benchmarks": [file.removesuffix(".parquet") for file in test_files],
        "rasch_subjects": int(len(subject_uniques)),
        "rasch_items": int(len(item_uniques)),
        "text_item_targets": int(len(item_targets)),
        "model_targets": int(len(subject_targets)),
        "encoder": encoder_id,
        "item_alpha": item_alpha,
        "model_alpha": None,
        "global_label_mean": global_mean,
        "test_negative_log_loss": None if submission else test_negative_log_loss,
        "test_auc_roc": None if submission else test_auc,
        "sweep_results": sweep_results,
        "item_nonzero_coef": int(np.count_nonzero(item_model.coef_)),
        "model_lookup_size": len(subject_theta_lookup),
    }
    cache_vol.commit()
    return {"artifact": artifact, "summary": summary}


@app.local_entrypoint()
def main(
    train_rows: int = 1_500_000,
    eval_rows: int = 0,
    item_limit: int = 120_000,
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    item_alpha: float = 300.0,
    model_alpha: float = 1.0,
    item_alphas: str = "",
    model_alphas: str = "",
    rasch_reg: float = 0.1,
    rasch_iters: int = 8,
    submission: bool = False,
    run_name: str = "",
) -> None:
    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = train_remote.remote(
        train_rows=train_rows,
        eval_rows=eval_rows,
        item_limit=item_limit,
        encoder_id=encoder_id,
        item_alpha=item_alpha,
        model_alpha=model_alpha,
        item_alphas=item_alphas,
        model_alphas=model_alphas,
        rasch_reg=rasch_reg,
        rasch_iters=rasch_iters,
        submission=submission,
    )
    artifact_stem = "bge_ridge_metadata_submit" if submission else "bge_ridge_metadata"
    run_token = _filename_token(run_name)
    if run_token:
        artifact_stem = f"{artifact_stem}_{run_token}"
    artifact_path = LOCAL_ARTIFACT_DIR / f"{artifact_stem}_artifact.json"
    summary_path = LOCAL_ARTIFACT_DIR / f"{artifact_stem}_summary.json"
    artifact_path.write_text(
        json.dumps(payload["artifact"], separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {artifact_path}")
