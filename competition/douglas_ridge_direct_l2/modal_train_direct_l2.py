from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("cs321m-final-bge-direct-l2-artifact")

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
    )
)

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


@app.function(gpu="H100", image=image, timeout=60 * 60 * 4)
def train_remote(
    baseline_stats: dict,
    train_rows: int = 1_500_000,
    item_limit: int = 120_000,
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    regularization_c: float = 1.0,
) -> dict:
    import math
    import re

    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer

    repo_id = "aims-foundations/measurement-db"

    def clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
        return max(lo, min(hi, float(value)))

    def logit(p: float) -> float:
        p = clamp(float(p))
        return math.log(p / (1.0 - p))

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

    def smooth(rate_count: list[float] | None, alpha: float, global_rate: float) -> float:
        if not rate_count:
            return global_rate
        rate, count = float(rate_count[0]), float(rate_count[1])
        return (rate * count + global_rate * alpha) / (count + alpha)

    def item_adjustment(item_content: object) -> float:
        text = str(item_content or "")
        lower = text.lower()
        length = len(text)
        adjustment = 0.0
        if length > 2500:
            adjustment -= 0.07
        elif length > 1000:
            adjustment -= 0.04
        elif 0 < length < 140:
            adjustment += 0.02
        if text.count("\n") > 12 or lower.count("part ") > 1:
            adjustment -= 0.03
        if any(symbol in text for symbol in ("∫", "∑", "∂", "√", "≤", "≥", "∈")):
            adjustment -= 0.04
        if re.search(r"\b(prove|derive|justify|rigorously|counterexample)\b", lower):
            adjustment -= 0.04
        if re.search(r"\b(def|class|import|function|bug|debug|runtime|algorithm)\b", lower):
            adjustment -= 0.03
        if re.search(r"\b(what is|who is|when did|where is)\b", lower) and length < 240:
            adjustment += 0.03
        if re.search(r"\b(a\)|b\)|c\)|d\)|multiple choice|choose the best)\b", lower):
            adjustment += 0.01
        return adjustment

    def numeric_features_for_texts(texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
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
    subject_name_lookup = {sid: parse_subject_name(content) for sid, content in subject_lookup.items()}
    item_lookup = dict(zip(items["item_id"], items["content"]))

    row_limit = None if train_rows <= 0 else train_rows
    per_file = None if row_limit is None else max(1, row_limit // len(RESPONSE_FILES))
    parts = []
    for filename in RESPONSE_FILES:
        path = hf_hub_download(repo_id, filename, repo_type="dataset")
        df = pd.read_parquet(
            path,
            columns=["subject_id", "item_id", "benchmark_id", "test_condition", "response"],
        )
        if per_file is not None and len(df) > per_file:
            df = df.sample(n=per_file, random_state=326)
        df["subject_name"] = df["subject_id"].map(subject_name_lookup).fillna(df["subject_id"])
        df["condition"] = df["test_condition"].fillna("none").astype(str)
        df["item_content"] = df["item_id"].map(item_lookup).fillna("")
        df["label"] = [
            binary_label(bench, response)
            for bench, response in zip(df["benchmark_id"], df["response"])
        ]
        parts.append(df[["subject_name", "condition", "item_content", "label"]])
    train_df = pd.concat(parts, ignore_index=True)
    if row_limit is not None and len(train_df) > row_limit:
        train_df = train_df.sample(n=row_limit, random_state=326)

    global_rate = float(train_df["label"].mean())
    subject_rates = rate_table(train_df, "subject_name")
    condition_rates = rate_table(train_df, "condition")
    subject_alpha = 250.0
    condition_alpha = 2000.0

    item_table = train_df[["item_content"]].drop_duplicates().reset_index(drop=True)
    if item_limit > 0 and len(item_table) > item_limit:
        item_table = item_table.sample(n=item_limit, random_state=326).reset_index(drop=True)
    item_texts = item_table["item_content"].fillna("").astype(str).tolist()

    class Embedder:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_id)
            self.model = AutoModel.from_pretrained(encoder_id).to("cuda")
            self.model.eval()

        @torch.no_grad()
        def encode(self, texts: list[str], batch_size: int = 96) -> np.ndarray:
            chunks = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                batch = [f"Represent this evaluation question for correctness prediction: {t}" for t in batch]
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
                print(f"encoded {min(start + len(batch), len(texts))}/{len(texts)} item texts", flush=True)
            return np.vstack(chunks)

    embedder = Embedder()
    item_embeddings = embedder.encode(item_texts)
    item_numeric = numeric_features_for_texts(item_texts)
    item_index = {text: idx for idx, text in enumerate(item_texts)}
    zero_embedding = np.zeros(item_embeddings.shape[1], dtype=np.float32)
    zero_numeric = np.zeros(item_numeric.shape[1], dtype=np.float32)

    feature_dim = 3 + item_embeddings.shape[1] + item_numeric.shape[1]
    x = np.empty((len(train_df), feature_dim), dtype=np.float32)
    for row_idx, row in enumerate(train_df.itertuples(index=False)):
        subject_rate = smooth(subject_rates.get(str(row.subject_name).strip().lower()), subject_alpha, global_rate)
        condition_rate = smooth(condition_rates.get(str(row.condition).strip().lower()), condition_alpha, global_rate)
        subject_logit = logit(subject_rate)
        condition_logit = logit(condition_rate)
        adjustment = item_adjustment(row.item_content)
        idx = item_index.get(str(row.item_content or ""))
        if idx is None:
            embedding = zero_embedding
            numeric = zero_numeric
        else:
            embedding = item_embeddings[idx]
            numeric = item_numeric[idx]
        x[row_idx, 0] = subject_logit
        x[row_idx, 1] = condition_logit
        x[row_idx, 2] = adjustment
        x[row_idx, 3 : 3 + item_embeddings.shape[1]] = embedding
        x[row_idx, 3 + item_embeddings.shape[1] :] = numeric
    y = train_df["label"].to_numpy(dtype=np.int64)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype(np.float32)
    model = LogisticRegression(
        C=float(regularization_c),
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=326,
    )
    model.fit(x_scaled, y)
    train_prob = model.predict_proba(x_scaled)[:, 1]
    eps = 1e-12
    train_bce = float(-np.mean(y * np.log(train_prob + eps) + (1 - y) * np.log(1 - train_prob + eps)))
    train_accuracy = float(((train_prob >= 0.5).astype(np.int64) == y).mean())

    feature_names = (
        ["subject_logit", "condition_logit", "item_adjustment"]
        + [f"embedding_{i}" for i in range(item_embeddings.shape[1])]
        + [
            "text_len",
            "word_count",
            "newline_count",
            "math_symbol",
            "proof_word",
            "code_word",
            "visual_word",
            "multiple_choice",
        ]
    )
    return {
        "artifact": {
            "encoder_id": encoder_id,
            "feature_names": feature_names,
            "intercept": float(model.intercept_[0]),
            "coef": [float(v) for v in model.coef_[0]],
            "scaler_mean": [float(v) for v in scaler.mean_],
            "scaler_scale": [float(v) if float(v) != 0.0 else 1.0 for v in scaler.scale_],
            "embedding_dim": int(item_embeddings.shape[1]),
            "numeric_dim": int(item_numeric.shape[1]),
            "max_length": 256,
            "regularization_c": float(regularization_c),
        },
        "summary": {
            "train_rows": int(len(train_df)),
            "unique_encoded_items": int(len(item_texts)),
            "encoder": encoder_id,
            "regularization_c": float(regularization_c),
            "feature_dim": int(x.shape[1]),
            "train_bce": train_bce,
            "train_accuracy": train_accuracy,
            "global_rate": global_rate,
        },
    }


@app.local_entrypoint()
def main(
    train_rows: int = 1_500_000,
    item_limit: int = 120_000,
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    regularization_c: float = 1.0,
) -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "artifacts"
    stats_path = out_dir / "baseline_stats.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    payload = train_remote.remote(
        stats,
        train_rows,
        item_limit,
        encoder_id,
        regularization_c,
    )
    (out_dir / "bge_direct_l2_artifact.json").write_text(
        json.dumps(payload["artifact"], separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "bge_direct_l2_summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {out_dir / 'bge_direct_l2_artifact.json'}")
