from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("cs321m-final-bge-capability-ridge-artifact")

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
    blend_weight: float = 0.30,
    capability_weight: float = 0.25,
    bucket_alpha: float = 200.0,
) -> dict:
    import math

    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer

    repo_id = "aims-foundations/measurement-db"

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

    bucket_names = [
        "math",
        "code",
        "visual",
        "agent_tool",
        "preference",
        "science_medical",
        "security",
        "general_reasoning",
    ]
    benchmark_buckets = {
        "matharena": "math",
        "mathvista_mini": "visual",
        "livecodebench": "code",
        "swebench": "code",
        "bfcl": "agent_tool",
        "agentdojo": "agent_tool",
        "androidworld": "agent_tool",
        "ai2d_test": "visual",
        "mmbench_v11": "visual",
        "rewardbench": "preference",
        "ultrafeedback": "preference",
        "mtbench": "preference",
        "afrimedqa": "science_medical",
        "cybench": "security",
        "hle": "general_reasoning",
        "mmlupro": "general_reasoning",
    }
    bucket_keywords = {
        "math": ("prove", "derive", "calculate", "equation", "theorem", "probability", "geometry", "∫", "∑"),
        "code": ("def ", "class ", "import ", "function", "debug", "algorithm", "runtime", "traceback"),
        "visual": ("image", "figure", "diagram", "chart", "graph", "visual", "picture", "table"),
        "agent_tool": ("tool", "agent", "browser", "environment", "action", "api", "request", "endpoint"),
        "preference": ("preference", "ranking", "better response", "judge", "rubric", "reward", "feedback"),
        "science_medical": ("patient", "diagnosis", "treatment", "clinical", "biology", "chemistry", "physics"),
        "security": ("security", "attack", "injection", "malicious", "vulnerability", "exploit"),
        "general_reasoning": ("reason", "justify", "because", "therefore", "constraint", "must", "should"),
    }

    def bucket_loadings(benchmark: str, item_content: str) -> dict[str, float]:
        lower = str(item_content or "").lower()
        values = {name: 0.0 for name in bucket_names}
        benchmark_bucket = benchmark_buckets.get(str(benchmark or "").strip().lower())
        if benchmark_bucket:
            values[benchmark_bucket] = 1.0
        for bucket, keywords in bucket_keywords.items():
            hits = sum(1 for keyword in keywords if keyword in lower)
            if hits:
                values[bucket] = max(values[bucket], min(1.0, 0.25 + 0.15 * hits))
        if not any(values.values()):
            values["general_reasoning"] = 0.5
        return values

    def primary_bucket(benchmark: str, item_content: str) -> str:
        values = bucket_loadings(benchmark, item_content)
        return max(values, key=values.get)

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

    per_file = max(1, train_rows // len(RESPONSE_FILES))
    parts = []
    for filename in RESPONSE_FILES:
        path = hf_hub_download(repo_id, filename, repo_type="dataset")
        df = pd.read_parquet(
            path,
            columns=["subject_id", "item_id", "benchmark_id", "test_condition", "response"],
        )
        if len(df) > per_file:
            df = df.sample(n=per_file, random_state=326)
        df["subject_name"] = df["subject_id"].map(subject_name_lookup).fillna(df["subject_id"])
        df["item_content"] = df["item_id"].map(item_lookup).fillna("")
        df["item_key"] = df["benchmark_id"].astype(str) + "::" + df["item_id"].astype(str)
        df["label"] = [
            binary_label(bench, response)
            for bench, response in zip(df["benchmark_id"], df["response"])
        ]
        df["capability_bucket"] = [
            primary_bucket(bench, text)
            for bench, text in zip(df["benchmark_id"], df["item_content"])
        ]
        parts.append(df[["subject_name", "item_key", "benchmark_id", "item_content", "label", "capability_bucket"]])
    train_df = pd.concat(parts, ignore_index=True)
    if len(train_df) > train_rows:
        train_df = train_df.sample(n=train_rows, random_state=326)

    subject_global = train_df.groupby("subject_name")["label"].agg(["mean", "count"])
    subject_bucket = train_df.groupby(["subject_name", "capability_bucket"])["label"].agg(["mean", "count"])
    global_rate = float(train_df["label"].mean())
    capability_offsets = {}
    for subject_name, global_row in subject_global.iterrows():
        global_mean = float(global_row["mean"])
        global_count = float(global_row["count"])
        smoothed_global = (global_mean * global_count + global_rate * bucket_alpha) / (global_count + bucket_alpha)
        offsets = {}
        for bucket in bucket_names:
            if (subject_name, bucket) in subject_bucket.index:
                bucket_row = subject_bucket.loc[(subject_name, bucket)]
                bucket_mean = float(bucket_row["mean"])
                bucket_count = float(bucket_row["count"])
            else:
                bucket_mean = smoothed_global
                bucket_count = 0.0
            smoothed_bucket = (
                bucket_mean * bucket_count + smoothed_global * bucket_alpha
            ) / (bucket_count + bucket_alpha)
            offsets[bucket] = float(math.log(clamp(smoothed_bucket) / (1.0 - clamp(smoothed_bucket))) - math.log(clamp(smoothed_global) / (1.0 - clamp(smoothed_global))))
        capability_offsets[str(subject_name).strip().lower()] = offsets

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
    if len(item_targets) > item_limit:
        item_targets = item_targets.sample(n=item_limit, random_state=326)

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
                batch = [f"Represent this evaluation question for difficulty prediction: {t}" for t in batch]
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

    def numeric_features(df: pd.DataFrame) -> np.ndarray:
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
    scaler = StandardScaler()
    numeric = scaler.fit_transform(numeric_features(item_targets))
    x_train = np.hstack([embeddings, numeric]).astype(np.float32)
    y_train = item_targets["centered_difficulty"].to_numpy(dtype=np.float32)
    model = Ridge(alpha=300.0, random_state=326)
    model.fit(x_train, y_train)

    return {
        "artifact": {
            "encoder_id": encoder_id,
            "blend_weight": float(blend_weight),
            "capability_weight": float(capability_weight),
            "bucket_alpha": float(bucket_alpha),
            "bucket_names": bucket_names,
            "capability_offsets": capability_offsets,
            "ridge_intercept": float(model.intercept_),
            "ridge_coef": [float(x) for x in model.coef_],
            "scaler_mean": [float(x) for x in scaler.mean_],
            "scaler_scale": [float(x) if float(x) != 0.0 else 1.0 for x in scaler.scale_],
            "embedding_dim": int(embeddings.shape[1]),
            "numeric_dim": int(numeric.shape[1]),
            "max_length": 256,
        },
        "summary": {
            "train_rows": int(len(train_df)),
            "rasch_subjects": int(len(subject_uniques)),
            "rasch_items": int(len(item_uniques)),
            "text_item_targets": int(len(item_targets)),
            "encoder": encoder_id,
            "ridge_alpha": 300.0,
            "blend_weight": float(blend_weight),
            "capability_weight": float(capability_weight),
            "bucket_alpha": float(bucket_alpha),
            "capability_subjects": int(len(capability_offsets)),
        },
    }


@app.local_entrypoint()
def main(
    train_rows: int = 1_500_000,
    item_limit: int = 120_000,
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    blend_weight: float = 0.30,
    capability_weight: float = 0.25,
    bucket_alpha: float = 200.0,
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
        blend_weight,
        capability_weight,
        bucket_alpha,
    )
    (out_dir / "bge_irt_ridge_artifact.json").write_text(
        json.dumps(payload["artifact"], separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "bge_irt_ridge_summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {out_dir / 'bge_irt_ridge_artifact.json'}")
