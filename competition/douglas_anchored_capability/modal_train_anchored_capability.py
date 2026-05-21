from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("cs321m-final-anchored-capability")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "pyarrow", "scikit-learn", "huggingface_hub", "hf_xet")
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


@app.function(image=image, cpu=8.0, memory=32768, timeout=4 * 60 * 60)
def train_remote(
    train_rows: int = 0,
    subject_alpha: float = 250.0,
    condition_alpha: float = 2000.0,
    benchmark_alpha: float = 5000.0,
    bucket_alpha: float = 500.0,
    regularization_c: float = 1.0,
) -> dict:
    import math
    import re

    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    repo_id = "aims-foundations/measurement-db"

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

    def clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
        return max(lo, min(hi, float(value)))

    def logit(p: float) -> float:
        p = clamp(p)
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
        total = sum(values.values())
        if total <= 0.0:
            values["general_reasoning"] = 0.5
            total = 0.5
        return {key: value / total for key, value in values.items()}

    def primary_bucket(benchmark: str, item_content: str) -> str:
        loadings = bucket_loadings(benchmark, item_content)
        return max(loadings, key=loadings.get)

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

    def rate_table(df: pd.DataFrame, column: str) -> dict[str, list[float]]:
        grouped = df.groupby(column)["label"].agg(["mean", "count"])
        return {
            str(index).strip().lower(): [float(row["mean"]), float(row["count"])]
            for index, row in grouped.iterrows()
        }

    def smooth(rate_count: list[float] | None, alpha: float, prior: float) -> float:
        if not rate_count:
            return prior
        rate, count = float(rate_count[0]), float(rate_count[1])
        return (rate * count + prior * alpha) / (count + alpha)

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

    per_file = None if train_rows <= 0 else max(1, train_rows // len(RESPONSE_FILES))
    parts = []
    for filename in RESPONSE_FILES:
        path = hf_hub_download(repo_id, filename, repo_type="dataset")
        df = pd.read_parquet(
            path,
            columns=["subject_id", "item_id", "benchmark_id", "test_condition", "response"],
        )
        if per_file is not None and len(df) > per_file:
            df = df.sample(n=per_file, random_state=729)
        df["subject_name"] = df["subject_id"].map(subject_name_lookup).fillna(df["subject_id"])
        df["condition"] = df["test_condition"].fillna("none").astype(str)
        df["item_content"] = df["item_id"].map(item_lookup).fillna("")
        df["label"] = [
            binary_label(bench, response)
            for bench, response in zip(df["benchmark_id"], df["response"])
        ]
        df["capability_bucket"] = [
            primary_bucket(bench, text)
            for bench, text in zip(df["benchmark_id"], df["item_content"])
        ]
        parts.append(df[["subject_name", "benchmark_id", "condition", "item_content", "capability_bucket", "label"]])
    train_df = pd.concat(parts, ignore_index=True)
    if train_rows > 0 and len(train_df) > train_rows:
        train_df = train_df.sample(n=train_rows, random_state=729)

    global_rate = float(train_df["label"].mean())
    subject_rates = rate_table(train_df, "subject_name")
    condition_rates = rate_table(train_df, "condition")
    benchmark_rates = rate_table(train_df, "benchmark_id")
    bucket_rates = rate_table(train_df, "capability_bucket")

    subject_bucket = train_df.groupby(["subject_name", "capability_bucket"])["label"].agg(["mean", "count"])
    capability_offsets = {}
    for subject_name, rate_count in subject_rates.items():
        subject_rate = smooth(rate_count, subject_alpha, global_rate)
        offsets = {}
        for bucket in bucket_names:
            if (subject_name, bucket) in subject_bucket.index:
                row = subject_bucket.loc[(subject_name, bucket)]
                bucket_rate = float(row["mean"])
                bucket_count = float(row["count"])
            else:
                bucket_rate = subject_rate
                bucket_count = 0.0
            smoothed_bucket = (bucket_rate * bucket_count + subject_rate * bucket_alpha) / (bucket_count + bucket_alpha)
            offsets[bucket] = logit(smoothed_bucket) - logit(subject_rate)
        capability_offsets[subject_name] = offsets

    bucket_difficulty_offsets = {}
    for bucket in bucket_names:
        bucket_rate = smooth(bucket_rates.get(bucket), bucket_alpha, global_rate)
        bucket_difficulty_offsets[bucket] = logit(bucket_rate) - logit(global_rate)

    rows = []
    for row in train_df.itertuples(index=False):
        subject_name = str(row.subject_name).strip().lower()
        subject_rate = smooth(subject_rates.get(subject_name), subject_alpha, global_rate)
        condition_rate = smooth(condition_rates.get(str(row.condition).strip().lower()), condition_alpha, global_rate)
        benchmark_rate = smooth(benchmark_rates.get(str(row.benchmark_id).strip().lower()), benchmark_alpha, global_rate)
        loadings = bucket_loadings(row.benchmark_id, row.item_content)
        capability = sum(capability_offsets.get(subject_name, {}).get(bucket, 0.0) * value for bucket, value in loadings.items())
        bucket_difficulty = sum(bucket_difficulty_offsets.get(bucket, 0.0) * value for bucket, value in loadings.items())
        base = [
            logit(subject_rate),
            logit(condition_rate),
            logit(benchmark_rate),
            item_adjustment(row.item_content),
            capability,
            bucket_difficulty,
            capability * bucket_difficulty,
            abs(capability),
        ]
        rows.append(base + [float(loadings[name]) for name in bucket_names])

    x = np.asarray(rows, dtype=np.float32)
    y = train_df["label"].to_numpy(dtype=np.int64)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype(np.float32)
    model = LogisticRegression(C=regularization_c, solver="lbfgs", max_iter=1000, random_state=729)
    model.fit(x_scaled, y)

    artifact = {
        "global_rate": global_rate,
        "subject_rates": subject_rates,
        "condition_rates": condition_rates,
        "benchmark_rates": benchmark_rates,
        "bucket_rates": bucket_rates,
        "bucket_names": bucket_names,
        "benchmark_buckets": benchmark_buckets,
        "bucket_keywords": {key: list(value) for key, value in bucket_keywords.items()},
        "capability_offsets": capability_offsets,
        "bucket_difficulty_offsets": bucket_difficulty_offsets,
        "alphas": {
            "subject": float(subject_alpha),
            "condition": float(condition_alpha),
            "benchmark": float(benchmark_alpha),
            "bucket": float(bucket_alpha),
        },
        "calibrator": {
            "feature_names": [
                "subject_logit",
                "condition_logit",
                "benchmark_logit",
                "item_adjustment",
                "anchored_capability",
                "bucket_difficulty",
                "capability_x_bucket_difficulty",
                "abs_capability",
            ]
            + [f"loading_{name}" for name in bucket_names],
            "intercept": float(model.intercept_[0]),
            "coef": [float(value) for value in model.coef_[0]],
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) if float(value) != 0.0 else 1.0 for value in scaler.scale_],
            "c": float(regularization_c),
        },
    }
    summary = {
        "train_rows": int(len(train_df)),
        "feature_dim": int(x.shape[1]),
        "global_rate": global_rate,
        "subjects": len(subject_rates),
        "conditions": len(condition_rates),
        "benchmarks": len(benchmark_rates),
        "bucket_names": bucket_names,
        "regularization_c": float(regularization_c),
        "train_accuracy": float(model.score(x_scaled, y)),
    }
    return {"artifact": artifact, "summary": summary}


@app.local_entrypoint()
def main(
    train_rows: int = 0,
    subject_alpha: float = 250.0,
    condition_alpha: float = 2000.0,
    benchmark_alpha: float = 5000.0,
    bucket_alpha: float = 500.0,
    regularization_c: float = 1.0,
) -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = train_remote.remote(
        train_rows,
        subject_alpha,
        condition_alpha,
        benchmark_alpha,
        bucket_alpha,
        regularization_c,
    )
    (out_dir / "anchored_capability_artifact.json").write_text(
        json.dumps(payload["artifact"], separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "anchored_capability_summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
