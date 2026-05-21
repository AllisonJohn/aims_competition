from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("cs321m-final-gbdt")

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
    max_iter: int = 250,
    learning_rate: float = 0.04,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 1.0,
    subject_alpha: float = 250.0,
    condition_alpha: float = 2000.0,
    benchmark_alpha: float = 5000.0,
    bucket_alpha: float = 500.0,
) -> dict:
    import base64
    import math
    import pickle
    import re

    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from sklearn.ensemble import HistGradientBoostingClassifier

    repo_id = "aims-foundations/measurement-db"

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

    def bucket_for(benchmark: str, item_content: str) -> str:
        lower = str(item_content or "").lower()
        bucket = benchmark_buckets.get(str(benchmark or "").strip().lower())
        if bucket:
            return bucket
        scores = {
            name: sum(1 for keyword in keywords if keyword in lower)
            for name, keywords in bucket_keywords.items()
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general_reasoning"

    def item_adjustment(item_content: str) -> float:
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

    def text_features(item_content: str) -> list[float]:
        text = str(item_content or "")
        lower = text.lower()
        words = re.findall(r"[A-Za-z0-9_]+", lower)
        char_count = len(text)
        word_count = len(words)

        def bounded(value: float, scale: float) -> float:
            return min(float(value), scale) / scale

        def has_any(values: tuple[str, ...]) -> float:
            return float(any(value in lower for value in values))

        return [
            bounded(char_count, 5000),
            math.log1p(char_count) / math.log1p(10000),
            bounded(word_count, 1000),
            math.log1p(word_count) / math.log1p(2000),
            bounded(text.count("\n"), 80),
            bounded(text.count("?"), 20),
            bounded(sum(ch.isdigit() for ch in text), 200),
            bounded(text.count("(") + text.count(")"), 80),
            bounded(text.count("`"), 80),
            bounded(text.count("="), 80),
            float(any(symbol in text for symbol in ("∫", "∑", "∂", "√", "≤", "≥", "∈"))),
            has_any(("prove", "proof", "derive", "justify", "counterexample", "reason")),
            has_any(("calculate", "compute", "solve", "evaluate", "simplify")),
            has_any(("def ", "class ", "import ", "function", "algorithm", "pseudocode")),
            has_any(("bug", "debug", "error", "exception", "runtime", "traceback")),
            has_any(("api", "http", "request", "server", "endpoint")),
            has_any(("security", "attack", "injection", "malicious", "vulnerability")),
            has_any(("tool", "agent", "browser", "environment", "action")),
            has_any(("patient", "diagnosis", "treatment", "clinical", "medical")),
            has_any(("image", "figure", "diagram", "visual", "picture")),
            has_any(("chart", "graph", "plot", "table", "axis")),
            has_any(("a)", "b)", "c)", "d)", "multiple choice", "choose the best")),
            has_any(("preference", "ranking", "better response", "judge", "rubric")),
            has_any(("repository", "issue", "pull request", "patch", "test file")),
        ]

    def rate_table(df: pd.DataFrame, column: str) -> dict[str, list[float]]:
        grouped = df.groupby(column)["label"].agg(["mean", "count"])
        return {
            str(index).strip().lower(): [float(row["mean"]), float(row["count"])]
            for index, row in grouped.iterrows()
        }

    def pair_rate_table(df: pd.DataFrame, left: str, right: str) -> dict[str, list[float]]:
        grouped = df.groupby([left, right])["label"].agg(["mean", "count"])
        return {
            f"{str(index[0]).strip().lower()}||{str(index[1]).strip().lower()}": [
                float(row["mean"]),
                float(row["count"]),
            ]
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
            df = df.sample(n=per_file, random_state=527)
        df["subject_name"] = df["subject_id"].map(subject_name_lookup).fillna(df["subject_id"])
        df["condition"] = df["test_condition"].fillna("none").astype(str)
        df["item_content"] = df["item_id"].map(item_lookup).fillna("")
        df["bucket"] = [bucket_for(b, t) for b, t in zip(df["benchmark_id"], df["item_content"])]
        df["label"] = [
            binary_label(bench, response)
            for bench, response in zip(df["benchmark_id"], df["response"])
        ]
        parts.append(df[["subject_name", "benchmark_id", "condition", "bucket", "item_content", "label"]])
    train_df = pd.concat(parts, ignore_index=True)
    if train_rows > 0 and len(train_df) > train_rows:
        train_df = train_df.sample(n=train_rows, random_state=527)

    global_rate = float(train_df["label"].mean())
    subject_rates = rate_table(train_df, "subject_name")
    condition_rates = rate_table(train_df, "condition")
    benchmark_rates = rate_table(train_df, "benchmark_id")
    bucket_rates = rate_table(train_df, "bucket")
    subject_bucket_rates = pair_rate_table(train_df, "subject_name", "bucket")

    rows = []
    for row in train_df.itertuples(index=False):
        subject_rate = smooth(subject_rates.get(row.subject_name), subject_alpha, global_rate)
        condition_rate = smooth(condition_rates.get(row.condition), condition_alpha, global_rate)
        benchmark_rate = smooth(benchmark_rates.get(row.benchmark_id), benchmark_alpha, global_rate)
        bucket_rate = smooth(bucket_rates.get(row.bucket), bucket_alpha, global_rate)
        subject_bucket_rate = smooth(
            subject_bucket_rates.get(f"{row.subject_name}||{row.bucket}"),
            bucket_alpha,
            subject_rate,
        )
        base = [
            logit(subject_rate),
            logit(condition_rate),
            logit(benchmark_rate),
            logit(bucket_rate),
            logit(subject_bucket_rate) - logit(bucket_rate),
            item_adjustment(row.item_content),
        ]
        bucket_one_hot = [float(row.bucket == name) for name in bucket_names]
        rows.append(base + bucket_one_hot + text_features(row.item_content))

    x = np.asarray(rows, dtype=np.float32)
    y = train_df["label"].to_numpy(dtype=np.int64)
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        random_state=527,
    )
    model.fit(x, y)

    artifact = {
        "global_rate": global_rate,
        "subject_rates": subject_rates,
        "condition_rates": condition_rates,
        "benchmark_rates": benchmark_rates,
        "bucket_rates": bucket_rates,
        "subject_bucket_rates": subject_bucket_rates,
        "benchmark_buckets": benchmark_buckets,
        "bucket_keywords": {key: list(value) for key, value in bucket_keywords.items()},
        "bucket_names": bucket_names,
        "alphas": {
            "subject": float(subject_alpha),
            "condition": float(condition_alpha),
            "benchmark": float(benchmark_alpha),
            "bucket": float(bucket_alpha),
        },
        "model_pickle_b64": base64.b64encode(pickle.dumps(model)).decode("ascii"),
        "max_iter": int(max_iter),
        "learning_rate": float(learning_rate),
        "max_leaf_nodes": int(max_leaf_nodes),
        "l2_regularization": float(l2_regularization),
    }
    summary = {
        "train_rows": int(len(train_df)),
        "feature_dim": int(x.shape[1]),
        "global_rate": global_rate,
        "max_iter": int(max_iter),
        "learning_rate": float(learning_rate),
        "max_leaf_nodes": int(max_leaf_nodes),
        "l2_regularization": float(l2_regularization),
        "train_accuracy": float(model.score(x, y)),
    }
    return {"artifact": artifact, "summary": summary}


@app.local_entrypoint()
def main(
    train_rows: int = 0,
    max_iter: int = 250,
    learning_rate: float = 0.04,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 1.0,
    subject_alpha: float = 250.0,
    condition_alpha: float = 2000.0,
    benchmark_alpha: float = 5000.0,
    bucket_alpha: float = 500.0,
) -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = train_remote.remote(
        train_rows,
        max_iter,
        learning_rate,
        max_leaf_nodes,
        l2_regularization,
        subject_alpha,
        condition_alpha,
        benchmark_alpha,
        bucket_alpha,
    )
    (out_dir / "gbdt_artifact.json").write_text(
        json.dumps(payload["artifact"], separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "gbdt_summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
