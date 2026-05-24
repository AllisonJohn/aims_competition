from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("cs321m-final-bge-kfactor-k3-pairwise-adapter-only")

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
def train_adapter_remote(
    baseline_stats: dict,
    base_artifact: dict,
    train_rows: int = 1_500_000,
    pairwise_residual_weight: float = 0.75,
    pairwise_c: float = 0.5,
) -> dict:
    import math

    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer

    repo_id = "aims-foundations/measurement-db"

    global_rate = float(baseline_stats["global_rate"])
    subject_rates = baseline_stats["subject_rates"]
    condition_rates = baseline_stats["condition_rates"]
    benchmark_rates = baseline_stats["benchmark_rates"]
    subject_alpha = 250.0
    condition_alpha = 2000.0
    benchmark_alpha = 5000.0

    encoder_id = str(base_artifact["encoder_id"])
    latent_dim = int(base_artifact["latent_dim"])
    blend_weight = float(base_artifact["blend_weight"])
    logit_cap = float(base_artifact.get("logit_cap", 4.0))
    calibrator = base_artifact.get("calibrator")
    subject_factors = {
        str(key).strip().lower(): np.asarray(value, dtype=np.float32)
        for key, value in base_artifact["subject_factors"].items()
    }
    global_subject_factor = np.asarray(
        base_artifact.get("global_subject_factor", [0.0] * latent_dim),
        dtype=np.float32,
    )
    ridge_heads = base_artifact["ridge_heads"]
    scaler_mean = np.asarray(base_artifact["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(
        [float(x) if float(x) != 0.0 else 1.0 for x in base_artifact["scaler_scale"]],
        dtype=np.float32,
    )
    embedding_dim = int(base_artifact["embedding_dim"])
    max_length = int(base_artifact.get("max_length", 256))

    def clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
        return max(lo, min(hi, float(value)))

    def logit(p: float) -> float:
        p = clamp(float(p))
        return math.log(p / (1.0 - p))

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def parse_subject_name(subject_content: object) -> str:
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

    def smooth(rate_count: list[float] | None, alpha: float) -> float:
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
        if any(word in lower for word in ("prove", "derive", "justify", "rigorously", "counterexample")):
            adjustment -= 0.04
        if any(word in lower for word in ("def ", "class ", "import ", "function", "bug", "debug", "runtime", "algorithm")):
            adjustment -= 0.03
        if any(word in lower for word in ("what is", "who is", "when did", "where is")) and length < 240:
            adjustment += 0.03
        if any(word in lower for word in ("a)", "b)", "c)", "d)", "multiple choice", "choose the best")):
            adjustment += 0.01
        return adjustment

    def numeric_features(text: object) -> list[float]:
        text = str(text or "")
        lower = text.lower()
        return [
            min(len(text), 5000) / 5000.0,
            min(len(text.split()), 1000) / 1000.0,
            min(text.count("\n"), 40) / 40.0,
            float(any(s in text for s in ("∫", "∑", "∂", "√", "≤", "≥", "∈"))),
            float(any(w in lower for w in ("prove", "derive", "justify", "counterexample"))),
            float(any(w in lower for w in ("def ", "class ", "import ", "function", "debug", "algorithm"))),
            float(any(w in lower for w in ("image", "figure", "diagram", "chart", "visual"))),
            float(any(w in lower for w in ("a)", "b)", "c)", "d)", "multiple choice", "choose the best"))),
        ]

    pairwise_benchmarks = {
        "mtbench",
        "rewardbench",
        "ultrafeedback",
        "lmarena",
        "lm_arena",
        "lm-arena",
        "chatbot_arena",
        "chatbot-arena",
    }
    pairwise_keywords = (
        "response a",
        "response b",
        "assistant a",
        "assistant b",
        "answer a",
        "answer b",
        "chosen",
        "rejected",
        "preference",
        "preferred",
        "which response",
        "better response",
        "judge",
        "rubric",
        "helpfulness",
        "honesty",
        "truthfulness",
        "instruction_following",
        "ranking",
        "winner",
    )

    def pairwise_cue_score(benchmark: object, condition: object, item_content: object) -> tuple[bool, float, float]:
        benchmark_key = str(benchmark or "").strip().lower()
        text = f"{condition or ''}\n{item_content or ''}".lower()
        benchmark_flag = float(benchmark_key in pairwise_benchmarks)
        cue_count = sum(1 for keyword in pairwise_keywords if keyword in text)
        cue_score = min(1.0, cue_count / 3.0)
        return bool(benchmark_flag or cue_score >= 0.67), benchmark_flag, cue_score

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
            df = df.sample(n=per_file, random_state=326)
        df["subject_name"] = df["subject_id"].map(subject_name_lookup).fillna(df["subject_id"])
        df["condition"] = df["test_condition"].fillna("none").astype(str)
        df["item_content"] = df["item_id"].map(item_lookup).fillna("")
        df["label"] = [
            binary_label(bench, response)
            for bench, response in zip(df["benchmark_id"], df["response"])
        ]
        keep = [
            pairwise_cue_score(bench, condition, content)[0]
            for bench, condition, content in zip(df["benchmark_id"], df["condition"], df["item_content"])
        ]
        df = df.loc[keep, ["subject_name", "benchmark_id", "condition", "item_content", "label"]]
        parts.append(df)
    train_df = pd.concat(parts, ignore_index=True)
    if train_rows > 0 and len(train_df) > train_rows:
        train_df = train_df.sample(n=train_rows, random_state=326)
    if train_df.empty:
        raise ValueError("No pairwise-like rows found for adapter training")

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
                batch = [f"Represent this evaluation question for capability-factor prediction: {t}" for t in batch]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
                output = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                chunks.append(pooled.cpu().numpy().astype(np.float32))
                if (start // batch_size + 1) % 25 == 0:
                    print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)} pairwise item texts", flush=True)
            return np.vstack(chunks)

    unique_items = train_df["item_content"].drop_duplicates().reset_index(drop=True)
    print(
        f"Training pairwise adapter only: rows={len(train_df)} unique_item_texts={len(unique_items)}",
        flush=True,
    )
    embeddings = Embedder().encode(unique_items.fillna("").tolist())
    item_latents = {}
    for text, embedding in zip(unique_items.tolist(), embeddings):
        numeric = np.asarray(numeric_features(text), dtype=np.float32)
        numeric = (numeric - scaler_mean) / scaler_scale
        features = np.concatenate([embedding.astype(np.float32), numeric]).astype(np.float32)
        if len(features) != embedding_dim + len(scaler_mean):
            raise ValueError(f"Bad feature length {len(features)}")
        values = {}
        for head in ridge_heads:
            prediction = float(head["ridge_intercept"])
            prediction += float(np.dot(np.asarray(head["ridge_coef"], dtype=np.float32), features))
            if head.get("target_standardized"):
                prediction = float(head.get("target_mean", 0.0)) + float(head.get("target_std", 1.0)) * prediction
            values[str(head["name"])] = prediction
        item_factors = np.asarray([values.get(f"factor_{k}", 0.0) for k in range(latent_dim)], dtype=np.float32)
        item_bias = float(values.get("item_bias", 0.0))
        item_latents[str(text)] = (item_factors, item_bias)

    calibrator_mean = np.asarray(calibrator["scaler_mean"], dtype=np.float32)
    calibrator_scale = np.asarray(
        [float(x) if float(x) != 0.0 else 1.0 for x in calibrator["scaler_scale"]],
        dtype=np.float32,
    )
    calibrator_coef = np.asarray(calibrator["coef"], dtype=np.float32)
    calibrator_intercept = float(calibrator["intercept"])

    pairwise_rows = []
    pairwise_y = []
    for row in train_df.itertuples(index=False):
        subject_key = str(row.subject_name).strip().lower()
        subject_vector = subject_factors.get(subject_key, global_subject_factor)
        item_factors, item_bias = item_latents[str(row.item_content)]
        factor_logit = item_bias + float(np.dot(subject_vector, item_factors))
        factor_logit = max(-logit_cap, min(logit_cap, factor_logit))

        subject_rate = smooth(subject_rates.get(subject_key), subject_alpha)
        condition_rate = smooth(condition_rates.get(str(row.condition).strip().lower()), condition_alpha)
        benchmark_rate = smooth(benchmark_rates.get(str(row.benchmark_id).strip().lower()), benchmark_alpha)
        subject_logit = logit(subject_rate)
        condition_logit = logit(condition_rate)
        benchmark_logit = logit(benchmark_rate)
        adjustment = item_adjustment(row.item_content)
        base_features = np.asarray(
            [
                subject_logit,
                condition_logit,
                benchmark_logit,
                adjustment,
                factor_logit,
                subject_logit * factor_logit,
                condition_logit * factor_logit,
                benchmark_logit * factor_logit,
                abs(factor_logit),
            ],
            dtype=np.float32,
        )
        calibrated_logit = calibrator_intercept + float(
            np.dot(calibrator_coef, (base_features - calibrator_mean) / calibrator_scale)
        )
        _, benchmark_flag, cue_score = pairwise_cue_score(row.benchmark_id, row.condition, row.item_content)
        pairwise_rows.append(
            [
                calibrated_logit,
                *[float(x) for x in base_features],
                float(benchmark_flag),
                float(cue_score),
            ]
        )
        pairwise_y.append(int(row.label))

    pairwise_x = np.asarray(pairwise_rows, dtype=np.float32)
    pairwise_y_array = np.asarray(pairwise_y, dtype=np.int64)
    if len(set(pairwise_y)) != 2:
        raise ValueError("Pairwise adapter labels contain only one class")
    pairwise_scaler = StandardScaler()
    pairwise_x_scaled = pairwise_scaler.fit_transform(pairwise_x).astype(np.float32)
    pairwise_calibrator = LogisticRegression(
        C=float(pairwise_c),
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=326,
    )
    pairwise_calibrator.fit(pairwise_x_scaled, pairwise_y_array)

    pairwise_residual = {
        "enabled": True,
        "weight": float(pairwise_residual_weight),
        "known_benchmarks": sorted(pairwise_benchmarks),
        "keywords": list(pairwise_keywords),
        "feature_names": [
            "global_calibrated_logit",
            "subject_logit",
            "condition_logit",
            "benchmark_logit",
            "item_adjustment",
            "factor_logit",
            "subject_x_factor",
            "condition_x_factor",
            "benchmark_x_factor",
            "abs_factor_logit",
            "known_pairwise_benchmark",
            "pairwise_cue_score",
        ],
        "rows": int(len(pairwise_y_array)),
        "positive_rows": int(pairwise_y_array.sum()),
        "negative_rows": int(len(pairwise_y_array) - pairwise_y_array.sum()),
        "intercept": float(pairwise_calibrator.intercept_[0]),
        "coef": [float(x) for x in pairwise_calibrator.coef_[0]],
        "scaler_mean": [float(x) for x in pairwise_scaler.mean_],
        "scaler_scale": [float(x) if float(x) != 0.0 else 1.0 for x in pairwise_scaler.scale_],
        "c": float(pairwise_c),
    }

    artifact = dict(base_artifact)
    artifact["pairwise_residual"] = pairwise_residual
    summary = {
        "base_train_rows": 1_086_835,
        "base_items": 88_886,
        "base_source": "ridge-k3-learnable-10epoch.zip",
        "pairwise_residual_enabled": True,
        "pairwise_residual_rows": int(len(pairwise_y_array)),
        "pairwise_residual_positive_rows": int(pairwise_y_array.sum()),
        "pairwise_residual_negative_rows": int(len(pairwise_y_array) - pairwise_y_array.sum()),
        "pairwise_residual_weight": float(pairwise_residual_weight),
        "pairwise_c": float(pairwise_c),
        "train_rows_requested": int(train_rows),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return {"artifact": artifact, "summary": summary}


@app.local_entrypoint()
def main(
    train_rows: int = 1_500_000,
    pairwise_residual_weight: float = 0.75,
    pairwise_c: float = 0.5,
) -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "artifacts"
    stats_path = out_dir / "baseline_stats.json"
    artifact_path = out_dir / "bge_kfactor_ridge_artifact.json"
    frozen_base_path = out_dir / "bge_kfactor_ridge_base_10epoch_artifact.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    base_artifact = json.loads(
        (frozen_base_path if frozen_base_path.exists() else artifact_path).read_text(encoding="utf-8")
    )
    base_artifact.pop("pairwise_residual", None)
    payload = train_adapter_remote.remote(
        stats,
        base_artifact,
        train_rows,
        pairwise_residual_weight,
        pairwise_c,
    )
    artifact_path.write_text(
        json.dumps(payload["artifact"], separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "bge_kfactor_ridge_summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {artifact_path}")
