from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("cs321m-final-bge-2pl-learned-coeffs")

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
    blend_weight: float = 0.20,
    subject_weight: float = 1.0,
    condition_weight: float = 0.0,
    benchmark_weight: float = 0.0,
    item_adjustment_weight: float = 0.0,
    irt_epochs: int = 8,
    irt_lr: float = 3e-3,
    irt_l2: float = 1e-3,
    irt_batch_size: int = 8192,
    ridge_alpha: float = 300.0,
    discrimination_floor: float = 0.05,
    discrimination_cap: float = 5.0,
) -> dict:
    import math

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer

    repo_id = "aims-foundations/measurement-db"

    def clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
        return max(lo, min(hi, float(value)))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def logit(p):
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
        if any(word in lower for word in ("prove", "derive", "justify", "rigorously", "counterexample")):
            adjustment -= 0.04
        if any(word in lower for word in ("def ", "class ", "import ", "function", "bug", "debug", "runtime", "algorithm")):
            adjustment -= 0.03
        if any(word in lower for word in ("what is", "who is", "when did", "where is")) and length < 240:
            adjustment += 0.03
        if any(word in lower for word in ("a)", "b)", "c)", "d)", "multiple choice", "choose the best")):
            adjustment += 0.01
        return adjustment

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
        df["item_key"] = df["benchmark_id"].astype(str) + "::" + df["item_id"].astype(str)
        df["label"] = [
            binary_label(bench, response)
            for bench, response in zip(df["benchmark_id"], df["response"])
        ]
        parts.append(df[["subject_name", "item_key", "benchmark_id", "condition", "item_content", "label"]])
    train_df = pd.concat(parts, ignore_index=True)
    if train_rows > 0 and len(train_df) > train_rows:
        train_df = train_df.sample(n=train_rows, random_state=326)
    train_df = train_df.reset_index(drop=True)

    subject_codes, subject_uniques = pd.factorize(train_df["subject_name"], sort=True)
    item_codes, item_uniques = pd.factorize(train_df["item_key"], sort=True)
    y = train_df["label"].to_numpy(dtype=np.float32)
    print(
        f"Fitting 2PL IRT: rows={len(train_df)} subjects={len(subject_uniques)} "
        f"items={len(item_uniques)} epochs={irt_epochs} lr={irt_lr} l2={irt_l2}",
        flush=True,
    )

    subject_rate = train_df.groupby("subject_name")["label"].mean().reindex(subject_uniques).fillna(train_df["label"].mean())
    subject_theta_init = np.asarray([logit(rate) for rate in subject_rate], dtype=np.float32)
    subject_theta_init -= float(subject_theta_init.mean())
    item_rate = train_df.groupby("item_key")["label"].mean().reindex(item_uniques).fillna(train_df["label"].mean())
    item_beta_init = np.asarray([-logit(rate) for rate in item_rate], dtype=np.float32)
    item_beta_init -= float(item_beta_init.mean())

    class TwoPL(torch.nn.Module):
        def __init__(self, n_subjects: int, n_items: int):
            super().__init__()
            self.theta = torch.nn.Embedding(n_subjects, 1)
            self.beta = torch.nn.Embedding(n_items, 1)
            self.raw_disc = torch.nn.Embedding(n_items, 1)

        def forward(self, subject_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
            theta_value = self.theta(subject_idx).squeeze(-1)
            beta_value = self.beta(item_idx).squeeze(-1)
            discrimination = F.softplus(self.raw_disc(item_idx)).squeeze(-1) + float(discrimination_floor)
            if discrimination_cap > 0:
                discrimination = discrimination.clamp(max=float(discrimination_cap))
            return discrimination * theta_value - beta_value

    device = "cuda" if torch.cuda.is_available() else "cpu"
    irt = TwoPL(len(subject_uniques), len(item_uniques)).to(device)
    with torch.no_grad():
        irt.theta.weight[:, 0].copy_(torch.from_numpy(subject_theta_init).to(device))
        irt.beta.weight[:, 0].copy_(torch.from_numpy(item_beta_init).to(device))
        # softplus(0.54) + 0.05 is about 1.05, so we start near 1PL and let 2PL move.
        irt.raw_disc.weight.fill_(0.54)
    optimizer = torch.optim.AdamW(irt.parameters(), lr=irt_lr, weight_decay=0.0)
    subjects_tensor = torch.from_numpy(subject_codes.astype(np.int64))
    items_tensor = torch.from_numpy(item_codes.astype(np.int64))
    labels_tensor = torch.from_numpy(y)
    generator = torch.Generator().manual_seed(326)
    for epoch in range(1, int(irt_epochs) + 1):
        permutation = torch.randperm(len(labels_tensor), generator=generator)
        total_loss = 0.0
        total_examples = 0
        for start in range(0, len(labels_tensor), int(irt_batch_size)):
            idx = permutation[start : start + int(irt_batch_size)]
            subject_batch = subjects_tensor[idx].to(device)
            item_batch = items_tensor[idx].to(device)
            label_batch = labels_tensor[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = irt(subject_batch, item_batch)
            bce = F.binary_cross_entropy_with_logits(logits, label_batch)
            reg = irt_l2 * (
                irt.theta.weight.square().mean()
                + irt.beta.weight.square().mean()
                + irt.raw_disc.weight.square().mean()
            )
            loss = bce + reg
            loss.backward()
            optimizer.step()
            total_loss += float(bce.detach().cpu()) * len(idx)
            total_examples += len(idx)
        with torch.no_grad():
            irt.theta.weight[:, 0] -= irt.theta.weight[:, 0].mean()
        print(f"2pl_epoch={epoch} train_bce={total_loss / max(total_examples, 1):.6f}", flush=True)

    with torch.no_grad():
        theta = irt.theta.weight[:, 0].detach().cpu().numpy().astype(np.float32)
        beta = irt.beta.weight[:, 0].detach().cpu().numpy().astype(np.float32)
        discrimination = (
            F.softplus(irt.raw_disc.weight[:, 0]).detach().cpu().numpy().astype(np.float32)
            + float(discrimination_floor)
        )
        if discrimination_cap > 0:
            discrimination = np.minimum(discrimination, float(discrimination_cap)).astype(np.float32)
    log_discrimination = np.log(np.clip(discrimination, 1e-4, None)).astype(np.float32)
    global_log_discrimination = float(log_discrimination.mean())
    subject_abilities = {
        str(subject): float(value)
        for subject, value in zip(subject_uniques, theta)
    }

    item_targets = pd.DataFrame(
        {
            "item_key": item_uniques,
            "difficulty": beta,
            "discrimination": discrimination,
            "log_discrimination": log_discrimination,
        }
    )
    item_meta = train_df[["item_key", "benchmark_id", "item_content"]].drop_duplicates("item_key")
    item_targets = item_targets.merge(item_meta, on="item_key", how="left")
    item_targets["centered_difficulty"] = item_targets["difficulty"] - item_targets.groupby("benchmark_id")[
        "difficulty"
    ].transform("mean")
    item_targets["centered_log_discrimination"] = item_targets["log_discrimination"] - global_log_discrimination
    if item_limit > 0 and len(item_targets) > item_limit:
        item_targets = item_targets.sample(n=item_limit, random_state=326)

    class Embedder:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_id)
            self.model = AutoModel.from_pretrained(encoder_id).to(device)
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
                encoded = {key: value.to(device) for key, value in encoded.items()}
                output = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
                pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                chunks.append(pooled.cpu().numpy().astype(np.float32))
                print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)} item targets", flush=True)
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

    difficulty_model = Ridge(alpha=ridge_alpha, random_state=326)
    difficulty_model.fit(x_train, item_targets["centered_difficulty"].to_numpy(dtype=np.float32))
    log_disc_model = Ridge(alpha=ridge_alpha, random_state=326)
    log_disc_model.fit(x_train, item_targets["centered_log_discrimination"].to_numpy(dtype=np.float32))
    item_targets["difficulty_hat"] = difficulty_model.predict(x_train).astype(np.float32)
    item_targets["log_discrimination_hat"] = (
        log_disc_model.predict(x_train).astype(np.float32) + global_log_discrimination
    )
    difficulty_by_item = dict(zip(item_targets["item_key"], item_targets["difficulty_hat"]))
    log_disc_by_item = dict(zip(item_targets["item_key"], item_targets["log_discrimination_hat"]))

    global_rate = float(train_df["label"].mean())
    subject_rates = rate_table(train_df, "subject_name")
    condition_rates = rate_table(train_df, "condition")
    benchmark_rates = rate_table(train_df, "benchmark_id")
    train_df["difficulty_hat"] = train_df["item_key"].map(difficulty_by_item).fillna(0.0).astype(float)
    train_df["log_discrimination_hat"] = train_df["item_key"].map(log_disc_by_item).fillna(global_log_discrimination).astype(float)
    train_df["theta_hat"] = train_df["subject_name"].map(subject_abilities).fillna(0.0).astype(float)
    train_df["two_pl_logit"] = (
        np.exp(np.clip(train_df["log_discrimination_hat"].to_numpy(dtype=float), math.log(0.05), math.log(5.0)))
        * train_df["theta_hat"].to_numpy(dtype=float)
        - train_df["difficulty_hat"].to_numpy(dtype=float)
    )
    subject_alpha = 250.0
    condition_alpha = 2000.0
    benchmark_alpha = 5000.0

    calibration_rows = []
    for row in train_df.itertuples(index=False):
        subject_rate_value = smooth(subject_rates.get(str(row.subject_name).strip().lower()), subject_alpha, global_rate)
        condition_rate = smooth(condition_rates.get(str(row.condition).strip().lower()), condition_alpha, global_rate)
        benchmark_rate = smooth(benchmark_rates.get(str(row.benchmark_id).strip().lower()), benchmark_alpha, global_rate)
        subject_logit = logit(subject_rate_value)
        condition_logit = logit(condition_rate)
        benchmark_logit = logit(benchmark_rate)
        adjustment = item_adjustment(row.item_content)
        difficulty = float(row.difficulty_hat)
        log_disc = float(row.log_discrimination_hat)
        two_pl_logit = float(row.two_pl_logit)
        calibration_rows.append(
            [
                subject_logit,
                condition_logit,
                benchmark_logit,
                adjustment,
                difficulty,
                log_disc,
                two_pl_logit,
                subject_logit * difficulty,
                subject_logit * log_disc,
                condition_logit * difficulty,
                benchmark_logit * difficulty,
                abs(difficulty),
            ]
        )
    calibration_x = np.asarray(calibration_rows, dtype=np.float32)
    calibration_y = train_df["label"].to_numpy(dtype=np.int64)
    calibration_scaler = StandardScaler()
    calibration_x_scaled = calibration_scaler.fit_transform(calibration_x).astype(np.float32)
    calibrator = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=326,
    )
    calibrator.fit(calibration_x_scaled, calibration_y)

    feature_names = [
        "subject_logit",
        "condition_logit",
        "benchmark_logit",
        "item_adjustment",
        "difficulty",
        "log_discrimination",
        "two_pl_logit",
        "subject_x_difficulty",
        "subject_x_log_discrimination",
        "condition_x_difficulty",
        "benchmark_x_difficulty",
        "abs_difficulty",
    ]

    return {
        "artifact": {
            "irt_model": "2pl",
            "encoder_id": encoder_id,
            "blend_weight": float(blend_weight),
            "prior_weights": {
                "subject": float(subject_weight),
                "condition": float(condition_weight),
                "benchmark": float(benchmark_weight),
                "item_adjustment": float(item_adjustment_weight),
            },
            "calibrator": {
                "feature_names": feature_names,
                "intercept": float(calibrator.intercept_[0]),
                "coef": [float(x) for x in calibrator.coef_[0]],
                "scaler_mean": [float(x) for x in calibration_scaler.mean_],
                "scaler_scale": [float(x) if float(x) != 0.0 else 1.0 for x in calibration_scaler.scale_],
                "c": 1.0,
            },
            "difficulty_ridge_intercept": float(difficulty_model.intercept_),
            "difficulty_ridge_coef": [float(x) for x in difficulty_model.coef_],
            "log_discrimination_ridge_intercept": float(log_disc_model.intercept_),
            "log_discrimination_ridge_coef": [float(x) for x in log_disc_model.coef_],
            "global_log_discrimination": global_log_discrimination,
            "subject_abilities": subject_abilities,
            "global_subject_ability": 0.0,
            "discrimination_floor": float(discrimination_floor),
            "discrimination_cap": float(discrimination_cap),
            "scaler_mean": [float(x) for x in scaler.mean_],
            "scaler_scale": [float(x) if float(x) != 0.0 else 1.0 for x in scaler.scale_],
            "embedding_dim": int(embeddings.shape[1]),
            "numeric_dim": int(numeric.shape[1]),
            "max_length": 256,
        },
        "summary": {
            "train_rows": int(len(train_df)),
            "irt_model": "2pl",
            "irt_subjects": int(len(subject_uniques)),
            "irt_items": int(len(item_uniques)),
            "irt_epochs": int(irt_epochs),
            "irt_lr": float(irt_lr),
            "irt_l2": float(irt_l2),
            "discrimination_mean": float(discrimination.mean()),
            "discrimination_std": float(discrimination.std()),
            "text_item_targets": int(len(item_targets)),
            "encoder": encoder_id,
            "ridge_alpha": float(ridge_alpha),
            "blend_weight": float(blend_weight),
            "calibrator_features": int(calibration_x.shape[1]),
            "prior_weights": {
                "subject": float(subject_weight),
                "condition": float(condition_weight),
                "benchmark": float(benchmark_weight),
                "item_adjustment": float(item_adjustment_weight),
            },
        },
    }


@app.local_entrypoint()
def main(
    train_rows: int = 1_500_000,
    item_limit: int = 120_000,
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    blend_weight: float = 0.20,
    subject_weight: float = 1.0,
    condition_weight: float = 0.0,
    benchmark_weight: float = 0.0,
    item_adjustment_weight: float = 0.0,
    irt_epochs: int = 8,
    irt_lr: float = 3e-3,
    irt_l2: float = 1e-3,
    irt_batch_size: int = 8192,
    ridge_alpha: float = 300.0,
    discrimination_floor: float = 0.05,
    discrimination_cap: float = 5.0,
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
        subject_weight,
        condition_weight,
        benchmark_weight,
        item_adjustment_weight,
        irt_epochs,
        irt_lr,
        irt_l2,
        irt_batch_size,
        ridge_alpha,
        discrimination_floor,
        discrimination_cap,
    )
    (out_dir / "bge_2pl_ridge_artifact.json").write_text(
        json.dumps(payload["artifact"], separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "bge_2pl_ridge_summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {out_dir / 'bge_2pl_ridge_artifact.json'}")
