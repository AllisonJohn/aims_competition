from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("cs321m-final-bge-kfactor-k3-neural-residual")

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


@app.function(gpu="H100", image=image, timeout=60 * 60 * 6)
def train_remote(
    baseline_stats: dict,
    train_rows: int = 1_500_000,
    item_limit: int = 120_000,
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    latent_dim: int = 3,
    irt_epochs: int = 10,
    irt_lr: float = 3e-3,
    irt_l2: float = 1e-3,
    irt_batch_size: int = 8192,
    ridge_alpha: float = 300.0,
    blend_weight: float = 0.2,
    logit_cap: float = 4.0,
    residual_hidden_dim: int = 8,
    residual_epochs: int = 3,
    residual_lr: float = 1e-3,
    residual_scale: float = 0.15,
) -> dict:
    import math

    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
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

    subject_codes, subject_uniques = pd.factorize(train_df["subject_name"], sort=True)
    item_codes, item_uniques = pd.factorize(train_df["item_key"], sort=True)
    y = train_df["label"].to_numpy(dtype=np.float32)

    class KFactorIRT(torch.nn.Module):
        def __init__(self, num_subjects: int, num_items: int, dim: int):
            super().__init__()
            self.subject_factors = torch.nn.Embedding(num_subjects, dim)
            self.item_factors = torch.nn.Embedding(num_items, dim)
            self.item_bias = torch.nn.Embedding(num_items, 1)
            torch.nn.init.normal_(self.subject_factors.weight, std=0.05)
            torch.nn.init.normal_(self.item_factors.weight, std=0.05)
            torch.nn.init.zeros_(self.item_bias.weight)

        def forward(self, subjects_tensor, items_tensor):
            u = self.subject_factors(subjects_tensor)
            v = self.item_factors(items_tensor)
            z = self.item_bias(items_tensor).squeeze(-1)
            return (u * v).sum(dim=1) + z, u, v

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = TensorDataset(
        torch.as_tensor(subject_codes, dtype=torch.long),
        torch.as_tensor(item_codes, dtype=torch.long),
        torch.as_tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=irt_batch_size, shuffle=True, num_workers=2, pin_memory=device == "cuda")
    irt = KFactorIRT(len(subject_uniques), len(item_uniques), latent_dim).to(device)
    optimizer = torch.optim.AdamW(irt.parameters(), lr=irt_lr, weight_decay=0.0)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(
        "Fitting K-factor IRT: "
        f"subjects={len(subject_uniques)} items={len(item_uniques)} examples={len(train_df)} "
        f"K={latent_dim} epochs={irt_epochs} lr={irt_lr} l2={irt_l2}",
        flush=True,
    )
    for epoch in range(1, irt_epochs + 1):
        running_loss = 0.0
        seen = 0
        for batch_index, (subjects_batch, items_batch, labels_batch) in enumerate(loader, start=1):
            subjects_batch = subjects_batch.to(device, non_blocking=True)
            items_batch = items_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            logits, u_batch, v_batch = irt(subjects_batch, items_batch)
            bce = criterion(logits, labels_batch)
            l2 = u_batch.pow(2).mean() + v_batch.pow(2).mean()
            loss = bce + irt_l2 * l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = int(labels_batch.numel())
            running_loss += float(bce.detach().cpu()) * batch_size
            seen += batch_size
            if batch_index % 100 == 0:
                print(
                    f"irt_epoch={epoch} batch={batch_index}/{len(loader)} "
                    f"batch_bce={float(bce.detach().cpu()):.6f} running_bce={running_loss / seen:.6f}",
                    flush=True,
                )
        print(f"irt_epoch={epoch} train_bce={running_loss / max(seen, 1):.6f}", flush=True)

    with torch.no_grad():
        subject_factor_matrix = irt.subject_factors.weight.detach().cpu().numpy().astype(np.float32)
        item_factor_matrix = irt.item_factors.weight.detach().cpu().numpy().astype(np.float32)
        item_bias = irt.item_bias.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)

    item_targets = pd.DataFrame({"item_key": item_uniques, "item_bias": item_bias})
    for k in range(latent_dim):
        item_targets[f"factor_{k}"] = item_factor_matrix[:, k]
    item_meta = train_df[["item_key", "benchmark_id", "item_content"]].drop_duplicates("item_key")
    item_targets = item_targets.merge(item_meta, on="item_key", how="left")
    if item_limit > 0 and len(item_targets) > item_limit:
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
                batch = [f"Represent this evaluation question for capability-factor prediction: {t}" for t in batch]
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
                if (start // batch_size + 1) % 25 == 0:
                    print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)} item targets", flush=True)
            return np.vstack(chunks)

    print(f"Training {latent_dim + 1} ridge heads for item factors/bias on {len(item_targets)} items", flush=True)
    embedder = Embedder()
    embeddings = embedder.encode(item_targets["item_content"].fillna("").tolist())
    x_train = embeddings.astype(np.float32)

    ridge_heads = []
    target_names = [f"factor_{k}" for k in range(latent_dim)] + ["item_bias"]
    for target_name in target_names:
        target = item_targets[target_name].to_numpy(dtype=np.float32)
        target_mean = float(target.mean())
        target_std = float(target.std())
        if target_std < 1e-6:
            target_std = 1.0
        target_normalized = ((target - target_mean) / target_std).astype(np.float32)
        ridge = Ridge(alpha=ridge_alpha, random_state=326)
        ridge.fit(x_train, target_normalized)
        print(
            f"ridge_head={target_name} target_mean={target_mean:.6f} "
            f"target_std={target_std:.6f} normalized_train_std={target_normalized.std():.6f}",
            flush=True,
        )
        ridge_heads.append(
            {
                "name": target_name,
                "ridge_intercept": float(ridge.intercept_),
                "ridge_coef": [float(x) for x in ridge.coef_],
                "ridge_alpha": float(ridge_alpha),
                "target_mean": target_mean,
                "target_std": target_std,
                "target_standardized": True,
            }
        )

    subject_factors = {
        str(name).strip().lower(): [float(x) for x in subject_factor_matrix[index]]
        for index, name in enumerate(subject_uniques)
    }
    global_subject_factor = [float(x) for x in subject_factor_matrix.mean(axis=0)]

    factor_logits = item_bias[item_codes] + np.sum(
        subject_factor_matrix[subject_codes] * item_factor_matrix[item_codes],
        axis=1,
    )
    factor_logits = np.clip(factor_logits, -logit_cap, logit_cap)
    global_rate = float(train_df["label"].mean())
    subject_rates = rate_table(train_df, "subject_name")
    condition_rates = rate_table(train_df, "condition")
    benchmark_rates = rate_table(train_df, "benchmark_id")
    subject_alpha = 250.0
    condition_alpha = 2000.0
    benchmark_alpha = 5000.0

    calibration_rows = []
    for row_index, row in enumerate(train_df.itertuples(index=False)):
        subject_rate = smooth(subject_rates.get(str(row.subject_name).strip().lower()), subject_alpha, global_rate)
        condition_rate = smooth(condition_rates.get(str(row.condition).strip().lower()), condition_alpha, global_rate)
        benchmark_rate = smooth(benchmark_rates.get(str(row.benchmark_id).strip().lower()), benchmark_alpha, global_rate)
        subject_logit = logit(subject_rate)
        condition_logit = logit(condition_rate)
        benchmark_logit = logit(benchmark_rate)
        factor_logit = float(factor_logits[row_index])
        adjustment = item_adjustment(row.item_content)
        calibration_rows.append(
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
    base_calibration_logits = (
        float(calibrator.intercept_[0])
        + calibration_x_scaled @ calibrator.coef_[0].astype(np.float32)
    ).astype(np.float32)

    class ResidualMLP(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, 1)
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.zeros_(self.fc2.weight)
            torch.nn.init.zeros_(self.fc2.bias)

        def forward(self, x_tensor):
            h = torch.tanh(self.fc1(x_tensor))
            return self.fc2(h).squeeze(-1)

    residual_dataset = TensorDataset(
        torch.as_tensor(calibration_x_scaled, dtype=torch.float32),
        torch.as_tensor(base_calibration_logits, dtype=torch.float32),
        torch.as_tensor(calibration_y.astype(np.float32), dtype=torch.float32),
    )
    residual_loader = DataLoader(
        residual_dataset,
        batch_size=32768,
        shuffle=True,
        num_workers=2,
        pin_memory=device == "cuda",
    )
    residual_model = ResidualMLP(calibration_x_scaled.shape[1], residual_hidden_dim).to(device)
    residual_optimizer = torch.optim.AdamW(residual_model.parameters(), lr=residual_lr, weight_decay=1e-4)
    residual_criterion = torch.nn.BCEWithLogitsLoss()
    print(
        "Training neural residual calibrator: "
        f"hidden={residual_hidden_dim} epochs={residual_epochs} lr={residual_lr} scale={residual_scale}",
        flush=True,
    )
    for epoch in range(1, residual_epochs + 1):
        running_loss = 0.0
        seen = 0
        for features_batch, base_logits_batch, labels_batch in residual_loader:
            features_batch = features_batch.to(device, non_blocking=True)
            base_logits_batch = base_logits_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            residual = residual_model(features_batch)
            logits = base_logits_batch + residual_scale * residual
            loss = residual_criterion(logits, labels_batch)
            residual_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            residual_optimizer.step()
            batch_size = int(labels_batch.numel())
            running_loss += float(loss.detach().cpu()) * batch_size
            seen += batch_size
        print(f"residual_epoch={epoch} train_bce={running_loss / max(seen, 1):.6f}", flush=True)

    residual_state = {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in residual_model.state_dict().items()
    }

    return {
        "artifact": {
            "encoder_id": encoder_id,
            "latent_dim": int(latent_dim),
            "blend_weight": float(blend_weight),
            "logit_cap": float(logit_cap),
            "calibrator": {
                "feature_names": [
                    "subject_logit",
                    "condition_logit",
                    "benchmark_logit",
                    "item_adjustment",
                    "factor_logit",
                    "subject_x_factor",
                    "condition_x_factor",
                    "benchmark_x_factor",
                    "abs_factor_logit",
                ],
                "intercept": float(calibrator.intercept_[0]),
                "coef": [float(x) for x in calibrator.coef_[0]],
                "scaler_mean": [float(x) for x in calibration_scaler.mean_],
                "scaler_scale": [float(x) if float(x) != 0.0 else 1.0 for x in calibration_scaler.scale_],
                "c": 1.0,
                "neural_residual": {
                    "hidden_dim": int(residual_hidden_dim),
                    "scale": float(residual_scale),
                    "activation": "tanh",
                    "fc1_weight": residual_state["fc1.weight"].tolist(),
                    "fc1_bias": residual_state["fc1.bias"].tolist(),
                    "fc2_weight": residual_state["fc2.weight"].reshape(-1).tolist(),
                    "fc2_bias": float(residual_state["fc2.bias"][0]),
                },
            },
            "subject_factors": subject_factors,
            "global_subject_factor": global_subject_factor,
            "ridge_heads": ridge_heads,
            "scaler_mean": [],
            "scaler_scale": [],
            "embedding_dim": int(embeddings.shape[1]),
            "numeric_dim": 0,
            "max_length": 256,
        },
        "summary": {
            "train_rows": int(len(train_df)),
            "subjects": int(len(subject_uniques)),
            "items": int(len(item_uniques)),
            "text_item_targets": int(len(item_targets)),
            "encoder": encoder_id,
            "latent_dim": int(latent_dim),
            "irt_epochs": int(irt_epochs),
            "irt_lr": float(irt_lr),
            "irt_l2": float(irt_l2),
            "ridge_alpha": float(ridge_alpha),
            "ridge_heads": int(len(ridge_heads)),
            "ridge_targets_standardized": True,
            "blend_weight": float(blend_weight),
            "logit_cap": float(logit_cap),
            "calibrator_features": int(calibration_x.shape[1]),
            "neural_residual_hidden_dim": int(residual_hidden_dim),
            "neural_residual_epochs": int(residual_epochs),
            "neural_residual_lr": float(residual_lr),
            "neural_residual_scale": float(residual_scale),
        },
    }


@app.local_entrypoint()
def main(
    train_rows: int = 0, #1_500_000,
    item_limit: int = 0, #120_000,
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    latent_dim: int = 3,
    irt_epochs: int = 10,
    irt_lr: float = 3e-3,
    irt_l2: float = 1e-3,
    irt_batch_size: int = 8192,
    ridge_alpha: float = 300.0,
    blend_weight: float = 0.2,
    logit_cap: float = 4.0,
    residual_hidden_dim: int = 8,
    residual_epochs: int = 3,
    residual_lr: float = 1e-3,
    residual_scale: float = 0.15,
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
        latent_dim,
        irt_epochs,
        irt_lr,
        irt_l2,
        irt_batch_size,
        ridge_alpha,
        blend_weight,
        logit_cap,
        residual_hidden_dim,
        residual_epochs,
        residual_lr,
        residual_scale,
    )
    (out_dir / "bge_kfactor_ridge_artifact.json").write_text(
        json.dumps(payload["artifact"], separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "bge_kfactor_ridge_summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {out_dir / 'bge_kfactor_ridge_artifact.json'}")
