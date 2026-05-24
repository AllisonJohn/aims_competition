from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("cs321m-final-bge-learned-skip-mlp-head-k3-interaction")

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
    blend_weight: float = 0.45,
    subject_weight: float = 0.68,
    condition_weight: float = 0.22,
    benchmark_weight: float = 0.10,
    item_adjustment_weight: float = 1.0,
    item_head_hidden_dim: int = 128,
    item_head_epochs: int = 50,
    item_head_lr: float = 1e-3,
    item_head_weight_decay: float = 1e-4,
    item_head_batch_size: int = 4096,
    interaction_latent_dim: int = 3,
    interaction_epochs: int = 10,
    interaction_lr: float = 3e-3,
    interaction_l2: float = 1e-3,
    interaction_batch_size: int = 8192,
    interaction_ridge_alpha: float = 300.0,
) -> dict:
    import math

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
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
        df["item_key"] = df["benchmark_id"].astype(str) + "::" + df["item_id"].astype(str)
        df["label"] = [
            binary_label(bench, response)
            for bench, response in zip(df["benchmark_id"], df["response"])
        ]
        parts.append(df[["subject_name", "item_key", "benchmark_id", "condition", "item_content", "label"]])
    train_df = pd.concat(parts, ignore_index=True)
    if row_limit is not None and len(train_df) > row_limit:
        train_df = train_df.sample(n=row_limit, random_state=326)

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

    class KFactorIRT(torch.nn.Module):
        def __init__(self, num_subjects: int, num_items: int, dim: int):
            super().__init__()
            self.subject_factors = torch.nn.Embedding(num_subjects, dim)
            self.item_factors = torch.nn.Embedding(num_items, dim)
            torch.nn.init.normal_(self.subject_factors.weight, std=0.05)
            torch.nn.init.normal_(self.item_factors.weight, std=0.05)

        def forward(self, subjects_tensor, items_tensor):
            u = self.subject_factors(subjects_tensor)
            v = self.item_factors(items_tensor)
            return (u * v).sum(dim=1), u, v

    device = "cuda" if torch.cuda.is_available() else "cpu"
    interaction_dataset = TensorDataset(
        torch.as_tensor(subject_codes, dtype=torch.long),
        torch.as_tensor(item_codes, dtype=torch.long),
        torch.as_tensor((theta[subject_codes] - beta[item_codes]).astype(np.float32), dtype=torch.float32),
        torch.as_tensor(y.astype(np.float32), dtype=torch.float32),
    )
    interaction_loader = DataLoader(
        interaction_dataset,
        batch_size=int(interaction_batch_size),
        shuffle=True,
        num_workers=2,
        pin_memory=device == "cuda",
    )
    interaction_model = KFactorIRT(
        len(subject_uniques),
        len(item_uniques),
        int(interaction_latent_dim),
    ).to(device)
    interaction_optimizer = torch.optim.AdamW(
        interaction_model.parameters(),
        lr=float(interaction_lr),
        weight_decay=0.0,
    )
    interaction_criterion = torch.nn.BCEWithLogitsLoss()
    print(
        "Fitting k-factor interaction teacher: "
        f"subjects={len(subject_uniques)} items={len(item_uniques)} examples={len(train_df)} "
        f"K={interaction_latent_dim} epochs={interaction_epochs} lr={interaction_lr} l2={interaction_l2} "
        "base=rasch_logit",
        flush=True,
    )
    for epoch in range(1, int(interaction_epochs) + 1):
        running_loss = 0.0
        seen = 0
        for subjects_batch, items_batch, base_logits_batch, labels_batch in interaction_loader:
            subjects_batch = subjects_batch.to(device, non_blocking=True)
            items_batch = items_batch.to(device, non_blocking=True)
            base_logits_batch = base_logits_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            residual_logits, u_batch, v_batch = interaction_model(subjects_batch, items_batch)
            logits = base_logits_batch + residual_logits
            bce = interaction_criterion(logits, labels_batch)
            l2 = u_batch.pow(2).mean() + v_batch.pow(2).mean()
            loss = bce + float(interaction_l2) * l2
            interaction_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            interaction_optimizer.step()
            batch_size = int(labels_batch.numel())
            running_loss += float(bce.detach().cpu()) * batch_size
            seen += batch_size
        print(f"interaction_epoch={epoch} train_bce={running_loss / max(seen, 1):.6f}", flush=True)

    with torch.no_grad():
        subject_factor_matrix = interaction_model.subject_factors.weight.detach().cpu().numpy().astype(np.float32)
        item_factor_matrix = interaction_model.item_factors.weight.detach().cpu().numpy().astype(np.float32)
    item_factor_targets = pd.DataFrame({"item_key": item_uniques})
    for k in range(int(interaction_latent_dim)):
        item_factor_targets[f"interaction_factor_{k}"] = item_factor_matrix[:, k]
    item_targets = item_targets.merge(item_factor_targets, on="item_key", how="left")

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

    class ItemMLPHead(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
            self.linear2 = torch.nn.Linear(hidden_dim, 1)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            hidden = F.leaky_relu(self.linear1(features), negative_slope=0.1)
            return self.linear2(hidden)

    item_head = ItemMLPHead(x_train.shape[1], int(item_head_hidden_dim)).to(device)
    item_optimizer = torch.optim.AdamW(
        item_head.parameters(),
        lr=float(item_head_lr),
        weight_decay=float(item_head_weight_decay),
    )
    item_x = torch.from_numpy(x_train)
    item_index = {str(key): idx for idx, key in enumerate(item_targets["item_key"].tolist())}
    row_item_index = train_df["item_key"].map(item_index).fillna(-1).to_numpy(dtype=np.int64)
    valid_rows = row_item_index >= 0
    bce_item_index = torch.from_numpy(row_item_index[valid_rows].astype(np.int64))
    bce_subject_theta = torch.from_numpy(theta[subject_codes[valid_rows]].astype(np.float32))
    item_irt_difficulty = torch.from_numpy(item_targets["difficulty"].to_numpy(dtype=np.float32))
    bce_label = torch.from_numpy(train_df["label"].to_numpy(dtype=np.float32)[valid_rows])
    item_generator = torch.Generator().manual_seed(326)
    print(
        f"Training 1PL item MLP residual head with row BCE: items={len(item_x)} "
        f"response_rows={len(bce_label)} input_dim={x_train.shape[1]} "
        f"hidden={item_head_hidden_dim} epochs={item_head_epochs} lr={item_head_lr} "
        f"weight_decay={item_head_weight_decay}",
        flush=True,
    )
    for epoch in range(1, int(item_head_epochs) + 1):
        permutation = torch.randperm(len(bce_label), generator=item_generator)
        total_loss = 0.0
        total_examples = 0
        for start in range(0, len(bce_label), int(item_head_batch_size)):
            idx = permutation[start : start + int(item_head_batch_size)]
            batch_item_idx = bce_item_index[idx]
            batch_x = item_x[batch_item_idx].to(device)
            batch_theta = bce_subject_theta[idx].to(device)
            batch_irt_difficulty = item_irt_difficulty[batch_item_idx].to(device)
            batch_y = bce_label[idx].to(device)
            item_optimizer.zero_grad(set_to_none=True)
            difficulty_residual = item_head(batch_x).squeeze(-1)
            logits = batch_theta - (batch_irt_difficulty + difficulty_residual)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y)
            loss.backward()
            item_optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
            total_examples += len(idx)
        if epoch == 1 or epoch % 5 == 0 or epoch == int(item_head_epochs):
            print(f"item_head_epoch={epoch} row_bce={total_loss / max(total_examples, 1):.6f}", flush=True)

    item_head.eval()
    with torch.no_grad():
        difficulty_residual_hat = item_head(item_x.to(device)).detach().cpu().numpy().astype(np.float32)
    item_targets["difficulty_hat"] = difficulty_residual_hat.reshape(-1).astype(np.float32)
    difficulty_by_item = dict(zip(item_targets["item_key"], item_targets["difficulty_hat"]))

    interaction_heads = []
    interaction_factor_hat_columns = []
    for k in range(int(interaction_latent_dim)):
        target_name = f"interaction_factor_{k}"
        target = item_targets[target_name].to_numpy(dtype=np.float32)
        target_mean = float(target.mean())
        target_std = float(target.std())
        if target_std < 1e-6:
            target_std = 1.0
        target_normalized = ((target - target_mean) / target_std).astype(np.float32)
        ridge = Ridge(alpha=float(interaction_ridge_alpha), random_state=326)
        ridge.fit(x_train, target_normalized)
        prediction = (ridge.predict(x_train).astype(np.float32) * target_std + target_mean).astype(np.float32)
        pred_col = f"{target_name}_hat"
        item_targets[pred_col] = prediction
        interaction_factor_hat_columns.append(pred_col)
        print(
            f"interaction_ridge_head={target_name} target_mean={target_mean:.6f} "
            f"target_std={target_std:.6f} alpha={interaction_ridge_alpha}",
            flush=True,
        )
        interaction_heads.append(
            {
                "name": target_name,
                "ridge_intercept": float(ridge.intercept_),
                "ridge_coef": [float(x) for x in ridge.coef_],
                "ridge_alpha": float(interaction_ridge_alpha),
                "target_mean": target_mean,
                "target_std": target_std,
            }
        )
    interaction_factor_by_item = {
        str(row.item_key): [float(getattr(row, col)) for col in interaction_factor_hat_columns]
        for row in item_targets.itertuples(index=False)
    }

    global_rate = float(train_df["label"].mean())
    subject_rates = rate_table(train_df, "subject_name")
    condition_rates = rate_table(train_df, "condition")
    benchmark_rates = rate_table(train_df, "benchmark_id")
    train_df["difficulty_hat"] = train_df["item_key"].map(difficulty_by_item).fillna(0.0).astype(float)
    row_interaction_vectors = train_df["item_key"].map(interaction_factor_by_item)
    interaction_logits = []
    for row_idx, vector in enumerate(row_interaction_vectors):
        if not isinstance(vector, list):
            interaction_logits.append(0.0)
            continue
        subject_vector = subject_factor_matrix[subject_codes[row_idx]]
        interaction_logits.append(float(np.dot(subject_vector, np.asarray(vector, dtype=np.float32))))
    train_df["interaction_logit"] = np.asarray(interaction_logits, dtype=np.float32)
    subject_alpha = 250.0
    condition_alpha = 2000.0
    benchmark_alpha = 5000.0

    calibration_rows = []
    for row in train_df.itertuples(index=False):
        subject_rate = smooth(subject_rates.get(str(row.subject_name).strip().lower()), subject_alpha, global_rate)
        condition_rate = smooth(condition_rates.get(str(row.condition).strip().lower()), condition_alpha, global_rate)
        benchmark_rate = smooth(benchmark_rates.get(str(row.benchmark_id).strip().lower()), benchmark_alpha, global_rate)
        subject_logit = logit(subject_rate)
        condition_logit = logit(condition_rate)
        benchmark_logit = logit(benchmark_rate)
        adjustment = item_adjustment(row.item_content)
        difficulty = float(row.difficulty_hat)
        interaction_logit = float(row.interaction_logit)
        calibration_rows.append(
            [
                subject_logit,
                condition_logit,
                benchmark_logit,
                adjustment,
                difficulty,
                interaction_logit,
                subject_logit * difficulty,
                condition_logit * difficulty,
                benchmark_logit * difficulty,
                abs(difficulty),
                subject_logit * interaction_logit,
                abs(interaction_logit),
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

    return {
        "artifact": {
            "encoder_id": encoder_id,
            "blend_weight": float(blend_weight),
            "prior_weights": {
                "subject": float(subject_weight),
                "condition": float(condition_weight),
                "benchmark": float(benchmark_weight),
                "item_adjustment": float(item_adjustment_weight),
            },
            "calibrator": {
                "feature_names": [
                    "subject_logit",
                    "condition_logit",
                    "benchmark_logit",
                    "item_adjustment",
                    "difficulty",
                    "interaction_logit",
                    "subject_x_difficulty",
                    "condition_x_difficulty",
                    "benchmark_x_difficulty",
                    "abs_difficulty",
                    "subject_x_interaction",
                    "abs_interaction",
                ],
                "intercept": float(calibrator.intercept_[0]),
                "coef": [float(x) for x in calibrator.coef_[0]],
                "scaler_mean": [float(x) for x in calibration_scaler.mean_],
                "scaler_scale": [float(x) if float(x) != 0.0 else 1.0 for x in calibration_scaler.scale_],
                "c": 1.0,
            },
            "item_head": {
                "type": "linear_leakyrelu_linear",
                "objective": "row_bce_1pl_irt_bias_residual",
                "output_semantics": "difficulty_residual_added_to_fitted_irt_bias_during_training",
                "input_dim": int(x_train.shape[1]),
                "hidden_dim": int(item_head_hidden_dim),
                "output_dim": 1,
                "target_mean": [0.0],
                "target_scale": [1.0],
                "linear1_weight": item_head.linear1.weight.detach().cpu().numpy().astype(float).tolist(),
                "linear1_bias": item_head.linear1.bias.detach().cpu().numpy().astype(float).tolist(),
                "linear2_weight": item_head.linear2.weight.detach().cpu().numpy().astype(float).tolist(),
                "linear2_bias": item_head.linear2.bias.detach().cpu().numpy().astype(float).tolist(),
            },
            "interaction_model": {
                "type": "kfactor_text_ridge_interaction",
                "objective": "row_bce_residual_on_rasch_logit",
                "latent_dim": int(interaction_latent_dim),
                "ridge_heads": interaction_heads,
                "subject_factors": {
                    str(subject_uniques[i]).strip().lower(): [
                        float(x) for x in subject_factor_matrix[i].astype(float).tolist()
                    ]
                    for i in range(len(subject_uniques))
                },
                "default_subject_factor": [
                    float(x) for x in subject_factor_matrix.mean(axis=0).astype(float).tolist()
                ],
            },
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
            "item_head_hidden_dim": int(item_head_hidden_dim),
            "item_head_epochs": int(item_head_epochs),
            "item_head_lr": float(item_head_lr),
            "item_head_weight_decay": float(item_head_weight_decay),
            "item_head_objective": "row_bce_1pl_irt_bias_residual",
            "item_head_response_rows": int(len(bce_label)),
            "interaction_latent_dim": int(interaction_latent_dim),
            "interaction_epochs": int(interaction_epochs),
            "interaction_lr": float(interaction_lr),
            "interaction_l2": float(interaction_l2),
            "interaction_ridge_alpha": float(interaction_ridge_alpha),
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
    blend_weight: float = 0.45,
    subject_weight: float = 0.68,
    condition_weight: float = 0.22,
    benchmark_weight: float = 0.10,
    item_adjustment_weight: float = 1.0,
    item_head_hidden_dim: int = 128,
    item_head_epochs: int = 50,
    item_head_lr: float = 1e-3,
    item_head_weight_decay: float = 1e-4,
    item_head_batch_size: int = 4096,
    interaction_latent_dim: int = 3,
    interaction_epochs: int = 10,
    interaction_lr: float = 3e-3,
    interaction_l2: float = 1e-3,
    interaction_batch_size: int = 8192,
    interaction_ridge_alpha: float = 300.0,
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
        item_head_hidden_dim,
        item_head_epochs,
        item_head_lr,
        item_head_weight_decay,
        item_head_batch_size,
        interaction_latent_dim,
        interaction_epochs,
        interaction_lr,
        interaction_l2,
        interaction_batch_size,
        interaction_ridge_alpha,
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
