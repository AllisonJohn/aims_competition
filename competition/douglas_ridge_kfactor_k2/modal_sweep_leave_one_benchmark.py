from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("cs321m-final-bge-kfactor-k2-lobo-sweep")

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


@app.function(gpu="H100", image=image, timeout=60 * 60 * 18)
def sweep_remote(
    train_rows: int = 1_500_000,
    item_limit: int = 120_000,
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    latent_dim: int = 2,
    irt_epochs: int = 5,
    irt_lrs_csv: str = "0.003",
    irt_l2s_csv: str = "0.001",
    irt_batch_size: int = 8192,
    ridge_alphas_csv: str = "30,100,300,1000,3000",
    calibrator_cs_csv: str = "0.1,0.3,1,3,10",
    logit_caps_csv: str = "3,4,5",
    blend_weights_csv: str = "0,0.1,0.2,0.3,0.45,0.6,0.8,1",
    prediction_modes_csv: str = "calibrated",
    heldout_benchmarks_csv: str = "all",
    max_folds: int = 0,
) -> dict:
    import math
    from collections import defaultdict

    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModel, AutoTokenizer

    repo_id = "aims-foundations/measurement-db"

    def parse_float_csv(text: str) -> list[float]:
        return [float(x.strip()) for x in str(text).split(",") if x.strip()]

    irt_lrs = parse_float_csv(irt_lrs_csv)
    irt_l2s = parse_float_csv(irt_l2s_csv)
    ridge_alphas = parse_float_csv(ridge_alphas_csv)
    calibrator_cs = parse_float_csv(calibrator_cs_csv)
    logit_caps = parse_float_csv(logit_caps_csv)
    blend_weights = parse_float_csv(blend_weights_csv)
    prediction_modes = [x.strip().lower() for x in prediction_modes_csv.split(",") if x.strip()]

    def clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
        return max(lo, min(hi, float(value)))

    def logit(p: float) -> float:
        p = clamp(float(p))
        return math.log(p / (1.0 - p))

    def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
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

    def smooth(rate_count: list[float] | None, alpha: float, global_rate: float) -> float:
        if not rate_count:
            return global_rate
        rate, count = float(rate_count[0]), float(rate_count[1])
        return (rate * count + global_rate * alpha) / (count + alpha)

    def rate_table(df: pd.DataFrame, column: str) -> dict[str, list[float]]:
        grouped = df.groupby(column)["label"].agg(["mean", "count"])
        return {
            str(index).strip().lower(): [float(row["mean"]), float(row["count"])]
            for index, row in grouped.iterrows()
        }

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

    class Embedder:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_id)
            self.model = AutoModel.from_pretrained(encoder_id).to("cuda")
            self.model.eval()

        @torch.no_grad()
        def encode(self, texts: list[str], batch_size: int = 96) -> np.ndarray:
            if not texts:
                return np.zeros((0, 1024), dtype=np.float32)
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
                    print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)} items", flush=True)
            return np.vstack(chunks)

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
    all_df = pd.concat(parts, ignore_index=True)
    if train_rows > 0 and len(all_df) > train_rows:
        all_df = all_df.sample(n=train_rows, random_state=326)

    all_benchmarks = [filename.removesuffix(".parquet") for filename in RESPONSE_FILES]
    if heldout_benchmarks_csv.strip().lower() == "all":
        heldout_benchmarks = all_benchmarks
    else:
        requested = {x.strip().lower() for x in heldout_benchmarks_csv.split(",") if x.strip()}
        heldout_benchmarks = [benchmark for benchmark in all_benchmarks if benchmark.lower() in requested]
    if max_folds > 0:
        heldout_benchmarks = heldout_benchmarks[:max_folds]

    embedder = Embedder()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subject_alpha = 250.0
    condition_alpha = 2000.0
    benchmark_alpha = 5000.0
    aggregates: dict[str, dict] = defaultdict(
        lambda: {
            "n": 0,
            "sum_log_loss": 0.0,
            "fold_negative_log_losses": [],
            "fold_auc_rocs": [],
            "folds": [],
        }
    )
    fold_summaries = []

    for heldout in heldout_benchmarks:
        valid_mask = all_df["benchmark_id"].astype(str).str.lower() == heldout.lower()
        train_df = all_df.loc[~valid_mask].copy()
        valid_df = all_df.loc[valid_mask].copy()
        if train_df.empty or valid_df.empty:
            continue

        print(
            f"fold={heldout} train_rows={len(train_df)} valid_rows={len(valid_df)} "
            f"grid={len(irt_lrs) * len(irt_l2s) * len(ridge_alphas) * len(calibrator_cs) * len(logit_caps)} "
            f"modes={prediction_modes} blend_weights={blend_weights}",
            flush=True,
        )

        subject_codes, subject_uniques = pd.factorize(train_df["subject_name"], sort=True)
        item_codes, item_uniques = pd.factorize(train_df["item_key"], sort=True)
        y = train_df["label"].to_numpy(dtype=np.float32)

        subject_index = {str(name).strip().lower(): int(i) for i, name in enumerate(subject_uniques)}
        train_item_meta = train_df[["item_key", "benchmark_id", "item_content"]].drop_duplicates("item_key")
        valid_item_meta = valid_df[["item_key", "benchmark_id", "item_content"]].drop_duplicates("item_key")

        train_item_texts = train_item_meta["item_content"].fillna("").tolist()
        valid_item_texts = valid_item_meta["item_content"].fillna("").tolist()
        train_embeddings = embedder.encode(train_item_texts)
        valid_embeddings = embedder.encode(valid_item_texts)
        numeric_scaler = StandardScaler()
        train_numeric = numeric_scaler.fit_transform(numeric_features(train_item_meta))
        valid_numeric = numeric_scaler.transform(numeric_features(valid_item_meta))
        x_train_items_full = np.hstack([train_embeddings, train_numeric]).astype(np.float32)
        x_valid_items_full = np.hstack([valid_embeddings, valid_numeric]).astype(np.float32)

        global_rate = float(train_df["label"].mean())
        subject_rates = rate_table(train_df, "subject_name")
        condition_rates = rate_table(train_df, "condition")
        benchmark_rates = rate_table(train_df, "benchmark_id")

        def stats_features(df: pd.DataFrame, factor_logits: np.ndarray) -> np.ndarray:
            rows = []
            for row_index, row in enumerate(df.itertuples(index=False)):
                subject_rate = smooth(subject_rates.get(str(row.subject_name).strip().lower()), subject_alpha, global_rate)
                condition_rate = smooth(condition_rates.get(str(row.condition).strip().lower()), condition_alpha, global_rate)
                benchmark_rate = smooth(benchmark_rates.get(str(row.benchmark_id).strip().lower()), benchmark_alpha, global_rate)
                subject_logit = logit(subject_rate)
                condition_logit = logit(condition_rate)
                benchmark_logit = logit(benchmark_rate)
                factor_logit = float(factor_logits[row_index])
                adjustment = item_adjustment(row.item_content)
                rows.append(
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
            return np.asarray(rows, dtype=np.float32)

        def base_logits_from_features(features: np.ndarray) -> np.ndarray:
            return (
                0.68 * features[:, 0]
                + 0.22 * features[:, 1]
                + 0.10 * features[:, 2]
                + features[:, 3]
            ).astype(np.float32)

        for irt_lr in irt_lrs:
            for irt_l2 in irt_l2s:
                dataset = TensorDataset(
                    torch.as_tensor(subject_codes, dtype=torch.long),
                    torch.as_tensor(item_codes, dtype=torch.long),
                    torch.as_tensor(y, dtype=torch.float32),
                )
                loader = DataLoader(
                    dataset,
                    batch_size=irt_batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=device == "cuda",
                )
                irt = KFactorIRT(len(subject_uniques), len(item_uniques), latent_dim).to(device)
                optimizer = torch.optim.AdamW(irt.parameters(), lr=irt_lr, weight_decay=0.0)
                criterion = torch.nn.BCEWithLogitsLoss()
                print(
                    f"fold={heldout} fitting_irt K={latent_dim} epochs={irt_epochs} lr={irt_lr} l2={irt_l2}",
                    flush=True,
                )
                for epoch in range(1, irt_epochs + 1):
                    running_loss = 0.0
                    seen = 0
                    for subjects_batch, items_batch, labels_batch in loader:
                        subjects_batch = subjects_batch.to(device, non_blocking=True)
                        items_batch = items_batch.to(device, non_blocking=True)
                        labels_batch = labels_batch.to(device, non_blocking=True)
                        logits, u_batch, v_batch = irt(subjects_batch, items_batch)
                        bce = criterion(logits, labels_batch)
                        l2 = u_batch.pow(2).mean() + v_batch.pow(2).mean()
                        loss = bce + float(irt_l2) * l2
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        batch_size = int(labels_batch.numel())
                        running_loss += float(bce.detach().cpu()) * batch_size
                        seen += batch_size
                    print(f"fold={heldout} irt_epoch={epoch} train_bce={running_loss / max(seen, 1):.6f}", flush=True)

                with torch.no_grad():
                    subject_factor_matrix = irt.subject_factors.weight.detach().cpu().numpy().astype(np.float32)
                    item_factor_matrix = irt.item_factors.weight.detach().cpu().numpy().astype(np.float32)
                    item_bias = irt.item_bias.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)

                item_targets = pd.DataFrame({"item_key": item_uniques, "item_bias": item_bias})
                for k in range(latent_dim):
                    item_targets[f"factor_{k}"] = item_factor_matrix[:, k]
                item_targets = item_targets.merge(train_item_meta, on="item_key", how="left")
                if item_limit > 0 and len(item_targets) > item_limit:
                    item_targets = item_targets.sample(n=item_limit, random_state=326)
                item_target_positions = train_item_meta.reset_index(drop=True).reset_index().set_index("item_key")["index"]
                target_positions = item_targets["item_key"].map(item_target_positions).to_numpy(dtype=np.int64)
                x_train_items = x_train_items_full[target_positions]

                train_teacher_logits = item_bias[item_codes] + np.sum(
                    subject_factor_matrix[subject_codes] * item_factor_matrix[item_codes],
                    axis=1,
                )

                subject_factor_lookup = {
                    str(name).strip().lower(): subject_factor_matrix[index]
                    for index, name in enumerate(subject_uniques)
                }
                global_subject_factor = subject_factor_matrix.mean(axis=0)
                valid_subject_matrix = np.vstack(
                    [
                        subject_factor_lookup.get(str(name).strip().lower(), global_subject_factor)
                        for name in valid_df["subject_name"]
                    ]
                ).astype(np.float32)

                valid_item_position = {
                    str(row.item_key): int(index)
                    for index, row in enumerate(valid_item_meta.itertuples(index=False))
                }
                valid_positions = valid_df["item_key"].map(valid_item_position).to_numpy(dtype=np.int64)

                target_names = [f"factor_{k}" for k in range(latent_dim)] + ["item_bias"]
                for ridge_alpha in ridge_alphas:
                    valid_predictions_by_target = {}
                    for target_name in target_names:
                        target = item_targets[target_name].to_numpy(dtype=np.float32)
                        target_mean = float(target.mean())
                        target_std = float(target.std())
                        if target_std < 1e-6:
                            target_std = 1.0
                        target_normalized = ((target - target_mean) / target_std).astype(np.float32)
                        ridge = Ridge(alpha=ridge_alpha, random_state=326)
                        ridge.fit(x_train_items, target_normalized)
                        pred = ridge.predict(x_valid_items_full).astype(np.float32)
                        valid_predictions_by_target[target_name] = target_mean + target_std * pred

                    valid_factor_matrix = np.vstack(
                        [valid_predictions_by_target[f"factor_{k}"] for k in range(latent_dim)]
                    ).T.astype(np.float32)
                    valid_item_bias = valid_predictions_by_target["item_bias"].astype(np.float32)
                    valid_factor_logits_raw = valid_item_bias[valid_positions] + np.sum(
                        valid_subject_matrix * valid_factor_matrix[valid_positions],
                        axis=1,
                    )

                    for logit_cap in logit_caps:
                        train_factor_logits = np.clip(train_teacher_logits, -logit_cap, logit_cap)
                        valid_factor_logits = np.clip(valid_factor_logits_raw, -logit_cap, logit_cap)
                        calibration_x = stats_features(train_df, train_factor_logits)
                        validation_x = stats_features(valid_df, valid_factor_logits)
                        validation_base_logits = base_logits_from_features(validation_x)
                        calibration_y = train_df["label"].to_numpy(dtype=np.int64)
                        labels = valid_df["label"].to_numpy(dtype=np.int64)
                        calibration_scaler = StandardScaler()
                        calibration_x_scaled = calibration_scaler.fit_transform(calibration_x).astype(np.float32)
                        validation_x_scaled = calibration_scaler.transform(validation_x).astype(np.float32)

                        for calibrator_c in calibrator_cs:
                            setting = {
                                "latent_dim": int(latent_dim),
                                "irt_epochs": int(irt_epochs),
                                "irt_lr": float(irt_lr),
                                "irt_l2": float(irt_l2),
                                "ridge_alpha": float(ridge_alpha),
                                "calibrator_c": float(calibrator_c),
                                "logit_cap": float(logit_cap),
                            }
                            setting_key = json.dumps(setting, sort_keys=True, separators=(",", ":"))
                            calibrator = LogisticRegression(
                                C=float(calibrator_c),
                                solver="lbfgs",
                                max_iter=1000,
                                random_state=326,
                            )
                            calibrator.fit(calibration_x_scaled, calibration_y)
                            calibrated_probabilities = np.clip(
                                calibrator.predict_proba(validation_x_scaled)[:, 1],
                                0.001,
                                0.999,
                            )
                            calibrated_logits = np.asarray(
                                [logit(probability) for probability in calibrated_probabilities],
                                dtype=np.float32,
                            )
                            candidates: list[tuple[str, float, np.ndarray]] = []
                            if "calibrated" in prediction_modes:
                                candidates.append(("calibrated", 1.0, calibrated_probabilities))
                            if "calibrated_blend" in prediction_modes:
                                for blend_weight in blend_weights:
                                    blended_logits = (
                                        (1.0 - float(blend_weight)) * validation_base_logits
                                        + float(blend_weight) * calibrated_logits
                                    )
                                    candidates.append(("calibrated_blend", float(blend_weight), sigmoid(blended_logits)))
                            if "raw_factor_blend" in prediction_modes:
                                for blend_weight in blend_weights:
                                    blended_logits = (
                                        (1.0 - float(blend_weight)) * validation_base_logits
                                        + float(blend_weight) * valid_factor_logits
                                    )
                                    candidates.append(("raw_factor_blend", float(blend_weight), sigmoid(blended_logits)))

                            for prediction_mode, blend_weight, probabilities in candidates:
                                setting_with_output = {
                                    **setting,
                                    "prediction_mode": prediction_mode,
                                    "blend_weight": float(blend_weight),
                                }
                                setting_key = json.dumps(setting_with_output, sort_keys=True, separators=(",", ":"))
                                probabilities = np.clip(probabilities, 0.001, 0.999)
                                loss = float(log_loss(labels, probabilities, labels=[0, 1]))
                                neg_log_loss = -loss
                                try:
                                    auc = float(roc_auc_score(labels, probabilities))
                                except ValueError:
                                    auc = None
                                aggregate = aggregates[setting_key]
                                aggregate["setting"] = setting_with_output
                                aggregate["n"] += int(len(labels))
                                aggregate["sum_log_loss"] += loss * int(len(labels))
                                aggregate["fold_negative_log_losses"].append(neg_log_loss)
                                if auc is not None:
                                    aggregate["fold_auc_rocs"].append(auc)
                                aggregate["folds"].append(
                                    {
                                        "heldout_benchmark": heldout,
                                        "rows": int(len(labels)),
                                        "negative_log_loss": neg_log_loss,
                                        "auc_roc": auc,
                                    }
                                )

        fold_summaries.append(
            {
                "heldout_benchmark": heldout,
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(valid_df)),
            }
        )

    results = []
    for aggregate in aggregates.values():
        n = max(int(aggregate["n"]), 1)
        results.append(
            {
                "setting": aggregate["setting"],
                "validation_rows": int(aggregate["n"]),
                "weighted_negative_log_loss": -float(aggregate["sum_log_loss"] / n),
                "mean_fold_negative_log_loss": float(np.mean(aggregate["fold_negative_log_losses"])),
                "mean_fold_auc_roc": (
                    float(np.mean(aggregate["fold_auc_rocs"]))
                    if aggregate["fold_auc_rocs"]
                    else None
                ),
                "folds": aggregate["folds"],
            }
        )
    results.sort(key=lambda row: row["weighted_negative_log_loss"], reverse=True)
    top_3 = results[:3]
    summary = {
        "split": "leave-one-benchmark-out",
        "leakage_policy": "all subject/condition/benchmark rates and IRT factors are fit on training benchmarks only; held-out benchmark rate falls back to the training global rate",
        "heldout_benchmarks": heldout_benchmarks,
        "fold_summaries": fold_summaries,
        "train_rows_arg": int(train_rows),
        "item_limit": int(item_limit),
        "encoder": encoder_id,
        "grid": {
            "latent_dim": [int(latent_dim)],
            "irt_epochs": [int(irt_epochs)],
            "irt_lrs": irt_lrs,
            "irt_l2s": irt_l2s,
            "ridge_alphas": ridge_alphas,
            "calibrator_cs": calibrator_cs,
            "logit_caps": logit_caps,
            "blend_weights": blend_weights,
            "prediction_modes": prediction_modes,
        },
        "sweep_count": int(len(results)),
        "top_3": top_3,
        "all_results": results,
    }
    print(json.dumps({"top_3": top_3}, indent=2, sort_keys=True), flush=True)
    return summary


@app.local_entrypoint()
def main(
    train_rows: int = 1_500_000,
    item_limit: int = 120_000,
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    latent_dim: int = 2,
    irt_epochs: int = 5,
    irt_lrs: str = "0.003",
    irt_l2s: str = "0.001",
    irt_batch_size: int = 8192,
    ridge_alphas: str = "30,100,300,1000,3000",
    calibrator_cs: str = "0.1,0.3,1,3,10",
    logit_caps: str = "3,4,5",
    blend_weights: str = "0,0.1,0.2,0.3,0.45,0.6,0.8,1",
    prediction_modes: str = "calibrated",
    heldout_benchmarks: str = "all",
    max_folds: int = 0,
) -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = sweep_remote.remote(
        train_rows,
        item_limit,
        encoder_id,
        latent_dim,
        irt_epochs,
        irt_lrs,
        irt_l2s,
        irt_batch_size,
        ridge_alphas,
        calibrator_cs,
        logit_caps,
        blend_weights,
        prediction_modes,
        heldout_benchmarks,
        max_folds,
    )
    summary_path = out_dir / "leave_one_benchmark_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"top_3": summary["top_3"]}, indent=2, sort_keys=True))
    print(f"Wrote {summary_path}")
