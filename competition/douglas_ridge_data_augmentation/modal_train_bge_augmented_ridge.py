from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("cs321m-final-bge-augmented-ridge-artifact")

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


@app.function(gpu="H100", image=image, timeout=60 * 60 * 5)
def train_remote(
    baseline_stats: dict,
    train_rows: int = 1_500_000,
    item_limit: int = 120_000,
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    ridge_alpha: float = 300.0,
    blend_weight: float = 0.45,
    difficulty_cap: float = 2.5,
    augmented_pca_dim: int = 64,
    augmented_scale: float = 0.5,
) -> dict:
    import math
    import re

    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer

    repo_id = "aims-foundations/measurement-db"

    def clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
        return max(lo, min(hi, float(value)))

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

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

    def bounded(value: float, scale: float) -> float:
        return min(float(value), scale) / scale

    def text_features(item_content: object) -> list[float]:
        text = str(item_content or "")
        lower = text.lower()
        words = re.findall(r"[A-Za-z0-9_]+", lower)
        sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
        unique_words = set(words)
        char_count = len(text)
        word_count = len(words)

        def count_any(values: tuple[str, ...]) -> int:
            return sum(lower.count(value) for value in values)

        def has_any(values: tuple[str, ...]) -> float:
            return float(any(value in lower for value in values))

        math_symbols = ("∫", "∑", "∂", "√", "≤", "≥", "∈", "\\frac", "\\sum", "\\int")
        operator_count = sum(text.count(symbol) for symbol in "+-*/=<>")
        variable_pattern_count = len(re.findall(r"\b[a-z]\d*\b", lower))
        option_count = len(re.findall(r"\b[a-d][\).]", lower))
        long_words = [word for word in words if len(word) >= 12]
        avg_word_len = (sum(len(word) for word in words) / word_count) if word_count else 0.0
        avg_sentence_words = word_count / max(1, len(sentences))
        upper_count = sum(1 for ch in text if ch.isupper())

        features = [
            bounded(char_count, 5000),
            math.log1p(char_count) / math.log1p(10000),
            bounded(word_count, 1000),
            math.log1p(word_count) / math.log1p(2000),
            bounded(avg_word_len, 20),
            bounded(text.count("\n"), 80),
            bounded(len([p for p in text.split("\n\n") if p.strip()]), 30),
            bounded(len(sentences), 100),
            bounded(text.count("?"), 20),
            bounded(text.count("!"), 20),
            bounded(sum(ch.isdigit() for ch in text), 200),
            upper_count / max(1, char_count),
            bounded(text.count(","), 80),
            bounded(text.count("."), 80),
            bounded(text.count(":"), 60),
            bounded(text.count(";"), 40),
            bounded(text.count('"') + text.count("'"), 80),
            bounded(text.count("(") + text.count(")"), 80),
            bounded(text.count("[") + text.count("]"), 50),
            bounded(text.count("{") + text.count("}"), 50),
            bounded(text.count("/") + text.count("\\"), 80),
            bounded(text.count("`"), 80),
            bounded(text.count("$"), 80),
            bounded(text.count("%"), 40),
            bounded(text.count("="), 80),
            bounded(text.count("<") + text.count(">"), 80),
            float(any(symbol in text or symbol in lower for symbol in math_symbols)),
            has_any(("prove", "proof", "derive", "justify", "counterexample", "reason")),
            has_any(("calculate", "compute", "solve", "evaluate", "simplify")),
            has_any(("theorem", "lemma", "corollary", "conjecture")),
            has_any(("triangle", "circle", "angle", "geometry", "polygon")),
            has_any(("probability", "expected", "random", "distribution", "variance")),
            has_any(("def ", "class ", "import ", "function", "algorithm", "pseudocode")),
            has_any(("bug", "debug", "error", "exception", "runtime", "traceback")),
            has_any(("sql", "database", "query", "schema", "table")),
            has_any(("bash", "shell", "terminal", "git ", "command line")),
            has_any(("api", "http", "request", "server", "endpoint")),
            has_any(("security", "attack", "injection", "malicious", "vulnerability")),
            has_any(("tool", "agent", "browser", "environment", "action")),
            has_any(("patient", "diagnosis", "treatment", "clinical", "medical")),
            has_any(("biology", "cell", "protein", "gene", "organism")),
            has_any(("chemistry", "molecule", "reaction", "compound", "acid")),
            has_any(("physics", "force", "energy", "velocity", "quantum")),
            has_any(("image", "figure", "diagram", "visual", "picture")),
            has_any(("chart", "graph", "plot", "table", "axis")),
            has_any(("a)", "b)", "c)", "d)", "multiple choice", "choose the best")),
            has_any(("preference", "ranking", "better response", "judge", "rubric")),
            has_any(("safe", "unsafe", "refuse", "harmful", "policy")),
            has_any(("reward", "feedback", "helpfulness", "honesty", "truthfulness")),
            has_any(("repository", "issue", "pull request", "patch", "test file")),
            has_any(("instruction", "follow", "constraint", "must", "should")),
            float("```" in text or "`" in text),
            float(bool(re.search(r"(^|\n)\s*[-*•]\s+", text))),
            float(bool(re.search(r"(^|\n)\s*\d+[\).]\s+", text))),
            float("```" in text),
            float("$" in text or "\\(" in text or "\\[" in text),
            float("http://" in lower or "https://" in lower or "www." in lower),
            float(any(marker in lower for marker in ("{", "}", "<xml", "</", "json"))),
            has_any(("traceback", "stack trace", "line ", "syntaxerror", "typeerror")),
            float(bool(re.search(r"[/\\][\w.-]+[/\\]", text))),
            float(bool(re.search(r"\w+\([^)]*\)", text))),
            float(char_count > 2500),
            float(count_any(("part a", "part b", "subproblem", "step 1", "step 2")) > 0),
            len(unique_words) / max(1, word_count),
            len(long_words) / max(1, word_count),
            bounded(max([len(word) for word in words], default=0), 50),
            bounded(avg_sentence_words, 80),
            sum(ch.isdigit() for ch in text) / max(1, char_count),
            bounded(operator_count, 100),
            bounded(variable_pattern_count, 100),
            bounded(option_count, 10),
            text.count("\n") / max(1, char_count),
            bounded(count_any(("because", "therefore", "however", "although", "unless")), 30),
        ]
        assert len(features) == 73
        return features

    continuous_indices = list(range(0, 26)) + list(range(62, 73))
    complexity_indices = [0, 1, 2, 3, 7, 8, 10, 62, 63, 66, 67, 68, 69, 70, 72]
    domain_indices = list(range(26, 51))
    math_indices = [26, 27, 28, 29, 30, 31, 54]
    code_indices = [32, 33, 34, 35, 36, 49, 51, 53, 56, 57, 58, 59, 60]
    visual_indices = [43, 44]
    choice_indices = [45, 70]
    reasoning_indices = [27, 28, 29, 50, 63, 72]
    format_indices = list(range(51, 61))

    def unique_pairs(left: list[int], right: list[int]) -> list[tuple[int, int]]:
        pairs = set()
        for i in left:
            for j in right:
                if i != j:
                    pairs.add((min(i, j), max(i, j)))
        return sorted(pairs)

    structured_pairs = sorted(
        set(
            unique_pairs(complexity_indices, domain_indices)
            + unique_pairs(math_indices, reasoning_indices + [0, 1, 2, 3, 10])
            + unique_pairs(code_indices, format_indices + [0, 1, 2, 3, 10])
            + unique_pairs(visual_indices, format_indices + [0, 1, 2, 3])
            + unique_pairs(choice_indices, [0, 1, 2, 3, 7, 8, 69])
            + unique_pairs([37, 46, 47, 48, 50], [0, 1, 2, 3, 7, 38, 50])
        )
    )

    def expanded_features(raw: np.ndarray, scaled: np.ndarray) -> np.ndarray:
        transformed = []
        for index in continuous_indices:
            raw_column = np.clip(raw[:, index : index + 1], 0.0, None)
            scaled_column = scaled[:, index : index + 1]
            transformed.append(scaled_column * scaled_column)
            transformed.append(np.sqrt(raw_column))
            transformed.append(np.log1p(10.0 * raw_column) / np.log1p(10.0))
        interactions = [(scaled[:, i] * scaled[:, j])[:, None] for i, j in structured_pairs]
        return np.hstack([scaled] + transformed + interactions).astype(np.float32)

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

    print(f"Encoding {len(item_targets)} item targets with {encoder_id}", flush=True)
    embedder = Embedder()
    embeddings = embedder.encode(item_targets["item_content"].fillna("").tolist())

    raw_text = np.asarray(
        [text_features(text) for text in item_targets["item_content"].fillna("")],
        dtype=np.float32,
    )
    base_scaler = StandardScaler()
    base_scaled = base_scaler.fit_transform(raw_text).astype(np.float32)
    structured_raw = expanded_features(raw_text, base_scaled)
    structured_scaler = StandardScaler()
    structured_scaled = structured_scaler.fit_transform(structured_raw).astype(np.float32)

    pca = None
    if augmented_pca_dim > 0 and augmented_pca_dim < structured_scaled.shape[1]:
        pca = PCA(n_components=augmented_pca_dim, svd_solver="randomized", random_state=326, whiten=True)
        augmented = pca.fit_transform(structured_scaled).astype(np.float32)
    else:
        augmented = structured_scaled
    augmented = (float(augmented_scale) * augmented).astype(np.float32)

    x_train = np.hstack([embeddings, augmented]).astype(np.float32)
    y_train = item_targets["centered_difficulty"].to_numpy(dtype=np.float32)
    model = Ridge(alpha=ridge_alpha, random_state=326)
    model.fit(x_train, y_train)
    item_targets["difficulty_hat"] = model.predict(x_train).astype(np.float32)

    difficulty_by_item = dict(zip(item_targets["item_key"], item_targets["difficulty_hat"]))
    train_df["difficulty_hat"] = train_df["item_key"].map(difficulty_by_item).fillna(0.0).astype(float)
    global_rate = float(train_df["label"].mean())
    subject_rates = rate_table(train_df, "subject_name")
    condition_rates = rate_table(train_df, "condition")
    benchmark_rates = rate_table(train_df, "benchmark_id")
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
        difficulty = max(-difficulty_cap, min(difficulty_cap, float(row.difficulty_hat)))
        calibration_rows.append(
            [
                subject_logit,
                condition_logit,
                benchmark_logit,
                adjustment,
                difficulty,
                subject_logit * difficulty,
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

    return {
        "artifact": {
            "encoder_id": encoder_id,
            "blend_weight": float(blend_weight),
            "difficulty_cap": float(difficulty_cap),
            "calibrator": {
                "feature_names": [
                    "subject_logit",
                    "condition_logit",
                    "benchmark_logit",
                    "item_adjustment",
                    "difficulty",
                    "subject_x_difficulty",
                    "condition_x_difficulty",
                    "benchmark_x_difficulty",
                    "abs_difficulty",
                ],
                "intercept": float(calibrator.intercept_[0]),
                "coef": [float(x) for x in calibrator.coef_[0]],
                "scaler_mean": [float(x) for x in calibration_scaler.mean_],
                "scaler_scale": [float(x) if float(x) != 0.0 else 1.0 for x in calibration_scaler.scale_],
                "c": 1.0,
            },
            "ridge_intercept": float(model.intercept_),
            "ridge_coef": [float(x) for x in model.coef_],
            "base_feature_mean": [float(x) for x in base_scaler.mean_],
            "base_feature_scale": [float(x) if float(x) != 0.0 else 1.0 for x in base_scaler.scale_],
            "structured_feature_mean": [float(x) for x in structured_scaler.mean_],
            "structured_feature_scale": [
                float(x) if float(x) != 0.0 else 1.0 for x in structured_scaler.scale_
            ],
            "pca_mean": [] if pca is None else [float(x) for x in pca.mean_],
            "pca_components": [] if pca is None else [[float(x) for x in row] for row in pca.components_],
            "pca_explained_variance": [] if pca is None else [float(x) for x in pca.explained_variance_],
            "pca_whiten": bool(pca is not None),
            "augmented_scale": float(augmented_scale),
            "base_feature_dim": 73,
            "structured_feature_dim": int(structured_raw.shape[1]),
            "augmented_dim": int(augmented.shape[1]),
            "embedding_dim": int(embeddings.shape[1]),
            "max_length": 256,
        },
        "summary": {
            "train_rows": int(len(train_df)),
            "rasch_subjects": int(len(subject_uniques)),
            "rasch_items": int(len(item_uniques)),
            "text_item_targets": int(len(item_targets)),
            "encoder": encoder_id,
            "ridge_alpha": float(ridge_alpha),
            "blend_weight": float(blend_weight),
            "difficulty_cap": float(difficulty_cap),
            "calibrator_features": int(calibration_x.shape[1]),
            "base_feature_dim": 73,
            "structured_feature_dim": int(structured_raw.shape[1]),
            "augmented_dim": int(augmented.shape[1]),
            "augmented_pca_dim": int(augmented_pca_dim),
            "augmented_scale": float(augmented_scale),
        },
    }


@app.local_entrypoint()
def main(
    train_rows: int = 1_500_000,
    item_limit: int = 120_000,
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    ridge_alpha: float = 300.0,
    blend_weight: float = 0.45,
    difficulty_cap: float = 2.5,
    augmented_pca_dim: int = 64,
    augmented_scale: float = 0.5,
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
        ridge_alpha,
        blend_weight,
        difficulty_cap,
        augmented_pca_dim,
        augmented_scale,
    )
    (out_dir / "bge_augmented_ridge_artifact.json").write_text(
        json.dumps(payload["artifact"], separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "bge_augmented_ridge_summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"Wrote {out_dir / 'bge_augmented_ridge_artifact.json'}")
