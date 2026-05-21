from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("cs321m-final-text-features-composite-artifact")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "huggingface_hub",
        "hf_xet",
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


@app.function(image=image, cpu=8.0, memory=32768, timeout=60 * 60 * 4)
def train_remote(
    baseline_stats: dict,
    train_rows: int = 1_500_000,
    item_limit: int = 120_000,
    blend_weight: float = 0.30,
    difficulty_cap: float = 2.5,
    ridge_alphas: str = "100,300,1000",
    bag_seeds: str = "326,327,328,329,330",
) -> dict:
    import math
    import re

    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    repo_id = "aims-foundations/measurement-db"
    base_feature_dim = 73

    def parse_float_list(value: str) -> list[float]:
        return [float(part.strip()) for part in value.split(",") if part.strip()]

    def parse_int_list(value: str) -> list[int]:
        return [int(part.strip()) for part in value.split(",") if part.strip()]

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
        parts.append(df[["subject_name", "item_key", "benchmark_id", "item_content", "label"]])
    train_df = pd.concat(parts, ignore_index=True)
    if len(train_df) > train_rows:
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
    if len(item_targets) > item_limit:
        item_targets = item_targets.sample(n=item_limit, random_state=326)

    def text_feature_vector(text: object) -> list[float]:
        text = str(text or "")
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

        def bounded(value: float, scale: float) -> float:
            return min(float(value), scale) / scale

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
        assert len(features) == base_feature_dim
        return features

    def composite_features(base: np.ndarray) -> np.ndarray:
        pairwise = []
        for i in range(base.shape[1]):
            for j in range(i, base.shape[1]):
                pairwise.append((base[:, i] * base[:, j])[:, None])
        return np.hstack([base] + pairwise).astype(np.float32)

    def text_features(df: pd.DataFrame) -> np.ndarray:
        rows = []
        for text in df["item_content"]:
            rows.append(text_feature_vector(text))
        return np.asarray(rows, dtype=np.float32)

    scaler = StandardScaler()
    base_train = scaler.fit_transform(text_features(item_targets)).astype(np.float32)
    x_train = composite_features(base_train)
    y_train = item_targets["centered_difficulty"].to_numpy(dtype=np.float32)
    alphas = parse_float_list(ridge_alphas)
    seeds = parse_int_list(bag_seeds)
    if not alphas:
        alphas = [300.0]
    if not seeds:
        seeds = [326]

    ridge_heads = []
    n_examples = len(y_train)
    for head_index, seed in enumerate(seeds):
        alpha = alphas[head_index % len(alphas)]
        rng = np.random.default_rng(seed)
        sample_indices = rng.integers(0, n_examples, size=n_examples)
        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(x_train[sample_indices], y_train[sample_indices])
        ridge_heads.append(
            {
                "ridge_intercept": float(model.intercept_),
                "ridge_coef": [float(x) for x in model.coef_],
                "ridge_alpha": float(alpha),
                "seed": int(seed),
            }
        )

    return {
        "artifact": {
            "encoder_id": "handcrafted-text-features-composite-v1",
            "blend_weight": float(blend_weight),
            "difficulty_cap": float(difficulty_cap),
            "ridge_heads": ridge_heads,
            "scaler_mean": [float(x) for x in scaler.mean_],
            "scaler_scale": [float(x) if float(x) != 0.0 else 1.0 for x in scaler.scale_],
            "base_feature_dim": base_feature_dim,
            "feature_dim": int(x_train.shape[1]),
            "pairwise_interactions": True,
        },
        "summary": {
            "train_rows": int(len(train_df)),
            "rasch_subjects": int(len(subject_uniques)),
            "rasch_items": int(len(item_uniques)),
            "text_item_targets": int(len(item_targets)),
            "encoder": "handcrafted-text-features-composite-v1",
            "ridge_alphas": [float(alpha) for alpha in alphas],
            "bag_seeds": [int(seed) for seed in seeds],
            "ridge_heads": int(len(ridge_heads)),
            "blend_weight": float(blend_weight),
            "difficulty_cap": float(difficulty_cap),
            "base_feature_dim": base_feature_dim,
            "feature_dim": int(x_train.shape[1]),
            "pairwise_interactions": True,
        },
    }


@app.local_entrypoint()
def main(
    train_rows: int = 1_500_000,
    item_limit: int = 120_000,
    blend_weight: float = 0.30,
    difficulty_cap: float = 2.5,
    ridge_alphas: str = "100,300,1000",
    bag_seeds: str = "326,327,328,329,330",
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
        blend_weight,
        difficulty_cap,
        ridge_alphas,
        bag_seeds,
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
