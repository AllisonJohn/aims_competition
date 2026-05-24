from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("aims-label-ensemble-validation")

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


CANDIDATES = {
    "k3": "douglas_ridge_kfactor_k3_bayes_prior10-cap035",
    "ridge_mlp": "douglas_ridge_learned_skip_mlp_head",
    "ridge_coeffs": "douglas_ridge_learned_skip",
    "ridge_2pl": "douglas_ridge_2pl_learned_skip",
}


def _read_tree(root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc"):
            files[str(path.relative_to(root))] = path.read_bytes()
    return files


@app.function(gpu="H100", image=image, timeout=60 * 60 * 2)
def validate_remote(
    candidate_files: dict[str, dict[str, bytes]],
    validator_source: str,
    benchmarks: str,
    pairs_per_benchmark: int,
    k_labels: int,
    score_labeled: bool,
    prior_count: float,
    cap: float,
) -> dict:
    import importlib.util
    import math
    import sys
    import types
    from pathlib import Path

    from huggingface_hub import snapshot_download

    work_dir = Path("/tmp/label_validation_ensemble")
    candidates_dir = work_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for name, files in candidate_files.items():
        root = candidates_dir / name
        root.mkdir(parents=True, exist_ok=True)
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    model_ids = set()
    for root in candidates_dir.iterdir():
        models_path = root / "models.txt"
        if models_path.exists():
            for line in models_path.read_text(encoding="utf-8").splitlines():
                model_id = line.strip()
                if model_id:
                    model_ids.add(model_id)
    for model_id in sorted(model_ids):
        print(f"snapshot_download({model_id})", flush=True)
        snapshot_download(model_id)

    validator_path = work_dir / "validate_labeling.py"
    validator_path.write_text(validator_source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("validate_labeling_remote", validator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import validator")
    validator = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(work_dir))
    spec.loader.exec_module(validator)

    models = {}
    for name in candidate_files:
        model_path = candidates_dir / name / "model.py"
        model_spec = importlib.util.spec_from_file_location(f"ensemble_candidate_{name}", model_path)
        if model_spec is None or model_spec.loader is None:
            raise RuntimeError(f"Could not import {model_path}")
        module = importlib.util.module_from_spec(model_spec)
        old_path = list(sys.path)
        sys.path.insert(0, str(model_path.parent))
        try:
            model_spec.loader.exec_module(module)
        finally:
            sys.path[:] = old_path
        models[name] = module

    def clamp(value: float, lo: float = 0.001, hi: float = 0.999) -> float:
        return max(lo, min(hi, float(value)))

    def logit(p: float) -> float:
        p = clamp(p)
        return math.log(p / (1.0 - p))

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def bce(y: float, p: float) -> float:
        p = clamp(p)
        return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))

    def log_loss(labels: list[int], probs: list[float]) -> float:
        return sum(bce(y, p) for y, p in zip(labels, probs)) / max(1, len(labels))

    def brier(labels: list[int], probs: list[float]) -> float:
        return sum((float(y) - clamp(p)) ** 2 for y, p in zip(labels, probs)) / max(1, len(labels))

    def bayes_delta_from_values(labels: list[float], expected: list[float]) -> float:
        if not labels:
            return 0.0
        prior_rate = clamp(sum(expected) / len(expected))
        observed_rate = clamp(sum(labels) / len(labels))
        posterior_rate = clamp(
            (prior_rate * float(prior_count) + observed_rate * len(labels))
            / (float(prior_count) + len(labels))
        )
        delta = logit(posterior_rate) - logit(prior_rate)
        return max(-float(cap), min(float(cap), delta))

    def weighted_prob(preds: dict[str, float], weights: dict[str, float]) -> float:
        return clamp(sum(weights[name] * preds[name] for name in weights))

    def weighted_logit(preds: dict[str, float], weights: dict[str, float]) -> float:
        return clamp(sigmoid(sum(weights[name] * logit(preds[name]) for name in weights)))

    def normalize(weights: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, float(v)) for v in weights.values())
        if total <= 0:
            return {"k3": 1.0}
        return {name: max(0.0, float(value)) / total for name, value in weights.items()}

    def softmax_weights(losses: dict[str, float], temp: float) -> dict[str, float]:
        best = min(losses.values())
        raw = {name: math.exp(-temp * (loss - best)) for name, loss in losses.items()}
        return normalize(raw)

    def anchor_weights(raw: dict[str, float], anchor: float) -> dict[str, float]:
        out = {name: (1.0 - anchor) * value for name, value in raw.items()}
        out["k3"] = out.get("k3", 0.0) + anchor
        return normalize(out)

    benchmarks_list = [b.strip() for b in benchmarks.split(",") if b.strip()]
    rows = validator.load_rows(benchmarks_list, int(pairs_per_benchmark), 326, None)
    by_benchmark: dict[str, list[dict]] = {}
    for row in rows:
        by_benchmark.setdefault(row["benchmark"], []).append(row)

    policy_probs: dict[str, list[float]] = {}
    all_y: list[int] = []

    candidate_names = list(candidate_files)
    fixed_85 = normalize({"k3": 0.85, "ridge_mlp": 0.05, "ridge_coeffs": 0.05, "ridge_2pl": 0.05})
    fixed_70 = normalize({"k3": 0.70, "ridge_mlp": 0.10, "ridge_coeffs": 0.10, "ridge_2pl": 0.10})

    for benchmark, bench_rows in sorted(by_benchmark.items()):
        labeled = validator.acquire_labels(bench_rows, int(k_labels), None, models["k3"], "random")
        labeled_ids = {
            f"{ex['benchmark']}::{ex['subject_content']}::{ex['item_content']}::{ex['condition']}"
            for ex in labeled
        }
        eval_rows = []
        for row in bench_rows:
            row_id = f"{row['benchmark']}::{row['subject_content']}::{row['item_content']}::{row['condition']}"
            if score_labeled or row_id not in labeled_ids:
                eval_rows.append(row)

        all_fold_rows = list(eval_rows)
        seen_keys = {
            f"{row['benchmark']}::{row['subject_content']}::{row['item_content']}::{row['condition']}"
            for row in all_fold_rows
        }
        for row in labeled:
            row_key = f"{row['benchmark']}::{row['subject_content']}::{row['item_content']}::{row['condition']}"
            if row_key not in seen_keys:
                all_fold_rows.append(row)
                seen_keys.add(row_key)

        pred_cache: dict[str, dict[str, float]] = {}
        for row in all_fold_rows:
            row_key = f"{row['benchmark']}::{row['subject_content']}::{row['item_content']}::{row['condition']}"
            inp = validator.public_input(row)
            pred_cache[row_key] = {
                name: clamp(float(model.predict(inp, None)))
                for name, model in models.items()
            }

        def row_key(row: dict) -> str:
            return f"{row['benchmark']}::{row['subject_content']}::{row['item_content']}::{row['condition']}"

        def row_preds(row: dict) -> dict[str, float]:
            return pred_cache[row_key(row)]

        label_losses: dict[str, float] = {}
        for name in models:
            losses = [
                bce(float(example["label"]), row_preds(example)[name])
                for example in labeled
            ]
            label_losses[name] = sum(losses) / max(1, len(losses))

        soft2 = anchor_weights(softmax_weights(label_losses, 2.0), 0.70)
        soft5 = anchor_weights(softmax_weights(label_losses, 5.0), 0.70)
        soft10 = anchor_weights(softmax_weights(label_losses, 10.0), 0.70)
        soft5_strong_anchor = anchor_weights(softmax_weights(label_losses, 5.0), 0.85)

        alt_names = [name for name in candidate_names if name != "k3"]
        best_alt = min(alt_names, key=lambda name: label_losses[name])
        gated_003 = {"k3": 1.0}
        if label_losses[best_alt] + 0.03 < label_losses["k3"]:
            gated_003 = normalize({"k3": 0.80, best_alt: 0.20})
        gated_000 = {"k3": 1.0}
        if label_losses[best_alt] < label_losses["k3"]:
            gated_000 = normalize({"k3": 0.80, best_alt: 0.20})

        raw_policies = {
            "k3_raw_plus_bayes": ({"k3": 1.0}, "prob"),
            "fixed85_prob_plus_bayes": (fixed_85, "prob"),
            "fixed85_logit_plus_bayes": (fixed_85, "logit"),
            "fixed70_logit_plus_bayes": (fixed_70, "logit"),
            "soft2_anchor70_logit_plus_bayes": (soft2, "logit"),
            "soft5_anchor70_logit_plus_bayes": (soft5, "logit"),
            "soft10_anchor70_logit_plus_bayes": (soft10, "logit"),
            "soft5_anchor85_logit_plus_bayes": (soft5_strong_anchor, "logit"),
            "gated_margin003_logit_plus_bayes": (gated_003, "logit"),
            "gated_margin000_logit_plus_bayes": (gated_000, "logit"),
        }

        for name in candidate_names:
            raw_policies[f"{name}_alone_plus_bayes"] = ({name: 1.0}, "prob")

        policy_deltas = {}
        label_values = [clamp(float(example["label"]), 0.0, 1.0) for example in labeled]
        for policy, (weights, mode) in raw_policies.items():
            expected = []
            for example in labeled:
                preds = row_preds(example)
                if mode == "prob":
                    expected.append(weighted_prob(preds, weights))
                else:
                    expected.append(weighted_logit(preds, weights))
            policy_deltas[policy] = bayes_delta_from_values(label_values, expected)

        for row in eval_rows:
            all_y.append(int(row["label"]))
            preds = row_preds(row)

            for policy, (weights, mode) in raw_policies.items():
                if mode == "prob":
                    raw_p = weighted_prob(preds, weights)
                else:
                    raw_p = weighted_logit(preds, weights)
                policy_probs.setdefault(policy, []).append(
                    clamp(sigmoid(logit(raw_p) + policy_deltas[policy]))
                )

            # Built-in k3 should match k3_raw_plus_bayes; keep it as a sanity check.
            inp = validator.public_input(row)
            policy_probs.setdefault("k3_builtin", []).append(float(models["k3"].predict(inp, labeled)))

    results = []
    for policy, probs in policy_probs.items():
        results.append(
            {
                "policy": policy,
                "log_loss": log_loss(all_y, probs),
                "brier": brier(all_y, probs),
                "eval_n": len(all_y),
            }
        )
    results.sort(key=lambda row: row["log_loss"])
    print("top policies:", flush=True)
    for row in results:
        print(json.dumps(row, sort_keys=True), flush=True)
    return {"results": results, "top": results[:10]}


@app.local_entrypoint()
def main(
    benchmarks: str = "afrimedqa,agentdojo,ai2d_test,androidworld,bfcl,cybench,hle,livecodebench,matharena,mathvista_mini,mmbench_v11,mmlupro,mtbench,rewardbench,swebench,ultrafeedback",
    pairs_per_benchmark: int = 80,
    k_labels: int = 5,
    score_labeled: bool = True,
    prior_count: float = 10.0,
    cap: float = 0.35,
    json_out: str | None = "label_validation_sandbox/ensemble_validation.json",
) -> None:
    root = Path(__file__).resolve().parents[1]
    candidate_files = {}
    for name, relative in CANDIDATES.items():
        path = root / relative
        if not path.exists():
            raise SystemExit(f"Candidate path does not exist: {path}")
        candidate_files[name] = _read_tree(path)
    validator_source = (Path(__file__).resolve().parent / "validate_labeling.py").read_text(encoding="utf-8")
    summary = validate_remote.remote(
        candidate_files,
        validator_source,
        benchmarks,
        pairs_per_benchmark,
        k_labels,
        score_labeled,
        prior_count,
        cap,
    )
    if json_out:
        (root / json_out).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["top"], indent=2, sort_keys=True))
