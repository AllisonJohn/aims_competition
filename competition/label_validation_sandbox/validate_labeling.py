from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any


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


def clamp(value: float, lo: float = 1e-6, hi: float = 1.0 - 1e-6) -> float:
    return max(lo, min(hi, float(value)))


def binary_label(benchmark: str, response: float) -> int:
    if benchmark == "mtbench":
        y = float(response) / 10.0
    elif benchmark == "ultrafeedback":
        y = (float(response) - 1.0) / 4.0
    else:
        y = float(response)
    return int(clamp(y, 0.0, 1.0) >= 0.5)


def render_subject_content(subject: dict[str, Any], fallback_subject_id: str) -> str:
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


def stable_uniform(*parts: object) -> float:
    text = "\x1f".join(str(part or "") for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def import_module(path: Path, name_hint: str) -> ModuleType:
    module_name = f"_label_sandbox_{name_hint}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def resolve_submission(args: argparse.Namespace) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if args.submission_dir:
        return Path(args.submission_dir).resolve(), None
    if not args.submission_zip:
        raise SystemExit("Provide --submission-dir or --submission-zip")
    tmp = tempfile.TemporaryDirectory(prefix="label_sandbox_submission_")
    with zipfile.ZipFile(args.submission_zip) as zf:
        zf.extractall(tmp.name)
    return Path(tmp.name).resolve(), tmp


def load_rows(
    benchmarks: list[str],
    pairs_per_benchmark: int,
    seed: int,
    cache_dir: str | None,
) -> list[dict[str, Any]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo_id = "aims-foundations/measurement-db"
    subjects_path = hf_hub_download(repo_id, "subjects.parquet", repo_type="dataset", cache_dir=cache_dir)
    items_path = hf_hub_download(repo_id, "items.parquet", repo_type="dataset", cache_dir=cache_dir)
    subjects = pd.read_parquet(subjects_path)
    items = pd.read_parquet(items_path, columns=["item_id", "content"])
    subject_lookup = {
        row["subject_id"]: render_subject_content(row.to_dict(), row["subject_id"])
        for _, row in subjects.iterrows()
    }
    item_lookup = dict(zip(items["item_id"], items["content"]))

    rows: list[dict[str, Any]] = []
    wanted = set(benchmarks)
    files = [f"{bench}.parquet" for bench in benchmarks] if benchmarks else RESPONSE_FILES
    for filename in files:
        benchmark = filename.removesuffix(".parquet")
        if wanted and benchmark not in wanted:
            continue
        path = hf_hub_download(repo_id, filename, repo_type="dataset", cache_dir=cache_dir)
        df = pd.read_parquet(
            path,
            columns=["subject_id", "item_id", "benchmark_id", "test_condition", "response"],
        )
        if pairs_per_benchmark > 0 and len(df) > pairs_per_benchmark:
            df = df.sample(n=pairs_per_benchmark, random_state=seed)
        for row in df.itertuples(index=False):
            subject_content = subject_lookup.get(row.subject_id, f"Name: {row.subject_id}")
            item_content = item_lookup.get(row.item_id, "")
            bench = str(row.benchmark_id)
            rows.append(
                {
                    "benchmark": bench,
                    "condition": str(row.test_condition or "none"),
                    "subject_content": subject_content,
                    "item_content": item_content,
                    "label": binary_label(bench, row.response),
                    "_row_id": f"{bench}::{row.subject_id}::{row.item_id}::{row.test_condition}",
                }
            )
    return rows


def public_input(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": row["benchmark"],
        "condition": row["condition"],
        "subject_content": row["subject_content"],
        "item_content": row["item_content"],
    }


def labeled_input(row: dict[str, Any]) -> dict[str, Any]:
    out = public_input(row)
    out["label"] = int(row["label"])
    return out


def _det3(matrix: list[list[float]]) -> float:
    a = matrix
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def _d_optimal_labels(
    rows: list[dict[str, Any]],
    k_labels: int,
    model: ModuleType,
) -> list[dict[str, Any]]:
    if not hasattr(model, "_raw_prediction_parts"):
        return []
    fisher = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    candidates = []
    for row in rows:
        try:
            p, item_factors, _ = model._raw_prediction_parts(public_input(row))
        except Exception:
            continue
        if not item_factors or len(item_factors) != 3:
            continue
        info = clamp(p) * (1.0 - clamp(p))
        candidates.append((row, [float(x) for x in item_factors], info))
    selected = []
    for _ in range(min(k_labels, len(candidates))):
        best_idx = None
        best_score = -1.0
        for idx, (_, q, info) in enumerate(candidates):
            trial = [line[:] for line in fisher]
            for i in range(3):
                for j in range(3):
                    trial[i][j] += info * q[i] * q[j]
            score = math.log(max(_det3(trial), 1e-12))
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        row, q, info = candidates.pop(best_idx)
        selected.append(row)
        for i in range(3):
            for j in range(3):
                fisher[i][j] += info * q[i] * q[j]
    return [labeled_input(row) for row in selected]


def acquire_labels(
    rows: list[dict[str, Any]],
    k_labels: int,
    labeling_module: ModuleType | None,
    model: ModuleType,
    selector: str,
) -> list[dict[str, Any]]:
    if k_labels <= 0:
        return []
    if selector == "random":
        scores = []
    elif selector == "d_optimal_k3":
        labels = _d_optimal_labels(rows, k_labels, model)
        if labels:
            return labels
        print("[warn] d_optimal_k3 unavailable; using deterministic random fallback", flush=True)
        scores = []
    elif selector != "submission":
        raise ValueError(f"Unknown selector: {selector}")
    else:
        scores = None
    if scores is None:
        scores = []
    if selector == "submission" and labeling_module is not None and hasattr(labeling_module, "acquisition_function"):
        try:
            for row in rows:
                score = float(labeling_module.acquisition_function(public_input(row)))
                if not math.isfinite(score):
                    raise ValueError("non-finite acquisition score")
                scores.append((score, row))
        except Exception as exc:
            print(f"[warn] acquisition failed ({exc}); using deterministic random fallback", flush=True)
            scores = []
    if not scores:
        scores = [
            (
                stable_uniform(
                    row.get("benchmark"),
                    row.get("condition"),
                    row.get("subject_content"),
                    row.get("item_content"),
                ),
                row,
            )
            for row in rows
        ]
    scores.sort(key=lambda pair: pair[0], reverse=True)
    return [labeled_input(row) for _, row in scores[:k_labels]]


def log_loss(labels: list[int], probs: list[float]) -> float:
    total = 0.0
    for y, p in zip(labels, probs):
        p = clamp(p)
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / max(1, len(labels))


def brier(labels: list[int], probs: list[float]) -> float:
    return sum((float(y) - clamp(p)) ** 2 for y, p in zip(labels, probs)) / max(1, len(labels))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    submission_dir, tmp = resolve_submission(args)
    try:
        model = import_module(submission_dir / "model.py", "model")
        if args.disable_encoder and hasattr(model, "_ensure_encoder"):
            model._ensure_encoder = lambda: False
        if args.benchmark_prior_count is not None:
            if hasattr(model, "BENCHMARK_BAYES_PRIOR_COUNT"):
                model.BENCHMARK_BAYES_PRIOR_COUNT = float(args.benchmark_prior_count)
            else:
                print("[warn] model has no BENCHMARK_BAYES_PRIOR_COUNT override", flush=True)
        labeling_path = Path(args.labeling_path).resolve() if args.labeling_path else submission_dir / "labeling.py"
        labeling = import_module(labeling_path, "labeling") if labeling_path.exists() else None
        if args.disable_encoder and labeling is not None and hasattr(labeling, "model"):
            labeling_model = getattr(labeling, "model")
            if hasattr(labeling_model, "_ensure_encoder"):
                labeling_model._ensure_encoder = lambda: False

        benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        rows = load_rows(benchmarks, args.pairs_per_benchmark, args.seed, args.cache_dir)
        by_benchmark: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_benchmark.setdefault(row["benchmark"], []).append(row)

        fold_results = []
        all_y: list[int] = []
        all_base: list[float] = []
        all_adapted: list[float] = []

        for benchmark, bench_rows in sorted(by_benchmark.items()):
            labeled = acquire_labels(bench_rows, args.k_labels, labeling, model, args.selector)
            labeled_ids = {
                f"{ex['benchmark']}::{ex['subject_content']}::{ex['item_content']}::{ex['condition']}"
                for ex in labeled
            }
            eval_rows = []
            for row in bench_rows:
                row_id = f"{row['benchmark']}::{row['subject_content']}::{row['item_content']}::{row['condition']}"
                if args.score_labeled or row_id not in labeled_ids:
                    eval_rows.append(row)

            y_true: list[int] = []
            p_base: list[float] = []
            p_adapted: list[float] = []
            for row in eval_rows:
                inp = public_input(row)
                y_true.append(int(row["label"]))
                p_base.append(clamp(float(model.predict(inp, None))))
                p_adapted.append(clamp(float(model.predict(inp, labeled))))

            result = {
                "benchmark": benchmark,
                "eval_n": len(eval_rows),
                "label_n": len(labeled),
                "label_mean": sum(ex["label"] for ex in labeled) / max(1, len(labeled)),
                "eval_mean": sum(y_true) / max(1, len(y_true)),
                "base_log_loss": log_loss(y_true, p_base),
                "adapted_log_loss": log_loss(y_true, p_adapted),
                "base_brier": brier(y_true, p_base),
                "adapted_brier": brier(y_true, p_adapted),
                "base_pred_mean": sum(p_base) / max(1, len(p_base)),
                "adapted_pred_mean": sum(p_adapted) / max(1, len(p_adapted)),
            }
            result["delta_log_loss"] = result["adapted_log_loss"] - result["base_log_loss"]
            fold_results.append(result)
            all_y.extend(y_true)
            all_base.extend(p_base)
            all_adapted.extend(p_adapted)

        summary = {
            "submission": str(submission_dir),
            "labeling": str(labeling_path) if labeling is not None else None,
            "benchmarks": benchmarks or sorted(by_benchmark),
            "k_labels": args.k_labels,
            "pairs_per_benchmark": args.pairs_per_benchmark,
            "score_labeled": args.score_labeled,
            "overall": {
                "eval_n": len(all_y),
                "base_log_loss": log_loss(all_y, all_base),
                "adapted_log_loss": log_loss(all_y, all_adapted),
                "delta_log_loss": log_loss(all_y, all_adapted) - log_loss(all_y, all_base),
                "base_brier": brier(all_y, all_base),
                "adapted_brier": brier(all_y, all_adapted),
            },
            "folds": fold_results,
        }
        return summary
    finally:
        if tmp is not None:
            tmp.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir")
    parser.add_argument("--submission-zip")
    parser.add_argument("--labeling-path")
    parser.add_argument("--benchmarks", default="mmlupro,ai2d_test,matharena")
    parser.add_argument("--pairs-per-benchmark", type=int, default=200)
    parser.add_argument("--k-labels", type=int, default=5)
    parser.add_argument("--seed", type=int, default=326)
    parser.add_argument("--cache-dir")
    parser.add_argument("--score-labeled", action="store_true")
    parser.add_argument("--disable-encoder", action="store_true")
    parser.add_argument("--benchmark-prior-count", type=float)
    parser.add_argument("--selector", choices=["submission", "random", "d_optimal_k3"], default="submission")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    summary = evaluate(args)
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print("\nPer benchmark:")
    for fold in summary["folds"]:
        print(
            f"{fold['benchmark']:18s} n={fold['eval_n']:5d} labels={fold['label_n']:2d} "
            f"base={fold['base_log_loss']:.6f} adapted={fold['adapted_log_loss']:.6f} "
            f"delta={fold['delta_log_loss']:+.6f} "
            f"label_mean={fold['label_mean']:.3f} eval_mean={fold['eval_mean']:.3f}"
        )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
