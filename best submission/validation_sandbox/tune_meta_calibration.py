from __future__ import annotations

import argparse
import importlib.util
import json
import math
import tempfile
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

from validate_labeling import RESPONSE_FILES, clamp, load_rows, log_loss, public_input, stable_uniform


def import_module(path: Path, name_hint: str) -> ModuleType:
    module_name = f"_meta_cal_{name_hint}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_submission(path: str) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    p = Path(path).resolve()
    if p.is_dir():
        return p, None
    tmp = tempfile.TemporaryDirectory(prefix="meta_cal_submission_")
    with zipfile.ZipFile(p) as zf:
        zf.extractall(tmp.name)
    return Path(tmp.name).resolve(), tmp


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float) -> float:
    p = clamp(p)
    return math.log(p / (1.0 - p))


def key(value: object) -> str:
    return str(value or "").strip().lower()


def bayes_delta(
    labels: list[float],
    expected: list[float],
    prior_count: float,
    cap: float,
) -> float:
    if not labels:
        return 0.0
    prior_rate = clamp(sum(expected) / len(expected), 0.001, 0.999)
    observed_rate = clamp(sum(labels) / len(labels), 0.001, 0.999)
    posterior_rate = clamp(
        (prior_rate * prior_count + observed_rate * len(labels))
        / (prior_count + len(labels)),
        0.001,
        0.999,
    )
    delta = logit(posterior_rate) - logit(prior_rate)
    return max(-cap, min(cap, delta))


def group_stats(model: ModuleType, inp: dict[str, Any], labeled: list[dict[str, Any]]) -> dict[str, Any]:
    benchmark = key(inp.get("benchmark"))
    input_subject = model._parse_subject_name(inp.get("subject_content"))
    input_category = model._infer_category(inp.get("benchmark"), inp.get("condition"), inp.get("item_content"))
    input_condition = key(inp.get("condition") or "none")

    labels: list[float] = []
    expected: list[float] = []
    category_labels: list[float] = []
    category_expected: list[float] = []
    subject_labels: list[float] = []
    subject_expected: list[float] = []
    condition_labels: list[float] = []
    condition_expected: list[float] = []

    for example in labeled:
        if "label" not in example or key(example.get("benchmark")) != benchmark:
            continue
        label = clamp(float(example["label"]), 0.0, 1.0)
        expected_value = clamp(float(model._raw_predict(example)), 0.001, 0.999)
        labels.append(label)
        expected.append(expected_value)
        if model._infer_category(example.get("benchmark"), example.get("condition"), example.get("item_content")) == input_category:
            category_labels.append(label)
            category_expected.append(expected_value)
        if model._parse_subject_name(example.get("subject_content")) == input_subject:
            subject_labels.append(label)
            subject_expected.append(expected_value)
        if key(example.get("condition") or "none") == input_condition:
            condition_labels.append(label)
            condition_expected.append(expected_value)

    b = bayes_delta(labels, expected, model.BENCHMARK_BAYES_PRIOR_COUNT, model.BAYES_DELTA_CAP)
    c = bayes_delta(category_labels, category_expected, 8.0, model.CATEGORY_DELTA_CAP)
    s = bayes_delta(subject_labels, subject_expected, 6.0, model.SUBJECT_DELTA_CAP)
    d = bayes_delta(condition_labels, condition_expected, 8.0, 0.12)
    residuals = [label - pred for label, pred in zip(labels, expected)]
    residual_mean = sum(residuals) / len(residuals) if residuals else 0.0
    residual_var = (
        sum((value - residual_mean) ** 2 for value in residuals) / len(residuals)
        if residuals
        else 0.0
    )
    return {
        "benchmark_delta": b,
        "category_delta": c,
        "subject_delta": s,
        "condition_delta": d,
        "n_benchmark": len(labels),
        "n_category": len(category_labels),
        "n_subject": len(subject_labels),
        "n_condition": len(condition_labels),
        "residual_var": residual_var,
    }


def feature_vector(raw_logit: float, stats: dict[str, Any]) -> list[float]:
    b = float(stats["benchmark_delta"])
    c = float(stats["category_delta"])
    s = float(stats["subject_delta"])
    d = float(stats["condition_delta"])
    nc = float(stats["n_category"])
    ns = float(stats["n_subject"])
    nd = float(stats["n_condition"])
    return [
        1.0,
        b,
        c,
        s,
        d,
        abs(c),
        abs(s),
        c * s,
        b * c,
        b * s,
        min(nc, 5.0) / 5.0,
        min(ns, 5.0) / 5.0,
        min(nd, 5.0) / 5.0,
        1.0 if b * c >= 0.0 else 0.0,
        1.0 if b * s >= 0.0 else 0.0,
        max(-3.0, min(3.0, raw_logit)) / 3.0,
    ]


def exact_prediction(raw_p: float, stats: dict[str, Any]) -> float:
    delta = (
        float(stats["benchmark_delta"])
        + 0.65 * float(stats["category_delta"])
        + 0.45 * float(stats["subject_delta"])
    )
    return clamp(sigmoid(logit(raw_p) + delta))


def benchmark_only_prediction(raw_p: float, stats: dict[str, Any]) -> float:
    return clamp(sigmoid(logit(raw_p) + float(stats["benchmark_delta"])))


def coherence_prediction(raw_p: float, stats: dict[str, Any]) -> float:
    b = float(stats["benchmark_delta"])
    c = float(stats["category_delta"])
    s = float(stats["subject_delta"])
    extras = 0.65 * c + 0.45 * s
    coherence = 0
    if b * c >= 0.0:
        coherence += 1
    if b * s >= 0.0:
        coherence += 1
    if c * s >= 0.0:
        coherence += 1
    if coherence >= 2:
        scale = 1.0
    elif coherence == 1:
        scale = 0.5
    else:
        scale = 0.0
    return clamp(sigmoid(logit(raw_p) + b + scale * extras))


def residual_variance_prediction(raw_p: float, stats: dict[str, Any], tau: float) -> float:
    b = float(stats["benchmark_delta"])
    c = float(stats["category_delta"])
    s = float(stats["subject_delta"])
    shrink = 1.0 / (1.0 + float(stats.get("residual_var", 0.0)) / tau)
    return clamp(sigmoid(logit(raw_p) + b + shrink * (0.65 * c + 0.45 * s)))


def coherence_variance_prediction(raw_p: float, stats: dict[str, Any], tau: float = 0.15) -> float:
    b = float(stats["benchmark_delta"])
    c = float(stats["category_delta"])
    s = float(stats["subject_delta"])
    extras = 0.65 * c + 0.45 * s
    coherence = 0
    if b * c >= 0.0:
        coherence += 1
    if b * s >= 0.0:
        coherence += 1
    if c * s >= 0.0:
        coherence += 1
    gate = 1.0 if coherence >= 2 else (0.5 if coherence == 1 else 0.0)
    shrink = 1.0 / (1.0 + float(stats.get("residual_var", 0.0)) / tau)
    return clamp(sigmoid(logit(raw_p) + b + gate * shrink * extras))


def jackknife_shrink(stats: dict[str, Any], model: ModuleType, inp: dict[str, Any], labeled: list[dict[str, Any]]) -> tuple[float, float]:
    full_c = float(stats["category_delta"])
    full_s = float(stats["subject_delta"])
    if not labeled:
        return 1.0, 1.0
    c_diffs = []
    s_diffs = []
    for idx in range(len(labeled)):
        leave_one = labeled[:idx] + labeled[idx + 1 :]
        st = group_stats(model, inp, leave_one)
        c_diffs.append(abs(full_c - float(st["category_delta"])))
        s_diffs.append(abs(full_s - float(st["subject_delta"])))
    c_instability = max(c_diffs) if c_diffs else 0.0
    s_instability = max(s_diffs) if s_diffs else 0.0
    c_shrink = 1.0 / (1.0 + c_instability / 0.12)
    s_shrink = 1.0 / (1.0 + s_instability / 0.12)
    return c_shrink, s_shrink


def jackknife_prediction(
    raw_p: float,
    stats: dict[str, Any],
    model: ModuleType,
    inp: dict[str, Any],
    labeled: list[dict[str, Any]],
) -> float:
    c_shrink, s_shrink = jackknife_shrink(stats, model, inp, labeled)
    delta = (
        float(stats["benchmark_delta"])
        + 0.65 * c_shrink * float(stats["category_delta"])
        + 0.45 * s_shrink * float(stats["subject_delta"])
    )
    return clamp(sigmoid(logit(raw_p) + delta))


def train_logistic_ridge(
    features: list[list[float]],
    labels: list[int],
    offsets: list[float],
    l2: float,
    steps: int = 800,
    lr: float = 0.08,
) -> list[float]:
    if not features:
        return []
    n = len(features)
    p = len(features[0])
    beta = [0.0] * p
    # Exact-root is a good initialization.
    if p >= 4:
        beta[1] = 1.0
        beta[2] = 0.65
        beta[3] = 0.45
    mean = [sum(row[j] for row in features) / n for j in range(p)]
    scale = []
    for j in range(p):
        var = sum((row[j] - mean[j]) ** 2 for row in features) / max(1, n)
        scale.append(math.sqrt(var) if var > 1e-12 else 1.0)
    mean[0] = 0.0
    scale[0] = 1.0
    zfeatures = [[(row[j] - mean[j]) / scale[j] for j in range(p)] for row in features]
    beta = [beta[j] * scale[j] for j in range(p)]
    for _ in range(steps):
        grad = [0.0] * p
        for x, y, offset in zip(zfeatures, labels, offsets):
            eta = offset + sum(beta[j] * x[j] for j in range(p))
            pred = sigmoid(eta)
            err = pred - float(y)
            for j, value in enumerate(x):
                grad[j] += err * value
        for j in range(p):
            penalty = 0.0 if j == 0 else l2 * beta[j]
            grad[j] = grad[j] / n + penalty
            beta[j] -= lr * grad[j]
    # Convert back to raw feature space.
    raw_beta = [0.0] * p
    intercept_shift = 0.0
    for j in range(p):
        raw_beta[j] = beta[j] / scale[j]
        intercept_shift -= raw_beta[j] * mean[j]
    raw_beta[0] += intercept_shift
    return raw_beta


def meta_prediction(raw_p: float, x: list[float], beta: list[float]) -> float:
    if not beta:
        return exact_prediction(raw_p, {
            "benchmark_delta": x[1],
            "category_delta": x[2],
            "subject_delta": x[3],
        })
    delta = sum(weight * value for weight, value in zip(beta, x))
    return clamp(sigmoid(logit(raw_p) + delta))


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    submission_dir, tmp = resolve_submission(args.submission)
    try:
        model = import_module(submission_dir / "model.py", "model")
        if args.disable_encoder and hasattr(model, "_ensure_encoder"):
            model._ensure_encoder = lambda: False

        benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        if not benchmarks:
            benchmarks = [filename.removesuffix(".parquet") for filename in RESPONSE_FILES]
        rows = load_rows(benchmarks, args.pairs_per_benchmark, args.seed, args.cache_dir)
        by_benchmark: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_benchmark.setdefault(row["benchmark"], []).append(row)

        salts = [salt.strip() for salt in args.label_salts.split(",") if salt.strip()]
        if not salts:
            salts = ["0"]
        fold_data: dict[str, list[dict[str, Any]]] = {}
        for benchmark, bench_rows in sorted(by_benchmark.items()):
            records = []
            for salt in salts:
                scored_rows = [
                    (
                        stable_uniform(
                            salt,
                            row.get("benchmark"),
                            row.get("condition"),
                            row.get("subject_content"),
                            row.get("item_content"),
                        ),
                        row,
                    )
                    for row in bench_rows
                ]
                scored_rows.sort(key=lambda pair: pair[0], reverse=True)
                labeled = []
                for _, row in scored_rows[: args.k_labels]:
                    example = public_input(row)
                    example["label"] = int(row["label"])
                    labeled.append(example)
                for row in bench_rows:
                    inp = public_input(row)
                    raw_p = clamp(float(model._raw_predict(inp)), 0.001, 0.999)
                    raw_logit = logit(raw_p)
                    stats = group_stats(model, inp, labeled)
                    records.append({
                        "y": int(row["label"]),
                        "raw_p": raw_p,
                        "raw_logit": raw_logit,
                        "features": feature_vector(raw_logit, stats),
                        "exact_p": exact_prediction(raw_p, stats),
                        "bench_p": benchmark_only_prediction(raw_p, stats),
                        "coherence_p": coherence_prediction(raw_p, stats),
                        "resvar005_p": residual_variance_prediction(raw_p, stats, 0.05),
                        "resvar010_p": residual_variance_prediction(raw_p, stats, 0.10),
                        "resvar020_p": residual_variance_prediction(raw_p, stats, 0.20),
                        "coherence_var_p": coherence_variance_prediction(raw_p, stats, 0.15),
                        "jackknife_p": jackknife_prediction(raw_p, stats, model, inp, labeled),
                    })
            fold_data[benchmark] = records

        methods = [
            "base",
            "benchmark_only",
            "exact",
            "coherence",
            "resvar005",
            "resvar010",
            "resvar020",
            "coherence_var",
            "jackknife",
        ]
        for l2 in args.l2_grid:
            methods.append(f"meta_l2_{l2:g}")
        aggregate: dict[str, list[float]] = {method: [] for method in methods}
        aggregate_y: list[int] = []
        fold_results = []
        meta_betas: dict[str, list[float]] = {}

        for holdout, records in sorted(fold_data.items()):
            train_records = [
                record
                for bench, bench_records in fold_data.items()
                if bench != holdout
                for record in bench_records
            ]
            train_x = [r["features"] for r in train_records]
            train_y = [int(r["y"]) for r in train_records]
            train_offsets = [float(r["raw_logit"]) for r in train_records]
            betas = {
                l2: train_logistic_ridge(train_x, train_y, train_offsets, l2=l2, steps=args.steps, lr=args.lr)
                for l2 in args.l2_grid
            }

            y = [int(r["y"]) for r in records]
            preds: dict[str, list[float]] = {
                "base": [float(r["raw_p"]) for r in records],
                "benchmark_only": [float(r["bench_p"]) for r in records],
                "exact": [float(r["exact_p"]) for r in records],
                "coherence": [float(r["coherence_p"]) for r in records],
                "resvar005": [float(r["resvar005_p"]) for r in records],
                "resvar010": [float(r["resvar010_p"]) for r in records],
                "resvar020": [float(r["resvar020_p"]) for r in records],
                "coherence_var": [float(r["coherence_var_p"]) for r in records],
                "jackknife": [float(r["jackknife_p"]) for r in records],
            }
            for l2, beta in betas.items():
                name = f"meta_l2_{l2:g}"
                preds[name] = [meta_prediction(float(r["raw_p"]), r["features"], beta) for r in records]
                meta_betas.setdefault(name, beta)

            fold_result = {"benchmark": holdout, "n": len(records)}
            for method, values in preds.items():
                loss = log_loss(y, values)
                fold_result[method] = loss
                aggregate[method].extend(values)
            aggregate_y.extend(y)
            fold_results.append(fold_result)

        overall = {method: log_loss(aggregate_y, aggregate[method]) for method in methods}
        best_method = min(overall, key=overall.get)
        all_records = [record for records in fold_data.values() for record in records]
        final_beta = None
        if best_method.startswith("meta_l2_"):
            l2 = float(best_method.removeprefix("meta_l2_"))
            final_beta = train_logistic_ridge(
                [r["features"] for r in all_records],
                [int(r["y"]) for r in all_records],
                [float(r["raw_logit"]) for r in all_records],
                l2=l2,
                steps=args.steps,
                lr=args.lr,
            )
        return {
            "overall": overall,
            "best_method": best_method,
            "folds": fold_results,
            "final_beta": final_beta,
            "feature_names": [
                "intercept",
                "benchmark_delta",
                "category_delta",
                "subject_delta",
                "condition_delta",
                "abs_category_delta",
                "abs_subject_delta",
                "category_x_subject",
                "benchmark_x_category",
                "benchmark_x_subject",
                "n_category",
                "n_subject",
                "n_condition",
                "benchmark_category_agree",
                "benchmark_subject_agree",
                "raw_logit_scaled",
            ],
        }
    finally:
        if tmp is not None:
            tmp.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="competition/ridge-k3-labeled-type-residual-exact-root.zip")
    parser.add_argument("--benchmarks", default=",".join(filename.removesuffix(".parquet") for filename in RESPONSE_FILES))
    parser.add_argument("--pairs-per-benchmark", type=int, default=200)
    parser.add_argument("--k-labels", type=int, default=5)
    parser.add_argument("--seed", type=int, default=326)
    parser.add_argument("--label-salts", default="0")
    parser.add_argument("--cache-dir")
    parser.add_argument("--disable-encoder", action="store_true")
    parser.add_argument("--l2-grid", type=float, nargs="+", default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    summary = evaluate(args)
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(f"best_method={summary['best_method']}")
    print("\nPer benchmark:")
    methods = list(summary["overall"].keys())
    for fold in summary["folds"]:
        losses = " ".join(f"{method}={fold[method]:.4f}" for method in methods)
        print(f"{fold['benchmark']:18s} n={fold['n']:5d} {losses}")
    if summary.get("final_beta"):
        print("\nFinal beta:")
        for name, value in zip(summary["feature_names"], summary["final_beta"]):
            print(f"{name:28s} {value:+.6f}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
