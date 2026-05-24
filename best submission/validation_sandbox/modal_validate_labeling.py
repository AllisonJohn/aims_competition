from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal


app = modal.App("aims-label-validation-sandbox")

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


def _read_tree(root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc"):
            files[str(path.relative_to(root))] = path.read_bytes()
    return files


@app.function(gpu="H100", image=image, timeout=60 * 60 * 2)
def validate_remote(
    submission_files: dict[str, bytes],
    validator_source: str,
    benchmarks: str,
    pairs_per_benchmark: int,
    k_labels: int,
    selector: str,
    score_labeled: bool,
    benchmark_prior_count: float | None,
) -> dict:
    import importlib.util
    import sys
    import types
    from pathlib import Path

    from huggingface_hub import snapshot_download

    work_dir = Path("/tmp/label_validation_sandbox")
    submission_dir = work_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    for relative, content in submission_files.items():
        path = submission_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    validator_path = work_dir / "validate_labeling.py"
    validator_path.write_text(validator_source, encoding="utf-8")

    models_path = submission_dir / "models.txt"
    if models_path.exists():
        for line in models_path.read_text(encoding="utf-8").splitlines():
            model_id = line.strip()
            if model_id:
                print(f"snapshot_download({model_id})", flush=True)
                snapshot_download(model_id)

    spec = importlib.util.spec_from_file_location("validate_labeling_remote", validator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import validator")
    validator = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(work_dir))
    spec.loader.exec_module(validator)

    args = types.SimpleNamespace(
        submission_dir=str(submission_dir),
        submission_zip=None,
        labeling_path=None,
        benchmarks=benchmarks,
        pairs_per_benchmark=int(pairs_per_benchmark),
        k_labels=int(k_labels),
        seed=326,
        cache_dir=None,
        score_labeled=bool(score_labeled),
        disable_encoder=False,
        benchmark_prior_count=benchmark_prior_count,
        selector=selector,
        json_out=None,
    )
    summary = validator.evaluate(args)
    print(json.dumps(summary["overall"], indent=2, sort_keys=True), flush=True)
    for fold in summary["folds"]:
        print(
            f"{fold['benchmark']:18s} n={fold['eval_n']:5d} labels={fold['label_n']:2d} "
            f"base={fold['base_log_loss']:.6f} adapted={fold['adapted_log_loss']:.6f} "
            f"delta={fold['delta_log_loss']:+.6f} "
            f"label_mean={fold['label_mean']:.3f} eval_mean={fold['eval_mean']:.3f}",
            flush=True,
        )
    return summary


@app.local_entrypoint()
def main(
    submission_dir: str = "label_validation_sandbox/submissions/k3_theta_update",
    benchmarks: str = "mmlupro,ai2d_test,matharena",
    pairs_per_benchmark: int = 200,
    k_labels: int = 5,
    selector: str = "submission",
    score_labeled: bool = True,
    benchmark_prior_count: float | None = None,
    json_out: str | None = None,
) -> None:
    root = Path(__file__).resolve().parents[1]
    submission_path = (root / submission_dir).resolve()
    if not submission_path.exists():
        raise SystemExit(f"Submission dir does not exist: {submission_path}")
    validator_source = (Path(__file__).resolve().parent / "validate_labeling.py").read_text(encoding="utf-8")
    summary = validate_remote.remote(
        _read_tree(submission_path),
        validator_source,
        benchmarks,
        pairs_per_benchmark,
        k_labels,
        selector,
        score_labeled,
        benchmark_prior_count,
    )
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print("\nPer benchmark:")
    for fold in summary["folds"]:
        print(
            f"{fold['benchmark']:18s} n={fold['eval_n']:5d} labels={fold['label_n']:2d} "
            f"base={fold['base_log_loss']:.6f} adapted={fold['adapted_log_loss']:.6f} "
            f"delta={fold['delta_log_loss']:+.6f} "
            f"label_mean={fold['label_mean']:.3f} eval_mean={fold['eval_mean']:.3f}"
        )
    if json_out:
        Path(json_out).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
