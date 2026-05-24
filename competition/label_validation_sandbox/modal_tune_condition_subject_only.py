from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("aims-label-condition-subject-only-sweep")

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
def tune_remote(
    submission_files: dict[str, bytes],
    validator_source: str,
    benchmarks: str,
    pairs_per_benchmark: int,
    k_labels: int,
    selector: str,
    score_labeled: bool,
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
        benchmark_prior_count=None,
        selector=selector,
        json_out=None,
    )

    grid = [{"name": "none", "cond_w": 0.0, "subj_w": 0.0, "cond_prec": 12.0, "subj_prec": 8.0}]
    for cond_w in (0.15, 0.25, 0.35, 0.50):
        for cond_prec in (8.0, 12.0, 20.0):
            grid.append(
                {
                    "name": "condition_only",
                    "cond_w": cond_w,
                    "subj_w": 0.0,
                    "cond_prec": cond_prec,
                    "subj_prec": 8.0,
                }
            )
    for subj_w in (0.10, 0.20, 0.30):
        for subj_prec in (6.0, 8.0, 12.0):
            grid.append(
                {
                    "name": "subject_only",
                    "cond_w": 0.0,
                    "subj_w": subj_w,
                    "cond_prec": 12.0,
                    "subj_prec": subj_prec,
                }
            )
    for cond_w, subj_w in ((0.15, 0.10), (0.25, 0.10), (0.25, 0.20), (0.35, 0.10)):
        grid.append(
            {
                "name": "condition_subject_only",
                "cond_w": cond_w,
                "subj_w": subj_w,
                "cond_prec": 12.0,
                "subj_prec": 8.0,
            }
        )

    results = []
    for config in grid:
        module_name = f"model_{len(results)}"
        model_path = submission_dir / "model.py"
        model_spec = importlib.util.spec_from_file_location(module_name, model_path)
        if model_spec is None or model_spec.loader is None:
            raise RuntimeError("Could not import model")
        model = importlib.util.module_from_spec(model_spec)
        old_path = list(sys.path)
        sys.path.insert(0, str(submission_dir))
        try:
            model_spec.loader.exec_module(model)
        finally:
            sys.path[:] = old_path

        def calibrate(prediction, input, labeled, cfg=config, m=model):
            if not labeled:
                return prediction
            delta = 0.0
            delta += float(cfg["cond_w"]) * m._residual_delta(
                input,
                labeled,
                "condition",
                float(cfg["cond_prec"]),
                0.12,
            )
            delta += float(cfg["subj_w"]) * m._residual_delta(
                input,
                labeled,
                "subject",
                float(cfg["subj_prec"]),
                0.16,
            )
            delta = max(-0.20, min(0.20, delta))
            return m._clamp(m._sigmoid(m._logit(prediction) + delta))

        model._calibrate_with_labeled = calibrate

        original_import_module = validator.import_module

        def import_module_override(path, name_hint, m=model):
            if Path(path).name == "model.py":
                return m
            return original_import_module(path, name_hint)

        validator.import_module = import_module_override
        summary = validator.evaluate(args)
        validator.import_module = original_import_module

        result = dict(config)
        result.update(summary["overall"])
        results.append(result)
        print(
            "config "
            f"{config['name']} cond={config['cond_w']} cp={config['cond_prec']} "
            f"subj={config['subj_w']} sp={config['subj_prec']} "
            f"adapted={result['adapted_log_loss']:.6f} delta={result['delta_log_loss']:+.6f}",
            flush=True,
        )

    results.sort(key=lambda row: row["adapted_log_loss"])
    print("top configs:", flush=True)
    for row in results[:10]:
        print(json.dumps(row, sort_keys=True), flush=True)
    return {"results": results, "top": results[:10]}


@app.local_entrypoint()
def main(
    submission_dir: str = "label_validation_sandbox/submissions/k3_benchmark_condition_subject_bayes",
    benchmarks: str = "afrimedqa,agentdojo,ai2d_test,androidworld,bfcl,cybench,hle,livecodebench,matharena,mathvista_mini,mmbench_v11,mmlupro,mtbench,rewardbench,swebench,ultrafeedback",
    pairs_per_benchmark: int = 100,
    k_labels: int = 5,
    selector: str = "random",
    score_labeled: bool = True,
    json_out: str | None = "label_validation_sandbox/condition_subject_only_sweep.json",
) -> None:
    root = Path(__file__).resolve().parents[1]
    submission_path = (root / submission_dir).resolve()
    if not submission_path.exists():
        raise SystemExit(f"Submission dir does not exist: {submission_path}")
    validator_source = (Path(__file__).resolve().parent / "validate_labeling.py").read_text(encoding="utf-8")
    summary = tune_remote.remote(
        _read_tree(submission_path),
        validator_source,
        benchmarks,
        pairs_per_benchmark,
        k_labels,
        selector,
        score_labeled,
    )
    if json_out:
        (root / json_out).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["top"], indent=2, sort_keys=True))
