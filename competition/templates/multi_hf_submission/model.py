"""Advanced template: load multiple declared HuggingFace repos.

Use this only when your method really needs more than one local HuggingFace
model. The worker pre-downloads every repo in models.txt before the container
starts. Loading uses the local HF cache only.
"""

from __future__ import annotations

import os
from pathlib import Path


LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"


def _declared_models() -> list[str]:
    models_path = Path(__file__).with_name("models.txt")
    if not models_path.exists():
        raise RuntimeError(
            "models.txt is empty or missing; this advanced template needs at "
            "least one declared HuggingFace repo."
        )
    repos = [
        line.strip()
        for line in models_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not repos:
        raise RuntimeError(
            "models.txt is empty or comment-only; declare one or more "
            "HuggingFace repos or use sample_code_submission/ for a no-HF baseline."
        )
    return repos


def _resolve_cache_dir() -> str | None:
    candidates = [
        os.environ.get("HF_HOME", "").strip(),
        "/app/hf_cache",
        str(Path(__file__).with_name(".hf_cache")),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(path, os.W_OK):
            return str(path)
    return None


def _local_smoke_test_enabled() -> bool:
    value = os.environ.get(LOCAL_SMOKE_TEST_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


LOCAL_SMOKE_TEST = _local_smoke_test_enabled()
REPO_IDS = _declared_models()
CACHE_DIR = None if LOCAL_SMOKE_TEST else _resolve_cache_dir()
TOKENIZERS = {}
MODELS = {}

if LOCAL_SMOKE_TEST:
    print("[multi_hf_submission] Skipped HuggingFace loads for local smoke test.", flush=True)
else:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "This multi-model HF template requires torch and transformers. "
            "Use sample_code_submission/ for a no-HF baseline."
        ) from exc

    for repo_id in REPO_IDS:
        TOKENIZERS[repo_id] = AutoTokenizer.from_pretrained(
            repo_id,
            cache_dir=CACHE_DIR,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            repo_id,
            cache_dir=CACHE_DIR,
            local_files_only=True,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        MODELS[repo_id] = model


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    # Replace this with your scoring logic over MODELS and TOKENIZERS.
    return 0.5
