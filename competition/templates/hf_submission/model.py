"""Template: local HuggingFace model submission.

Use this template when your method needs exactly one HuggingFace model listed
in `models.txt`. The worker pre-downloads that repo before the container
starts; loading it at module init hits the local HF cache only.

Implement your scoring method in `predict()`.
"""

from __future__ import annotations

import os
from pathlib import Path


LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
EMPTY_MODELS_MESSAGE = (
    "models.txt is empty or missing; the HF template requires exactly one "
    "declared HuggingFace repo. Use sample_code_submission/ for a no-HF "
    "baseline, or the advanced multi-model example if you need multiple repos."
)
MULTIPLE_MODELS_MESSAGE = (
    "models.txt declares multiple HuggingFace repos, but the default HF "
    "template requires exactly one. Use the advanced multi-model example if "
    "you need multiple repos."
)


# ---------------------------------------------------------------------------
# Module-level init: runs once when the container starts.
# ---------------------------------------------------------------------------


def _declared_models() -> list[str]:
    models_path = Path(__file__).with_name("models.txt")
    if not models_path.exists():
        return []
    return [
        line.strip()
        for line in models_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _single_declared_model() -> str:
    declared = _declared_models()
    if not declared:
        raise RuntimeError(EMPTY_MODELS_MESSAGE)
    if len(declared) > 1:
        raise RuntimeError(f"{MULTIPLE_MODELS_MESSAGE} Found {len(declared)} entries.")
    return declared[0]


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


MODEL_LOADED = False
TOKENIZER = None
MODEL = None
REPO_ID = ""

REPO_ID = _single_declared_model()

if _local_smoke_test_enabled():
    print("[hf_submission] Skipped HuggingFace load for local smoke test.", flush=True)
else:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "The HF template requires torch and transformers in the runtime. "
            "Use sample_code_submission/ for a no-HF baseline."
        ) from exc

    try:
        cache_dir = _resolve_cache_dir()
        TOKENIZER = AutoTokenizer.from_pretrained(
            REPO_ID,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        MODEL = AutoModel.from_pretrained(
            REPO_ID,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        if torch.cuda.is_available():
            MODEL = MODEL.to("cuda")
            print("[hf_submission] Loaded model on CUDA.", flush=True)
        else:
            print("[hf_submission] Loaded model on CPU.", flush=True)
        MODEL_LOADED = True
    except Exception as exc:
        raise RuntimeError(
            f"Could not load HuggingFace repo '{REPO_ID}' from the local cache. "
            "Check models.txt and make sure the repo is available to the "
            "competition pre-download step."
        ) from exc


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    # Replace this with your actual scoring logic. `input` exposes the
    # curated keys: benchmark, condition, subject_content, item_content.
    return 0.5
