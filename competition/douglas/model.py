"""Douglas submission model.

Training happens offline in ``training.py``. This file only loads the saved
checkpoint and serves ``predict()``.
"""

from __future__ import annotations

import os
from pathlib import Path


LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "douglas_model.pt"


def _local_smoke_test_enabled() -> bool:
    value = os.environ.get(LOCAL_SMOKE_TEST_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


SCORER = None
LOAD_ERROR = None

try:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {ARTIFACT_PATH}")
    try:
        from .modeling import load_checkpoint
    except ImportError:
        from modeling import load_checkpoint

    SCORER = load_checkpoint(
        ARTIFACT_PATH,
        local_files_only=True,
        cache_dir=_resolve_cache_dir(),
    )
except Exception as exc:
    LOAD_ERROR = exc
    if not _local_smoke_test_enabled():
        print(f"[douglas/model.py] WARNING: using fallback predictor ({exc!r})", flush=True)


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    if SCORER is None:
        return 0.5
    return SCORER.predict_probability(input)
