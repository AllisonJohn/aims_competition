"""Douglas submission model.

Training happens offline in ``training.py``. This file only loads the saved
checkpoint and serves ``predict()``.
"""

from __future__ import annotations

import os
from pathlib import Path


LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "douglas_model.pt"
# Pick exactly one submit artifact. The LM artifact is an alternative, not an
# ensemble member.
SUBMIT_ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "douglas_submit_features.pt"
# SUBMIT_ARTIFACT_PATH = Path(__file__).with_name("artifacts") / "douglas_submit_lm.pt"


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
LOAD_ERRORS = []

try:
    try:
        from .modeling import load_checkpoint
    except ImportError:
        from modeling import load_checkpoint

    cache_dir = _resolve_cache_dir()
    artifact_paths = [SUBMIT_ARTIFACT_PATH, ARTIFACT_PATH]

    for artifact_path in artifact_paths:
        if SCORER is not None:
            break
        try:
            if not artifact_path.exists():
                raise FileNotFoundError(f"Missing model artifact: {artifact_path}")
            SCORER = load_checkpoint(
                artifact_path,
                local_files_only=True,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            LOAD_ERRORS.append((artifact_path, exc))

    if SCORER is None:
        if LOAD_ERRORS:
            raise LOAD_ERRORS[0][1]
        raise FileNotFoundError("No Douglas model artifacts found.")
except Exception as exc:
    LOAD_ERROR = exc
    if not _local_smoke_test_enabled():
        print(f"[douglas/model.py] WARNING: using fallback predictor ({exc!r})", flush=True)

if LOAD_ERRORS and SCORER is not None and not _local_smoke_test_enabled():
    for artifact_path, exc in LOAD_ERRORS:
        print(
            f"[douglas/model.py] WARNING: could not load {artifact_path.name} ({exc!r})",
            flush=True,
        )


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    if SCORER is None:
        return 0.5
    return SCORER.predict_probability(input)
