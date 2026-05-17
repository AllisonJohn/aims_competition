"""
model.py  --  Predictive AI Evaluation Challenge submission.

Implements the first three rungs of the slide-29 ladder behind one switch:

    STRATEGY = "constant_05"   # step 1: return 0.5 (smoke test)
    STRATEGY = "global_mean"   # step 2: overall train pass rate
    STRATEGY = "subject_mean"  # step 3: per-subject pass rate, smoothed

How this satisfies the competition contract
-------------------------------------------
* No training happens here. Fitting is done OFFLINE by modal_compute_stats.py,
  which writes stats.json. This file only LOADS that lookup, once, at import
  time (module scope) -- never inside predict(), never over the network.
* predict() does a dict lookup + a little arithmetic, so it is fast and cannot
  raise on the runtime path (it always returns a clipped Python float).
* The adaptive-labeling `labeled` argument is intentionally unused for steps
  1-3 (label-based calibration is step 7); it is still handled cleanly when
  empty or None.

To switch between steps for the report's ablation table, change STRATEGY,
re-zip, and resubmit. stats.json is identical across all three.
"""

from __future__ import annotations

import json
import os

# --------------------------------------------------------------------------- #
# Pick the rung. Default to step 3 (a smoothed subject mean is the strongest
# of the three and, per slide 30, also your best debugger).
# --------------------------------------------------------------------------- #
STRATEGY = "subject_mean"  # "constant_05" | "global_mean" | "subject_mean"

# Slide 28 / slide 35: never emit exactly 0 or 1.
_CLIP_LO, _CLIP_HI = 1e-3, 1.0 - 1e-3

# Safe fallbacks so predict() is valid even if stats.json is missing during a
# bare smoke test (e.g. before the first Modal run).
_DEFAULT_GLOBAL = 0.5
_DEFAULT_ALPHA = 25.0

# --------------------------------------------------------------------------- #
# Module init: runs once when the container starts. Load the fitted lookup
# that lives next to this file inside the submission ZIP.
# --------------------------------------------------------------------------- #
_STATS = {
    "global_mean": _DEFAULT_GLOBAL,
    "smoothing_alpha": _DEFAULT_ALPHA,
    "subjects": {},
}

try:
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_here, "artifacts/stats.json"), "r") as _fh:
        _loaded = json.load(_fh)
    _STATS["global_mean"] = float(_loaded.get("global_mean", _DEFAULT_GLOBAL))
    _STATS["smoothing_alpha"] = float(
        _loaded.get("smoothing_alpha", _DEFAULT_ALPHA)
    )
    _STATS["subjects"] = _loaded.get("subjects", {}) or {}
    print(
        f"[model.py] loaded stats.json: strategy={STRATEGY} "
        f"global_mean={_STATS['global_mean']:.4f} "
        f"alpha={_STATS['smoothing_alpha']} "
        f"subjects={len(_STATS['subjects'])}"
    )
except Exception as exc:  # never let import fail the submission
    print(
        f"[model.py] WARNING: could not load stats.json ({exc!r}); "
        f"falling back to constant global={_DEFAULT_GLOBAL}"
    )


def _clip(p: float) -> float:
    return max(_CLIP_LO, min(_CLIP_HI, float(p)))


def _subject_key(subject_content: str) -> str | None:
    """Extract the runtime "Name:" line so it matches the offline lookup key.

    The hosted runtime renders subject_content starting with `Name: <display>`
    and may append optional metadata lines. modal_compute_stats.py keyed the
    per-subject table by that exact display string, so we parse only the first
    line and strip the `Name:` prefix. Parse defensively (README): extra lines
    may be absent or present.
    """
    if not subject_content:
        return None
    first = subject_content.splitlines()[0].strip() if subject_content else ""
    if first.lower().startswith("name:"):
        first = first[len("name:"):].strip()
    return first or None


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Probability the subject answers this item correctly. Native float."""
    global_mean = _STATS["global_mean"]

    # ---- step 1: constant smoke test -------------------------------------
    if STRATEGY == "constant_05":
        return 0.5

    # ---- step 2: global mean ---------------------------------------------
    if STRATEGY == "global_mean":
        return _clip(global_mean)

    # ---- step 3: smoothed subject mean -----------------------------------
    # Subjects are NOT cold-start (the test subjects all appear in training),
    # so a per-subject rate is real signal. Shrink sparse subjects toward the
    # global mean with a pseudo-count alpha:
    #     p = (n * subject_mean + alpha * global_mean) / (n + alpha)
    name = _subject_key(input.get("subject_content", ""))
    rec = _STATS["subjects"].get(name) if name else None
    if not rec:
        # Unseen / unparseable subject -> calibrated global fallback.
        return _clip(global_mean)

    n = float(rec.get("n", 0.0))
    m = float(rec.get("mean", global_mean))
    alpha = _STATS["smoothing_alpha"]
    denom = n + alpha
    smoothed = (n * m + alpha * global_mean) / denom if denom > 0 else global_mean
    return _clip(smoothed)


def estimate_item_difficulty(item_text: str) -> float:
    """
    Estimate relative difficulty adjustment based on item text.
    Returns multiplier: 1.0 = average, <1.0 = harder, >1.0 = easier
    """
    
    if not item_text:
        return 1.0
    
    difficulty_multiplier = 1.0
    
    # Length-based heuristics
    text_length = len(item_text)
    if text_length > 1000:
        difficulty_multiplier *= 0.90  # Very long questions are harder
    elif text_length > 500:
        difficulty_multiplier *= 0.95
    elif text_length < 100:
        difficulty_multiplier *= 1.05  # Very short questions are easier
    
    # Content-based heuristics
    text_lower = item_text.lower()
    
    # Mathematical content
    if any(term in item_text for term in ['∫', '∑', '∂', '√', '≤', '≥', '∈']):
        difficulty_multiplier *= 0.92  # Math symbols = harder
    
    # Reasoning indicators
    reasoning_words = ['prove', 'derive', 'explain why', 'justify', 'demonstrate']
    if any(word in text_lower for word in reasoning_words):
        difficulty_multiplier *= 0.93  # Reasoning = harder
    
    # Code content
    if any(keyword in text_lower for keyword in ['def ', 'function', 'class ', 'import ']):
        difficulty_multiplier *= 0.94  # Code = harder
    
    # Multi-part questions
    if item_text.count('\n') > 10 or text_lower.count('part ') > 1:
        difficulty_multiplier *= 0.92  # Multi-part = harder
    
    # Simple factual questions (likely easier)
    simple_patterns = ['what is', 'who is', 'when did', 'where is']
    if any(pattern in text_lower for pattern in simple_patterns):
        if text_length < 200:  # Only if also short
            difficulty_multiplier *= 1.08
    
    return difficulty_multiplier


