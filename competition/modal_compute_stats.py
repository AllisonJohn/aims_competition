"""
modal_compute_stats.py
======================

OFFLINE fitting job for the Predictive AI Evaluation Challenge, run on Modal.

This script does NOT get submitted to Codabench. It is the offline step the
competition handbook requires: it downloads the public HuggingFace training
data, computes the statistics that the slide-29 baselines need, validates them
the way the hidden leaderboard works (whole-benchmark holdout), and writes a
small `stats.json` artifact that you bake into the submission ZIP next to
`model.py`.

What it produces (returned to your laptop by the local entrypoint):
  - stats.json              -> the fitted lookup model.py loads at import time
  - validation_report.json  -> negative-log-loss / AUC for steps 1, 2, 3
                               under whole-benchmark holdout (fills your
                               report's ablation table directly)

Slide-29 ladder covered here:
  step 1  return 0.5            (no fitting needed; reported for reference)
  step 2  global mean          (one number: overall train pass rate)
  step 3  smoothed subject mean (per-subject pass rate shrunk to the global
                                 mean by a pseudo-count alpha)

Run it with:
    pip install modal
    modal setup                      # one-time browser auth
    modal run modal_compute_stats.py

Optional: pin a specific smoothing strength instead of auto-selecting it:
    modal run modal_compute_stats.py --alpha 25
"""

from __future__ import annotations

import json
import math

import modal

# --------------------------------------------------------------------------- #
# Modal app, image, and a persistent volume used to cache the HF download so
# re-runs are fast and don't re-pull ~1 GB every time. This same volume/app is
# what the team reuses for the heavier later rungs (embeddings, local judge).
# --------------------------------------------------------------------------- #
app = modal.App("predeval-stats")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas==2.2.2",
        "pyarrow==16.1.0",
        "numpy==1.26.4",
        "huggingface_hub==0.27.1",
        "hf_xet",  # the dataset's parquet files are xet-backed
    )
)

cache_vol = modal.Volume.from_name("predeval-cache", create_if_missing=True)

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

# Smoothing pseudo-counts tried for step 3. alpha=0 is the raw subject mean;
# large alpha pulls every subject toward the global mean. We pick the alpha
# that maximizes the held-out objective and report the whole sweep (ablation).
ALPHA_GRID = [0.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0]

# Slide 28 / slide 35: never predict exactly 0 or 1.
CLIP_LO, CLIP_HI = 1e-3, 1.0 - 1e-3


def _clip(p: float) -> float:
    return max(CLIP_LO, min(CLIP_HI, p))


def _mean_loglik(rows, predict_fn) -> float:
    """Competition primary metric: mean log-likelihood of the true labels.

    `rows` is an iterable of (prediction_key_fields, n, s) where n is the
    number of (subject,item) responses in that group and s is how many were
    correct. Because the slide-29 baselines predict a single probability per
    group, the exact per-response log-loss collapses to a closed form:
        n_correct * log(p) + n_wrong * log(1 - p)
    Higher (closer to 0) is better, exactly as the leaderboard reports it.
    """
    total_ll = 0.0
    total_n = 0
    for key, n, s in rows:
        p = _clip(predict_fn(key))
        total_ll += s * math.log(p) + (n - s) * math.log(1.0 - p)
        total_n += n
    return total_ll / total_n if total_n else float("nan")


def _auc(rows, predict_fn) -> float:
    """Tie-aware AUC-ROC (secondary metric) for grouped constant predictions.

    Each group contributes s positives and (n - s) negatives, all at the same
    predicted probability p. AUC = P(a random correct response is scored above
    a random incorrect one), counting ties as 0.5.
    """
    by_p: dict[float, list[float]] = {}
    for key, n, s in rows:
        p = _clip(predict_fn(key))
        bucket = by_p.setdefault(p, [0.0, 0.0])
        bucket[0] += s            # positives at this score
        bucket[1] += (n - s)      # negatives at this score
    total_pos = sum(v[0] for v in by_p.values())
    total_neg = sum(v[1] for v in by_p.values())
    if total_pos == 0 or total_neg == 0:
        return float("nan")
    auc = 0.0
    cum_neg = 0.0
    for p in sorted(by_p):
        pos, neg = by_p[p]
        auc += pos * (cum_neg + 0.5 * neg)
        cum_neg += neg
    return auc / (total_pos * total_neg)


@app.function(
    image=image,
    volumes={"/cache": cache_vol},
    timeout=3600,
    cpu=4.0,
    memory=16384,
)
def compute_stats(alpha_override: float = -1.0) -> dict:
    import os

    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    hf_dir = "/cache/hf"
    os.makedirs(hf_dir, exist_ok=True)

    def fetch(filename: str) -> str:
        """Download one parquet into the volume; skip if already cached."""
        local = os.path.join(hf_dir, filename)
        if os.path.exists(local):
            return local
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=hf_dir,
        )

    # ---- discover the response tables exactly the way the README prescribes:
    # explicit files only, never load_dataset(REPO); exclude registry + traces.
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        f
        for f in repo_files
        if f.endswith(".parquet")
        and f not in REGISTRY_FILES
        and not f.endswith("_traces.parquet")
    )
    print(f"[fit] {len(response_files)} response tables: {response_files}")

    # ---- subject_id -> runtime "Name:" key.
    # At test time predict() receives subject_content beginning with
    # "Name: <display_name>". We must group the training responses by that
    # SAME string so the lookup actually hits. Mirror README.render_subject_
    # content: key = display_name if present else the subject_id.
    subjects = pd.read_parquet(fetch("subjects.parquet"))
    name_col = next(
        (c for c in ("display_name", "name", "model_name") if c in subjects.columns),
        None,
    )
    sid_to_name: dict[str, str] = {}
    for _, r in subjects.iterrows():
        sid = str(r["subject_id"])
        disp = str(r[name_col]).strip() if name_col and pd.notna(r.get(name_col)) else ""
        sid_to_name[sid] = disp if disp else sid
    print(f"[fit] {len(sid_to_name)} subjects; name column = {name_col!r}")

    # ---- aggregate per (benchmark, subject) into (n, s). Streaming one file
    # at a time keeps memory tiny: ~16 benchmarks x ~254 subjects of (n, s).
    agg_rows = []  # (benchmark, name_key, n, s)
    binary_benchmarks, dropped_continuous = [], []
    total_raw = total_kept = 0

    for fname in response_files:
        bench = fname[:-len(".parquet")]
        df = pd.read_parquet(
            fetch(fname), columns=["subject_id", "benchmark_id", "response"]
        )
        total_raw += len(df)

        df["response"] = pd.to_numeric(df["response"], errors="coerce")
        df = df[df["response"].notna()]
        # Binary-correctness objective: keep only true 0/1 responses. This
        # cleanly drops continuous/scored tables (mtbench, ultrafeedback)
        # exactly as the starter README advises for a binary model.
        df = df[df["response"].isin([0.0, 1.0])]
        if df.empty:
            dropped_continuous.append(bench)
            continue
        binary_benchmarks.append(bench)
        total_kept += len(df)

        df["name"] = df["subject_id"].astype(str).map(
            lambda s: sid_to_name.get(s, s)
        )
        bid = (
            str(df["benchmark_id"].iloc[0])
            if df["benchmark_id"].notna().any()
            else bench
        )
        grp = df.groupby("name")["response"].agg(n="count", s="sum").reset_index()
        for _, gr in grp.iterrows():
            agg_rows.append((bid, gr["name"], int(gr["n"]), float(gr["s"])))

    print(
        f"[fit] kept {total_kept:,}/{total_raw:,} binary responses "
        f"across {len(binary_benchmarks)} benchmarks; "
        f"dropped continuous: {dropped_continuous}"
    )

    benchmarks = sorted({b for b, _, _, _ in agg_rows})

    # ---- global mean (step 2) and per-subject mean (step 3 raw).
    g_s = sum(s for _, _, _, s in agg_rows)
    g_n = sum(n for _, _, n, _ in agg_rows)
    global_mean = g_s / g_n

    subj_ns: dict[str, list[float]] = {}
    for _, name, n, s in agg_rows:
        acc = subj_ns.setdefault(name, [0.0, 0.0])
        acc[0] += n
        acc[1] += s

    # ---- whole-benchmark holdout validation (slide 35). The hidden eval items
    # come from benchmarks you never trained on, so we score each baseline by
    # dropping one whole benchmark, fitting on the rest, predicting the dropped
    # one, and averaging over all folds.
    def evaluate(alpha: float):
        report = {"const_0.5": None, "global_mean": None, "subject_mean": None}
        accum = {k: [] for k in report}  # list of (key, n, s) across folds

        for held in benchmarks:
            train = [r for r in agg_rows if r[0] != held]
            test = [r for r in agg_rows if r[0] == held]
            if not test or not train:
                continue

            t_s = sum(s for _, _, _, s in train)
            t_n = sum(n for _, _, n, _ in train)
            t_global = t_s / t_n

            t_subj: dict[str, list[float]] = {}
            for _, name, n, s in train:
                a = t_subj.setdefault(name, [0.0, 0.0])
                a[0] += n
                a[1] += s
            t_smoothed = {
                name: (sv[1] + alpha * t_global) / (sv[0] + alpha)
                for name, sv in t_subj.items()
            }

            for _, name, n, s in test:
                accum["const_0.5"].append((0.5, n, s))
                accum["global_mean"].append((t_global, n, s))
                accum["subject_mean"].append(
                    (t_smoothed.get(name, t_global), n, s)
                )

        for k, rows in accum.items():
            report[k] = {
                "neg_log_loss": _mean_loglik(rows, lambda p: p),
                "auc_roc": _auc(rows, lambda p: p),
            }
        return report

    # const_0.5 and global_mean don't depend on alpha; sweep alpha for step 3.
    sweep = {}
    best_alpha, best_score = ALPHA_GRID[0], -1e18
    for a in ALPHA_GRID:
        rep = evaluate(a)
        sweep[a] = rep["subject_mean"]
        if rep["subject_mean"]["neg_log_loss"] > best_score:
            best_score, best_alpha = rep["subject_mean"]["neg_log_loss"], a

    chosen_alpha = best_alpha if alpha_override < 0 else float(alpha_override)
    final_report = evaluate(chosen_alpha)

    # ---- the artifact model.py loads at import time.
    stats = {
        "schema": "predeval-stats/v1",
        "global_mean": global_mean,
        "smoothing_alpha": chosen_alpha,
        "n_total_responses": int(g_n),
        "n_subjects": len(subj_ns),
        "benchmarks_used": benchmarks,
        "subjects": {
            name: {"mean": v[1] / v[0], "n": int(v[0])}
            for name, v in subj_ns.items()
        },
    }

    validation = {
        "method": "leave-one-benchmark-out (whole-benchmark holdout)",
        "n_folds": len(benchmarks),
        "metric_note": (
            "neg_log_loss = mean log-likelihood of true labels; higher "
            "(closer to 0) is better, matching the Codabench primary metric."
        ),
        "chosen_smoothing_alpha": chosen_alpha,
        "step1_constant_0.5": final_report["const_0.5"],
        "step2_global_mean": final_report["global_mean"],
        "step3_subject_mean": final_report["subject_mean"],
        "step3_alpha_sweep": {str(a): sweep[a] for a in ALPHA_GRID},
        "binary_benchmarks": binary_benchmarks,
        "dropped_continuous_benchmarks": dropped_continuous,
    }

    # Persist into the volume too (handy for the heavier later rungs).
    with open("/cache/stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    with open("/cache/validation_report.json", "w") as fh:
        json.dump(validation, fh, indent=2)
    cache_vol.commit()

    # Console summary so it shows up in `modal run` logs.
    print("\n===== whole-benchmark-holdout results =====")
    for label, key in (
        ("step 1  constant 0.5", "const_0.5"),
        ("step 2  global mean ", "global_mean"),
        ("step 3  subject mean", "subject_mean"),
    ):
        m = final_report[key]
        print(
            f"  {label}:  neg_log_loss={m['neg_log_loss']:.4f}  "
            f"auc={m['auc_roc']:.4f}"
        )
    print(f"  chosen smoothing alpha = {chosen_alpha}")
    print("===========================================\n")

    return {"stats": stats, "validation": validation}


@app.local_entrypoint()
def main(alpha: float = -1.0):
    """Runs the remote job, then writes both artifacts next to this script."""
    out = compute_stats.remote(alpha)

    with open("stats.json", "w") as fh:
        json.dump(out["stats"], fh, indent=2)
    with open("validation_report.json", "w") as fh:
        json.dump(out["validation"], fh, indent=2)

    s = out["stats"]
    print("\nWrote stats.json and validation_report.json to the current dir.")
    print(
        f"  global_mean={s['global_mean']:.4f}  "
        f"subjects={s['n_subjects']}  "
        f"alpha={s['smoothing_alpha']}  "
        f"responses={s['n_total_responses']:,}"
    )
    print("Next: copy stats.json into the folder with model.py, then zip.")