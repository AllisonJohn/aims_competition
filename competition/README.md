# Starting Kit

This starting kit supports the active code-submission path for the Predictive
AI Evaluation Challenge.

Download:

- the starter kit from Codabench
- the public training dataset from
  `https://huggingface.co/datasets/aims-foundations/measurement-db`

The real competition materializes a hidden test slice at runtime. The active
configuration samples 5000 hidden items per submission, stratified across data
categories, and your `predict()` is called once per hidden subject-item pair.
Codabench currently allows up to 50 scored submissions per team per calendar
day (UTC), with a 1000-submission total limit for the competition phase.

## Contents

```text
starting_kit/
  README.md
  sample_code_submission/
    model.py
    labeling.py
  sample_data/
    test/
    ref/
  templates/
    hf_submission/
    multi_hf_submission/
    labeling_addon/
  tools/
    check_submission_zip.py
    run_smoke_test.py
```

## Loading The Public Training Data

The HuggingFace repo is a collection of Parquet tables, not a single
`datasets` split. Do **not** use:

```python
load_dataset("aims-foundations/measurement-db")
```

That shortcut lets HuggingFace auto-select files and may mix response tables
with `*_traces.parquet` tables that have different schemas. Load the response
tables explicitly and keep the registry tables separate:

```python
from datasets import Features, Value, load_dataset
from huggingface_hub import HfApi

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
response_files = sorted(
    name
    for name in repo_files
    if name.endswith(".parquet")
    and name not in REGISTRY_FILES
    and not name.endswith("_traces.parquet")
)

response_features = Features(
    {
        "subject_id": Value("string"),
        "item_id": Value("string"),
        "benchmark_id": Value("string"),
        "trial": Value("int64"),
        "test_condition": Value("string"),
        "response": Value("float64"),
        "correct_answer": Value("string"),
        "trace": Value("string"),
    }
)

responses = load_dataset(
    REPO_ID,
    data_files=response_files,
    features=response_features,
    split="train",
)
items = load_dataset(REPO_ID, data_files="items.parquet", split="train")
subjects = load_dataset(REPO_ID, data_files="subjects.parquet", split="train")
benchmarks = load_dataset(REPO_ID, data_files="benchmarks.parquet", split="train")
```

To convert a response row into the shape your `predict()` function sees, join
through the registry tables. The hosted runtime passes `subject_content` as a
string beginning with a `Name:` line; optional metadata may be appended as
additional lines when available.

```python
items_by_id = {row["item_id"]: row for row in items}
subjects_by_id = {row["subject_id"]: row for row in subjects}
benchmarks_by_id = {row["benchmark_id"]: row for row in benchmarks}


def render_subject_content(subject, fallback_subject_id):
    display_name = subject.get("display_name") or fallback_subject_id
    lines = [f"Name: {display_name}"]
    optional_fields = (
        ("provider", "Organization"),
        ("params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    )
    for key, label in optional_fields:
        value = subject.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def to_training_example(row):
    item = items_by_id.get(row["item_id"], {})
    subject = subjects_by_id.get(row["subject_id"], {})
    benchmark = benchmarks_by_id.get(row["benchmark_id"], {})
    benchmark_id = row["benchmark_id"]
    if "benchmark_id" in benchmark and benchmark["benchmark_id"]:
        benchmark_id = benchmark["benchmark_id"]

    return {
        "benchmark": benchmark_id,
        "condition": row["test_condition"] or "none",
        "subject_content": render_subject_content(subject, row["subject_id"]),
        "item_content": item.get("content"),
        "label": row["response"],
    }
```

Use the public `benchmark_id` for the `benchmark` field to match hosted runtime
inputs; `benchmarks.parquet["name"]` is human-readable metadata and may differ
from the identifier passed to `predict()`.

Runtime `model_id` corresponds to the public `subject_id` in the training
dataset. This release does not expose stable IDs at runtime. Treat
`subject_content` as display text. Do not parse it as a stable serialization
of the full `subjects.parquet` row. At test time, extra metadata lines should
be parsed defensively because they may be absent. If you use the `labeled`
examples for adaptation, match them by the visible content fields.

Some public response tables are binary and some are continuous/scored. For a
binary correctness model, filter or transform `label` values to match your
training objective. If you need raw model outputs, load the `*_traces.parquet`
files separately; they intentionally do not have the same schema as the
response tables.

### Training Data Gotchas

- Public response files can include repeated trials. In the public export,
  `test_condition` is normalized, different conditions are kept as separate
  item variants, and only the smallest `trial` is kept within each
  `(subject_id, item_id, test_condition)` group. Any duplicates left after that
  step are rejected.
- `test_condition` is part of the task context. Preserve it when you build
  validation splits or aggregate public rows.
- Hidden scoring uses binary labels. Public response values may be binary,
  Likert-style, fractional, or otherwise scored. Check response values before
  treating them as `0` or `1` labels.

## Which Starter To Use

- `sample_code_submission/`
  Minimal `model.py` plus an optional `labeling.py` example. Remove
  `labeling.py` if you want the platform's default random label sample.
- `templates/hf_submission/`
  Local HuggingFace inference using exactly one repo declared in `models.txt`.
  Use `sample_code_submission/` if you do not need a HuggingFace model.
- `templates/multi_hf_submission/`
  Advanced example for loading multiple declared HuggingFace repos from the
  local cache.
- `templates/labeling_addon/`
  Optional `labeling.py` example.

## Local Preflight Checks

Start from the sample submission or one of the templates, then create the upload
ZIP from inside your submission directory so `model.py` is at the archive root:

```bash
cp -R sample_code_submission my_submission
(cd my_submission && zip -r ../my_submission.zip .)
```

From the starting-kit directory, check the ZIP layout and run a tiny local
smoke test:

```bash
python tools/check_submission_zip.py my_submission.zip
python tools/run_smoke_test.py my_submission/
```

The smoke test uses the bundled `sample_data/` files only. Passing it does not
guarantee a strong score. It catches common layout, import, interface, and
return-value mistakes before uploading to Codabench.

The local tools set `PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1` while importing your
submission. The shipped HuggingFace templates use that flag to check
`models.txt` and the callable interface without requiring a local model cache.
Hosted submissions still load declared repos from the pre-download cache before
hidden evaluation starts.

## Required Submission Contract

Your ZIP must contain `model.py` with:

```python
def predict(input: dict, labeled: list[dict] | None = None) -> float:
    ...
```

`input` keys:

| key | meaning |
| --- | --- |
| `benchmark` | benchmark identifier (e.g. `mmlupro`, `ai2d_test`) |
| `condition` | test condition (e.g. `zero-shot`); `"none"` when not applicable |
| `subject_content` | description of the AI subject under evaluation, including a name line and any organizer-provided metadata |
| `item_content` | the question/prompt/task text the subject is asked |

`predict()` returns a single finite float in `[0, 1]`: the predicted probability
that the subject answers the item correctly. `NaN`, infinity, strings, tensors,
and values outside `[0, 1]` fail the submission. Codabench shows this error:
`Invalid predict() output: predict() must return a finite float in [0, 1].`
Exceptions from `predict()` also fail the submission.

Training data lives on the public HuggingFace dataset with the same four string
input fields after joining the response tables to the registry tables shown
above. Download it and preprocess it however you prefer before submitting.

Module-level code runs once when the container starts, before `predict()` is
called. Import or setup failures in module-level code fail the submission before
any predictions are made. Do all heavy setup (load weights, tokenizers, prompt
templates, or lookup tables) at module init.
Training must happen **offline**. Small fitted state should be baked into the
submission ZIP; large checkpoints should be uploaded to a HuggingFace model
repository and declared in `models.txt`.

Each hosted run has a total wall-clock time limit. Module import/setup,
HuggingFace loading, adaptive labeling, and every `predict()` call all count
against that total. GPU tier timeouts are listed below. Operators may also set
per-call timeouts for `predict()` or `acquisition_function()`; a per-call
timeout fails `predict()` or falls back to random adaptive labels. Do not rely on
a specific per-call timeout value unless the hosted competition publishes one.

Hosted hidden-eval logs do not show raw stdout/stderr, whether the submission
finishes or fails. Use the local smoke test for raw `print()` debugging before
uploading.

## Adaptive Labeling

You may include `labeling.py` with:

```python
def acquisition_function(input: dict) -> float:
    ...
```

`acquisition_function()` is called once per hidden `(model_id, item_id)` pair
before `predict()`. Here, `model_id` is the runtime name for the public
`subject_id`. Higher scores indicate pairs you want labeled more. A data
category is an internal organizer grouping used for sampling and label budgets.
The runtime only passes the four `input` fields listed above. The platform
selects the top **K=5** inputs per data category, resolves their ground-truth
labels, and passes them to
`predict()` as the `labeled` argument: a list of dicts with the same visible
content fields as `input` plus a `label` field (0 or 1).

If you don't include `labeling.py`, the platform reveals a default random sample
per data category. If `acquisition_function()` raises an exception, times out,
or returns a non-finite value for any candidate, the platform falls back to that
same random-selection default for the round. For adaptive labeling, a round is
the whole hosted submission run. One bad acquisition score discards all
acquisition scores for that run. Tied acquisition scores are broken randomly.
Your `predict()` should handle the empty-list case cleanly.

## HuggingFace And GPU Routing

If you need local HuggingFace repos, list them in `models.txt`. The platform
pre-downloads those repos before participant code runs and routes the submission
to B200-family Modal hardware based on the largest declared model. The active
bundle allows at most `5` repos in `models.txt`. The default HF template is
intentionally single-model and fails clearly if `models.txt` is missing, empty,
comment-only, or lists multiple repos; use the advanced multi-model example if
you need more than one repo.

Active parameter bands:

| Max params | Active GPU tier | Tier timeout |
| --- | --- | --- |
| `<= 70B` | B200 | 8 hours |
| `<= 140B` | B200:2 | 8 hours |
| `<= 300B` | B200:4 | 8 hours |

Submissions above `300B` parameters for any single declared repo are rejected
during classification. The `300B` cap is per single declared model/repo, not the
sum of all `models.txt` entries. A `5x70B` setup is not rejected by parameter sum
alone, though the active model-count limit and download-size limits still
apply. Each repo has a `1000 GB` per-repo download limit, and the declared repos
also share a combined download limit of `1024 GB`. Organizers may disable tiers
operationally if capacity changes.

If your code imports GPU or HuggingFace libraries but does not declare
`models.txt`, the platform may still route it to a GPU tier based on source-code
patterns. Do not rely on runtime HuggingFace downloads inside the submission
container.

## Runtime Policy

- submissions have **no outbound internet access** except the organizer's
  internal data-service; calling third-party hosted LLM endpoints, remote
  embedding services, external object storage, remote databases, webhooks, or
  external cloud functions is blocked
- `models.txt` guarantees that repo files are pre-downloaded, not that every
  repo will load automatically
- `trust_remote_code` is organizer-controlled and defaults to disabled in the
  example deployment
- additive submission `requirements.txt` support is organizer-controlled and
  defaults to disabled in the example deployment
- when additive dependency installs are enabled, normal named pip requirements
  are accepted into a per-submission dependency layer; avoid pip options,
  editable installs, and source-build-only packages

The organizers provide the `torch_measure` package to support measurement-model
implementation. Use it only if it helps your approach.

## Leaderboard Metric

The primary leaderboard metric is negative log-loss, and higher is better for
the displayed score. AUC-ROC is secondary. Log-loss rewards calibrated
probabilities and punishes overconfident wrong answers, so a prediction like
`0.99` can help when it is right but hurts badly when it is wrong.
