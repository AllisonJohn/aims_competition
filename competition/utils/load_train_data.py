import math
import numbers
from collections.abc import Callable, Iterable

from datasets import Features, Value, load_dataset
from huggingface_hub import HfApi

# Please call get_training_data() to get your data.
# You usually don't need to pass any params. 
# If you really need to pass it:
# limit: Optional maximum number of samples loaded
# response_file_limit: Optional maximum number of parquet files loaded

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

RESPONSE_FEATURES = Features(
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


def get_response_files(repo_id: str = REPO_ID) -> list[str]:
    """Return response parquet files selected for training.

    Args:
        repo_id: Hugging Face dataset repository ID.

    Returns:
        Sorted parquet file names that contain response rows. Registry tables
        and trace tables are excluded because they have different schemas.
    """
    repo_files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
    return sorted(
        name
        for name in repo_files
        if name.endswith(".parquet")
        and name not in REGISTRY_FILES
        and not name.endswith("_traces.parquet")
    )


def load_registries(repo_id: str = REPO_ID):
    """Load small registry tables needed to join training rows.

    Args:
        repo_id: Hugging Face dataset repository ID.

    Returns:
        A dict containing the raw ``items``, ``subjects``, and ``benchmarks``
        datasets plus ID lookup dictionaries for fast joins.
    """
    items = load_dataset(repo_id, data_files="items.parquet", split="train")
    subjects = load_dataset(repo_id, data_files="subjects.parquet", split="train")
    benchmarks = load_dataset(repo_id, data_files="benchmarks.parquet", split="train")

    return {
        "items": items,
        "subjects": subjects,
        "benchmarks": benchmarks,
        "items_by_id": {row["item_id"]: row for row in items},
        "subjects_by_id": {row["subject_id"]: row for row in subjects},
        "benchmarks_by_id": {row["benchmark_id"]: row for row in benchmarks},
    }


def load_responses(
    response_files: list[str] | None = None,
    repo_id: str = REPO_ID,
):
    """Stream response rows without saving the full dataset locally.

    Args:
        response_files: Optional list of response parquet files to read. If
            omitted, all response files returned by ``get_response_files`` are
            used.
        repo_id: Hugging Face dataset repository ID.

    Returns:
        A Hugging Face iterable dataset of raw response rows.
    """
    if response_files is None:
        response_files = get_response_files(repo_id=repo_id)

    return load_dataset(
        repo_id,
        data_files=response_files,
        features=RESPONSE_FEATURES,
        split="train",
        streaming=True,
    )


def benchmark_id_from_response_file(response_file: str) -> str:
    """Return the benchmark ID implied by a response parquet filename."""
    suffix = ".parquet"
    return response_file[:-len(suffix)] if response_file.endswith(suffix) else response_file


DEFAULT_SPLIT_RATIOS = (0.75, 0.125, 0.125)


def split_response_files_by_benchmark(
    response_files: list[str],
    split_index: int = 0,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, list[str]]:
    """Split response files into train/validation/test by whole benchmark.

    The competition hidden set is best approximated by holding out complete
    benchmarks, not random rows. Ratios are converted into benchmark counts by
    rounding train and validation counts, then assigning test the remaining
    files. Slicing the rotated file list keeps the splits non-overlapping.
    """
    if len(response_files) < 3:
        raise ValueError("Need at least 3 response files to make train/validation/test splits.")

    train_ratio, validation_ratio, test_ratio = split_ratios
    if train_ratio < 0 or validation_ratio < 0 or test_ratio < 0:
        raise ValueError("split ratios must be non-negative.")
    ratio_total = train_ratio + validation_ratio + test_ratio
    if not math.isclose(ratio_total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1.0.")

    files = sorted(response_files)
    n_files = len(files)
    offset = split_index % n_files
    files = files[offset:] + files[:offset]

    train_count = round(n_files * train_ratio)
    validation_count = round(n_files * validation_ratio)
    train_count = max(0, min(train_count, n_files))
    validation_count = max(0, min(validation_count, n_files - train_count))

    train_end = train_count
    validation_end = train_end + validation_count

    train = files[:train_end]
    validation = files[train_end:validation_end]
    test = files[validation_end:]

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def render_subject_content(subject, fallback_subject_id):
    """Convert a subject registry row into hosted-runtime subject text.

    Args:
        subject: A row from ``subjects.parquet`` as a dict-like object.
        fallback_subject_id: Subject ID to use when ``display_name`` is absent.

    Returns:
        A string beginning with ``Name: ...`` and optional metadata lines.
    """
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


def to_training_example(row, registries: dict):
    """Join one raw response row into the shape expected by ``predict``.

    Args:
        row: One raw response row from a response parquet file.
        registries: Output from ``load_registries``. Used to look up item,
            subject, and benchmark metadata by ID.

    Returns:
        A dict with ``benchmark``, ``condition``, ``subject_content``,
        ``item_content``, and ``label``.
    """
    item = registries["items_by_id"].get(row["item_id"], {})
    subject = registries["subjects_by_id"].get(row["subject_id"], {})
    benchmark = registries["benchmarks_by_id"].get(row["benchmark_id"], {})
    benchmark_id = benchmark.get("benchmark_id") or row["benchmark_id"]

    return {
        "model_id": row["subject_id"],
        "item_id": row["item_id"],
        "benchmark": benchmark_id,
        "condition": row["test_condition"] or "none",
        "subject_content": render_subject_content(subject, row["subject_id"]),
        "item_content": item.get("content"),
        "label": row["response"],
    }


def iter_training_examples(
    limit: int | None = None,
    response_files: list[str] | None = None,
    repo_id: str = REPO_ID,
    registries: dict | None = None,
):
    """Yield joined training examples in the same shape as ``predict`` inputs.

    Args:
        limit: Optional maximum number of examples to yield. Use this for quick
            experiments or smoke tests.
        response_files: Optional list of response parquet files to stream. If
            omitted, all response files are used.
        repo_id: Hugging Face dataset repository ID.
        registries: Optional preloaded registry dict from ``load_registries``.
            Passing this avoids reloading registry tables.

    Yields:
        Joined training examples as dictionaries.
    """
    if registries is None:
        registries = load_registries(repo_id=repo_id)
    responses = load_responses(response_files=response_files, repo_id=repo_id)

    for index, row in enumerate(responses):
        if limit is not None and index >= limit:
            break
        yield to_training_example(row, registries)


def get_benchmark_ids(repo_id: str = REPO_ID) -> list[str]:
    """Return benchmark IDs from the small registry table.

    Args:
        repo_id: Hugging Face dataset repository ID.

    Returns:
        Sorted benchmark IDs from ``benchmarks.parquet``.
    """
    registries = load_registries(repo_id=repo_id)
    return sorted(row["benchmark_id"] for row in registries["benchmarks"])


def get_training_data(
    limit: int | None = None,
    response_file_limit: int | None = None,
    repo_id: str = REPO_ID,
) -> dict:
    """Return everything a trainer needs to stream joined training examples.

    Args:
        limit: Optional maximum number of samples loaded
        response_file_limit: Optional maximum number of parquet files loaded
        repo_id: Hugging Face dataset repository ID.

    Returns:
        A dict containing metadata, preloaded registries, and ``examples``. The
        ``examples`` value is a lazy iterator of joined training examples.

    Notes:
    The returned ``examples`` value is an iterator. It does not save the full
    dataset locally and only starts reading response rows when consumed.
    """
    response_files = get_response_files(repo_id=repo_id)
    if response_file_limit is not None:
        response_files = response_files[:response_file_limit]

    registries = load_registries(repo_id=repo_id)
    examples = iter_training_examples(
        limit=limit,
        response_files=response_files,
        repo_id=repo_id,
        registries=registries,
    )

    return {
        "repo_id": repo_id,
        "num_response_files": len(response_files),
        "response_files": response_files,
        "benchmark_ids": sorted(row["benchmark_id"] for row in registries["benchmarks"]),
        "registries": registries,
        "examples": examples,
    }


def load_split_data(
    split: str | None = None,
    split_index: int = 0,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
):
    """Return benchmark-disjoint train/validation/test data.

    Args:
        split: Optional split name. Use ``None`` to return all three splits as
            ``(train_data, validation_data, test_data)``, or pass one of
            ``"train"``, ``"validation"``, or ``"test"`` to return only that
            split.
        split_index: Rotates which complete benchmarks are held out.
        split_ratios: Train/validation/test benchmark ratios. For example,
            ``(0.8, 0.0, 0.2)`` uses about 80% of benchmarks for training, no
            validation split, and the remaining benchmarks for testing.

    Returns:
        Either one split dict or a tuple of three split dicts. Each split dict
        contains metadata, benchmark IDs, response files, and a lazy
        ``examples`` iterator of joined examples. A zero-ratio split returns
        ``None``.
    """
    selected_split = split.strip().lower() if split is not None else None
    valid_splits = {"train", "validation", "test"}
    if selected_split is not None and selected_split not in valid_splits:
        raise ValueError("split must be one of: train, validation, test.")

    response_files = get_response_files(repo_id=REPO_ID)
    split_files = split_response_files_by_benchmark(
        response_files=response_files,
        split_index=split_index,
        split_ratios=split_ratios,
    )
    registries = load_registries(repo_id=REPO_ID)

    def build_split(split_name: str) -> dict | None:
        selected_files = split_files[split_name]
        if not selected_files:
            return None
        return {
            "repo_id": REPO_ID,
            "split": split_name,
            "split_index": split_index,
            "split_ratios": split_ratios,
            "num_response_files": len(response_files),
            "response_files": selected_files,
            "benchmark_ids": [
                benchmark_id_from_response_file(file)
                for file in selected_files
            ],
            "registries": registries,
            "examples": iter_training_examples(
                response_files=selected_files,
                repo_id=REPO_ID,
                registries=registries,
            ),
        }

    if selected_split is not None:
        return build_split(selected_split)

    return (
        build_split("train"),
        build_split("validation"),
        build_split("test"),
    )


def evaluate(
    predict_fn: Callable[[dict, list[dict] | None], float],
    examples: Iterable[dict],
) -> dict:
    """Evaluate a predictor on one binary-labeled split.

    Pass ``validation_data["examples"]`` or ``test_data["examples"]``. The
    behavior is identical for either split; this function intentionally does
    not know which split it received.
    """
    labels: list[int] = []
    predictions: list[float] = []
    skipped_non_binary = 0

    for example in examples:
        label = example.get("label")
        if label not in (0, 1, 0.0, 1.0):
            skipped_non_binary += 1
            continue

        input_row = {
            key: value
            for key, value in example.items()
            if key != "label"
        }
        prediction = _assert_probability(predict_fn(input_row, labeled=[]))
        labels.append(int(label))
        predictions.append(prediction)

    return {
        "n": len(labels),
        "skipped_non_binary": skipped_non_binary,
        "negative_log_loss": _negative_log_loss(labels, predictions),
        "auc_roc": _auc_roc(labels, predictions),
    }


def _assert_probability(value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("predict_fn must return a finite numeric probability.")
    probability = float(value)
    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("predict_fn must return a finite probability in [0, 1].")
    return probability


def _negative_log_loss(labels: list[int], predictions: list[float]) -> float:
    if not labels:
        return float("nan")

    eps = 1e-7
    total = 0.0
    for label, prediction in zip(labels, predictions):
        p = min(max(prediction, eps), 1.0 - eps)
        total += label * math.log(p) + (1 - label) * math.log(1.0 - p)
    return total / len(labels)


def _auc_roc(labels: list[int], predictions: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    rows = sorted(zip(predictions, labels), key=lambda row: row[0])
    rank_sum_positive = 0.0
    rank = 1
    index = 0
    while index < len(rows):
        end = index + 1
        while end < len(rows) and rows[end][0] == rows[index][0]:
            end += 1

        average_rank = (rank + rank + (end - index) - 1) / 2.0
        positives_in_tie = sum(label for _, label in rows[index:end])
        rank_sum_positive += positives_in_tie * average_rank

        rank += end - index
        index = end

    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


if __name__ == "__main__":
    train_data, validation_data, test_data = load_split_data()
    print(f"Response parquet files: {train_data['num_response_files']}")
    print(f"train benchmarks: {train_data['benchmark_ids']}")
    if validation_data is not None:
        print(f"validation benchmarks: {validation_data['benchmark_ids']}")
    if test_data is not None:
        print(f"test benchmarks: {test_data['benchmark_ids']}")
    print("\nExample train rows:")
    for index, example in enumerate(train_data["examples"]):
        if index >= 3:
            break
        print(example)
