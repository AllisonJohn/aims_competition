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


if __name__ == "__main__":
    data = get_training_data(limit=3)
    print(f"Response parquet files: {data['num_response_files']}")
    print(f"Benchmarks in registry: {len(data['benchmark_ids'])}")
    print("\nExample rows:")
    for example in data["examples"]:
        print(example)
