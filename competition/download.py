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
    benchmark_id = benchmark.get("benchmark_id") or row["benchmark_id"]

    return {
        "benchmark": benchmark_id,
        "condition": row["test_condition"] or "none",
        "subject_content": render_subject_content(subject, row["subject_id"]),
        "item_content": item.get("content"),
        "label": row["response"],
    }
    
from pathlib import Path
import json

out_path = Path(__file__).with_name("data") / "train.jsonl"
out_path.parent.mkdir(exist_ok=True)

with out_path.open("w") as f:
    for row in responses:
        example = to_training_example(row)
        f.write(json.dumps(example) + "\n")

print(f"Saved {len(responses)} examples to {out_path}")