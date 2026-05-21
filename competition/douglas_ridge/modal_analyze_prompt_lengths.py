from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("cs321m-analyze-prompt-lengths")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "pyarrow", "transformers", "huggingface_hub", "hf_xet")
)

RESPONSE_FILES = [
    "afrimedqa.parquet",
    "agentdojo.parquet",
    "ai2d_test.parquet",
    "androidworld.parquet",
    "bfcl.parquet",
    "cybench.parquet",
    "hle.parquet",
    "livecodebench.parquet",
    "matharena.parquet",
    "mathvista_mini.parquet",
    "mmbench_v11.parquet",
    "mmlupro.parquet",
    "mtbench.parquet",
    "rewardbench.parquet",
    "swebench.parquet",
    "ultrafeedback.parquet",
]


@app.function(image=image, cpu=4.0, memory=16384, timeout=60 * 60)
def analyze_remote(
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    item_limit: int = 0,
    batch_size: int = 1024,
) -> dict:
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    repo_id = "aims-foundations/measurement-db"
    tokenizer = AutoTokenizer.from_pretrained(encoder_id)

    items_path = hf_hub_download(repo_id, "items.parquet", repo_type="dataset")
    items = pd.read_parquet(items_path, columns=["item_id", "content"])
    item_lookup = dict(zip(items["item_id"], items["content"]))

    parts = []
    for filename in RESPONSE_FILES:
        path = hf_hub_download(repo_id, filename, repo_type="dataset")
        df = pd.read_parquet(path, columns=["item_id", "benchmark_id"])
        df = df.drop_duplicates(["benchmark_id", "item_id"])
        df["item_content"] = df["item_id"].map(item_lookup).fillna("")
        parts.append(df[["benchmark_id", "item_id", "item_content"]])
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["benchmark_id", "item_id"])
    if item_limit > 0 and len(df) > item_limit:
        df = df.sample(n=item_limit, random_state=326)

    token_lengths = []
    texts = df["item_content"].fillna("").astype(str).tolist()
    for start in range(0, len(texts), batch_size):
        batch = [
            f"Represent this evaluation question for difficulty prediction: {text}"
            for text in texts[start : start + batch_size]
        ]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False)
        token_lengths.extend(len(ids) for ids in encoded["input_ids"])
        if start and start % (batch_size * 25) == 0:
            print(f"tokenized {start}/{len(texts)} items", flush=True)

    df["token_len"] = token_lengths
    df["char_len"] = [len(text) for text in texts]
    df["word_len"] = [len(text.split()) for text in texts]

    def summarize(group: pd.DataFrame) -> dict:
        token = group["token_len"].to_numpy()
        char = group["char_len"].to_numpy()
        return {
            "n_items": int(len(group)),
            "token_mean": float(token.mean()),
            "token_p50": float(np.percentile(token, 50)),
            "token_p75": float(np.percentile(token, 75)),
            "token_p90": float(np.percentile(token, 90)),
            "token_p95": float(np.percentile(token, 95)),
            "token_p99": float(np.percentile(token, 99)),
            "pct_gt_256": float((token > 256).mean()),
            "pct_gt_384": float((token > 384).mean()),
            "pct_gt_512": float((token > 512).mean()),
            "char_p50": float(np.percentile(char, 50)),
            "char_p90": float(np.percentile(char, 90)),
            "char_p99": float(np.percentile(char, 99)),
        }

    by_benchmark = {
        str(benchmark): summarize(group)
        for benchmark, group in df.groupby("benchmark_id")
    }
    overall = summarize(df)
    overall["encoder_id"] = encoder_id
    overall["item_limit"] = int(item_limit)
    return {"overall": overall, "by_benchmark": by_benchmark}


@app.local_entrypoint()
def main(
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    item_limit: int = 0,
    batch_size: int = 1024,
) -> None:
    root = Path(__file__).resolve().parent
    out_path = root / "artifacts" / "prompt_length_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = analyze_remote.remote(encoder_id, item_limit, batch_size)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2, sort_keys=True))
    print(f"Wrote {out_path}")
