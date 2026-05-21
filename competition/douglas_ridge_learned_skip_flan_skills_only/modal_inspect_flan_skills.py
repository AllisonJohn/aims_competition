from __future__ import annotations

import json

import modal


app = modal.App("cs321m-inspect-flan-skills")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "sentencepiece",
        "numpy",
        "pandas",
        "pyarrow",
        "huggingface_hub",
    )
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

SKILL_TAGS = [
    "math_proof",
    "algebra",
    "geometry",
    "exact_calculation",
    "coding",
    "debugging",
    "algorithms",
    "software_engineering",
    "visual_reasoning",
    "medical",
    "factual_qa",
    "preference_judgment",
    "safety",
    "security",
    "tool_use",
    "instruction_following",
    "long_context",
    "multi_step_reasoning",
    "commonsense",
    "domain_knowledge",
    "adversarial_prompting",
    "planning",
]


@app.function(gpu="H100", image=image, timeout=60 * 30)
def inspect_remote(
    n_per_benchmark: int = 2,
    flan_model_id: str = "google/flan-t5-base",
    encoder_id: str = "BAAI/bge-large-en-v1.5",
) -> dict:
    from collections import Counter

    import numpy as np
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

    repo_id = "aims-foundations/measurement-db"

    def parse_skill_tags(text: str) -> list[str]:
        lower = str(text or "").lower()
        normalized = lower.replace("-", "_").replace(" ", "_")
        found = [tag for tag in SKILL_TAGS if tag in normalized]
        return found[:6] or ["domain_knowledge"]

    items_path = hf_hub_download(repo_id, "items.parquet", repo_type="dataset")
    items = pd.read_parquet(items_path, columns=["item_id", "content"])
    item_lookup = dict(zip(items["item_id"], items["content"]))

    rows = []
    for filename in RESPONSE_FILES:
        path = hf_hub_download(repo_id, filename, repo_type="dataset")
        df = pd.read_parquet(path, columns=["item_id", "benchmark_id"])
        df = df.drop_duplicates("item_id")
        if len(df) > n_per_benchmark:
            df = df.sample(n=n_per_benchmark, random_state=326)
        for row in df.itertuples(index=False):
            content = str(item_lookup.get(row.item_id, ""))
            if not content.strip():
                continue
            rows.append(
                {
                    "benchmark": str(row.benchmark_id),
                    "item_id": str(row.item_id),
                    "content": content,
                }
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_id)
    flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_id).to(device)
    flan_model.eval()
    allowed = ", ".join(SKILL_TAGS)
    prompts = [
        (
            "Extract compact skill tags for predicting whether an LLM answers this evaluation item correctly. "
            "Return only comma-separated tags from this list: "
            f"{allowed}\n\nItem:\n{row['content']}\n\nTags:"
        )
        for row in rows
    ]
    raw_outputs = []
    with torch.no_grad():
        for start in range(0, len(prompts), 16):
            batch = prompts[start : start + 16]
            encoded = flan_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = flan_model.generate(
                **encoded,
                max_new_tokens=32,
                num_beams=1,
                do_sample=False,
            )
            raw_outputs.extend(flan_tokenizer.batch_decode(generated, skip_special_tokens=True))

    for row, raw in zip(rows, raw_outputs):
        row["flan_raw"] = raw
        row["tags"] = parse_skill_tags(raw)
        row["skill_text"] = "Required skills: " + ", ".join(row["tags"])

    bge_tokenizer = AutoTokenizer.from_pretrained(encoder_id)
    bge_model = AutoModel.from_pretrained(encoder_id).to(device)
    bge_model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            batch = [row["skill_text"] for row in rows[start : start + 32]]
            encoded = bge_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = bge_model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy().astype(np.float32))
    matrix = np.vstack(embeddings)
    similarities = matrix @ matrix.T
    nearest = []
    for idx, row in enumerate(rows):
        order = np.argsort(-similarities[idx])
        best = [int(j) for j in order if int(j) != idx][:3]
        nearest.append(
            {
                "benchmark": row["benchmark"],
                "tags": row["tags"],
                "neighbors": [
                    {
                        "benchmark": rows[j]["benchmark"],
                        "tags": rows[j]["tags"],
                        "cosine": float(similarities[idx, j]),
                    }
                    for j in best
                ],
            }
        )

    tag_counts = Counter(tag for row in rows for tag in row["tags"])
    return {
        "n_items": len(rows),
        "flan_model": flan_model_id,
        "encoder": encoder_id,
        "tag_counts": dict(tag_counts.most_common()),
        "samples": [
            {
                "benchmark": row["benchmark"],
                "item_id": row["item_id"],
                "item_preview": row["content"][:500].replace("\n", " "),
                "flan_raw": row["flan_raw"],
                "tags": row["tags"],
            }
            for row in rows[: min(len(rows), 24)]
        ],
        "nearest_neighbors": nearest[: min(len(nearest), 24)],
    }


@app.local_entrypoint()
def main(
    n_per_benchmark: int = 2,
    flan_model_id: str = "google/flan-t5-base",
    encoder_id: str = "BAAI/bge-large-en-v1.5",
) -> None:
    result = inspect_remote.remote(n_per_benchmark, flan_model_id, encoder_id)
    print(json.dumps(result, indent=2, sort_keys=True))
