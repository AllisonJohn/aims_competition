from datasets import load_dataset, Features, Value
from huggingface_hub import HfApi
import pandas as pd
from collections import defaultdict

REPO_ID = "aims-foundations/measurement-db"

# Load registries
print("Loading registries...")
subjects = load_dataset(REPO_ID, data_files=["subjects.parquet"], split="train").to_pandas()
items = load_dataset(REPO_ID, data_files=["items.parquet"], split="train").to_pandas()
benchmarks = load_dataset(REPO_ID, data_files=["benchmarks.parquet"], split="train").to_pandas()

# Create lookups
subject_names = subjects.set_index('subject_id')['display_name'].to_dict()
item_contents = items.set_index('item_id')['content'].to_dict()
benchmark_names = benchmarks.set_index('benchmark_id')['name'].to_dict()

# Find all response files
print("\nFinding response files...")
api = HfApi()
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet", "manifest.yaml", ".gitattributes", "README.md", "DATA_FORMAT.md"}

all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
response_files = sorted([
    f for f in all_files 
    if f.endswith('.parquet') 
    and f not in REGISTRY_FILES
    and not f.endswith('_traces.parquet')  # Exclude trace files
])

print(f"Found {len(response_files)} response files:")
for f in response_files:
    print(f"  {f}")

# Response schema
response_schema = Features({
    "subject_id": Value("string"),
    "item_id": Value("string"),
    "benchmark_id": Value("string"),
    "trial": Value("int64"),
    "test_condition": Value("string"),
    "response": Value("float64"),
    "correct_answer": Value("string"),
    "trace": Value("string"),
})

# Collect statistics
benchmark_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
subject_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
condition_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

print("\nProcessing response files...")
for i, file in enumerate(response_files):
    print(f"  [{i+1}/{len(response_files)}] {file}")
    
    try:
        responses = load_dataset(
            REPO_ID,
            data_files=[file],
            features=response_schema,
            split="train"
        ).to_pandas()
        
        print(f"    Loaded {len(responses)} rows")
        
        for _, row in responses.iterrows():
            # Get identifiers
            subject_name = subject_names.get(row['subject_id'], 'Unknown')
            benchmark = row['benchmark_id']
            condition = row['test_condition'] if pd.notna(row['test_condition']) else 'none'
            label = int(row['response']) if pd.notna(row['response']) else 0
            
            # Update stats
            benchmark_stats[benchmark]['total'] += 1
            benchmark_stats[benchmark]['correct'] += label
            
            subject_stats[subject_name]['total'] += 1
            subject_stats[subject_name]['correct'] += label
            
            condition_stats[condition]['total'] += 1
            condition_stats[condition]['correct'] += label
            
    except Exception as e:
        print(f"    Error: {e}")

# Print results
print("\n" + "="*60)
print("BENCHMARK STATISTICS")
print("="*60)
print(f"{'Benchmark':<20} {'Total':<10} {'Pass Rate':<10}")
print("-"*60)
for bench in sorted(benchmark_stats.keys()):
    stats = benchmark_stats[bench]
    rate = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    print(f"{bench:<20} {stats['total']:<10} {rate:.3f}")

print("\n" + "="*60)
print("CONDITION STATISTICS")
print("="*60)
print(f"{'Condition':<20} {'Total':<10} {'Pass Rate':<10}")
print("-"*60)
for cond in sorted(condition_stats.keys()):
    stats = condition_stats[cond]
    rate = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    print(f"{cond:<20} {stats['total']:<10} {rate:.3f}")

print("\n" + "="*60)
print("TOP 30 SUBJECTS BY PASS RATE (min 100 examples)")
print("="*60)
print(f"{'Subject':<35} {'Total':<10} {'Pass Rate':<10}")
print("-"*60)
subject_rates = []
for subj, stats in subject_stats.items():
    if stats['total'] >= 100:
        rate = stats['correct'] / stats['total']
        subject_rates.append((subj, stats['total'], rate))
subject_rates.sort(key=lambda x: x[2], reverse=True)
for subj, total, rate in subject_rates[:30]:
    print(f"{subj:<35} {total:<10} {rate:.3f}")

print("\n" + "="*60)
print("BOTTOM 30 SUBJECTS BY PASS RATE (min 100 examples)")
print("="*60)
print(f"{'Subject':<35} {'Total':<10} {'Pass Rate':<10}")
print("-"*60)
for subj, total, rate in subject_rates[-30:]:
    print(f"{subj:<35} {total:<10} {rate:.3f}")

# Save lookup tables for model.py
print("\n" + "="*60)
print("GENERATING LOOKUP TABLES FOR model.py")
print("="*60)

benchmark_rates = {
    bench: stats['correct'] / stats['total']
    for bench, stats in benchmark_stats.items()
    if stats['total'] > 0
}

subject_rates_dict = {
    subj: stats['correct'] / stats['total']
    for subj, stats in subject_stats.items()
    if stats['total'] >= 50  # Minimum examples
}

print("\n# Copy this into your model.py:")
print("\nBENCHMARK_RATES = {")
for bench, rate in sorted(benchmark_rates.items()):
    print(f"    '{bench}': {rate:.4f},")
print("}")

print("\nSUBJECT_RATES = {")
count = 0
for subj, rate in sorted(subject_rates_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"    '{subj}': {rate:.4f},")
    count += 1
    if count >= 100:  # Top 100
        print("    # ... truncated")
        break
print("}")

print("\nCONDITION_RATES = {")
for cond, stats in sorted(condition_stats.items()):
    if stats['total'] > 0:
        rate = stats['correct'] / stats['total']
        print(f"    '{cond}': {rate:.4f},")
print("}")

# Overall stats
total = sum(s['total'] for s in benchmark_stats.values())
correct = sum(s['correct'] for s in benchmark_stats.values())
if total > 0:
    print(f"\nOVERALL: {correct}/{total} = {correct/total:.4f}")
else:
    print("\nNo data loaded!")