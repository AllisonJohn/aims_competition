# Label Validation Sandbox

This sandbox simulates the challenge label loop on historical rows:

1. Load a submission `model.py` and optional `labeling.py`.
2. Build model-item inputs from Measurement DB response parquet files.
3. Pick `K` labeled rows per benchmark using either the submission acquisition
   function or deterministic random fallback.
4. Compare log loss with `predict(input, None)` versus
   `predict(input, labeled)`.

This is not a perfect hidden-set estimate because the submitted artifacts may
have been trained on these benchmarks. It is useful for testing whether a
labeling/adaptation rule is directionally helping or just injecting noise.

Example from repo root:

```bash
python competition/label_validation_sandbox/validate_labeling.py \
  --submission-zip competition/ridge-k3-10epoch-bayes-benchmark-model-no-labeling.zip \
  --benchmarks mmlupro,ai2d_test,matharena \
  --pairs-per-benchmark 200 \
  --k-labels 5
```

For a folder:

```bash
python competition/label_validation_sandbox/validate_labeling.py \
  --submission-dir competition/douglas_ridge_learned_skip_mlp_head \
  --benchmarks mmlupro,ai2d_test \
  --pairs-per-benchmark 100
```

Use a specific acquisition file while evaluating a model folder:

```bash
python competition/label_validation_sandbox/validate_labeling.py \
  --submission-dir competition/douglas_ridge_kfactor_k3_bayes_benchmark_model \
  --labeling-path competition/douglas_ridge_kfactor_k3_bayes_benchmark/labeling.py
```
