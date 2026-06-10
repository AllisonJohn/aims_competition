# torch_measure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Discord](https://img.shields.io/badge/Discord-join%20chat-5865F2.svg)](https://discord.gg/F6xbEwvvhb)

**PyTorch-native toolkit for predictive evaluation of AI systems.**

Benchmark scores increasingly gate deployment decisions but rarely predict how a model will behave in production. `torch_measure` treats evaluation itself as a predictive modeling problem: latent-variable models infer a system's capability directly from sparse benchmark observations and predict its performance on unseen tasks. Built on PyTorch, with GPU-accelerated IRT, factor models, amortized inference, adaptive testing, and tabular baselines.

## Predictive Evaluation Competition: Exact-Root K3 Submission

Our strongest competition submission was the **K=3 BGE ridge IRT model with exact-root labeled residual calibration**. The submitted artifact is:

```text
final-submission.zip
```

The zip is self-contained at the submission root:

```text
artifacts/baseline_stats.json
artifacts/bge_kfactor_ridge_artifact.json
artifacts/bge_kfactor_ridge_summary.json
model.py
models.txt
requirements.txt
```

### Pipeline Summary

The model has two stages: an offline trained base predictor and a test-time calibration layer that uses the small labeled set supplied by the competition.

**1. Offline K-factor IRT training.** We fit a K=3 item-response model on public response data:

```text
logit P(Y_ij = 1) = item_bias_j + subject_factor_i dot item_factor_j
```

This gives each model/subject a three-dimensional capability vector and each item a three-dimensional loading vector plus a bias/difficulty term.

**2. Amortized item prediction from text.** For unseen items, we cannot look up trained item parameters, so the training script encodes item text with `BAAI/bge-large-en-v1.5`. Ridge heads then predict:

```text
item_factor_1, item_factor_2, item_factor_3, item_bias
```

from the BGE embedding plus small text-shape features. This lets the submission estimate IRT parameters for new benchmark items at inference time.

**3. Base smoothed predictor.** The model also keeps smoothed empirical rates by subject, condition, and benchmark. The initial fallback logit is:

```text
base_logit =
  0.68 * subject_logit
+ 0.22 * condition_logit
+ 0.10 * benchmark_logit
+ item_text_adjustment
```

This gives a robust fallback if the encoder cannot load, and it also anchors the neural/text-based signal.

**4. Final learned calibration.** When item latents are available, the raw IRT factor logit is:

```text
factor_logit = item_bias + subject_factor dot predicted_item_factors
```

The submitted model applies a learned logistic/ridge-style calibrator over:

```text
subject_logit
condition_logit
benchmark_logit
item_adjustment
factor_logit
subject_logit * factor_logit
condition_logit * factor_logit
benchmark_logit * factor_logit
abs(factor_logit)
```

The result of this stage is the raw prediction before using the competition-provided labels.

### Exact-Root Labeled Residual Calibration

The live competition provides a tiny labeled set for the current hidden benchmark. The exact-root idea was to use those labels only as an inference-time residual correction, not to retrain the base model.

For a target row, the model computes same-benchmark residual shifts at three granularities:

1. **Benchmark shift:** all labeled examples from the same benchmark.
2. **Category shift:** labeled examples from the same benchmark and inferred category.
3. **Subject shift:** labeled examples from the same benchmark and parsed subject name.

Categories are inferred by lightweight benchmark/keyword rules such as `math`, `code`, `visual`, `medical`, `agent_tool`, `preference`, and `general`.

Each shift uses the same empirical-Bayes formula:

```text
prior_rate = mean(raw_model_prediction on matching labeled examples)
observed_rate = mean(observed labels on matching labeled examples)

posterior_rate =
  (prior_rate * prior_count + observed_rate * n_labels)
  / (prior_count + n_labels)

delta = logit(posterior_rate) - logit(prior_rate)
delta = clamp(delta, -cap, cap)
```

The exact-root submitted settings were:

```text
benchmark_delta: prior_count = 11, cap = 0.35
category_delta:  prior_count = 8,  cap = 0.18
subject_delta:   prior_count = 6,  cap = 0.16
```

The final prediction applies a weighted logit residual:

```text
total_delta = benchmark_delta + 0.65 * category_delta + 0.45 * subject_delta
final_p = sigmoid(logit(raw_p) + total_delta)
```

### Why This Helped

The base K=3 model already captured broad model ability, item difficulty, and item-skill interactions. The remaining error on hidden benchmarks was often local: a hidden benchmark could be globally easier/harder than expected, or specifically shifted for a category such as code, visual reasoning, or preference judgment. Exact-root used the five labeled examples as a small benchmark-specific survey and shrank the correction heavily to avoid overreacting.

The best live variants were the exact-root residual versions around `-0.58`. More aggressive category-only, subject-only, neural residual, ensemble, refit, and active-labeling variants were usually worse. The main lesson is that the labeled examples helped most when used as a conservative same-benchmark residual, especially with category/subject matching as a small add-on.

## Installation

With **pip**:

```bash
pip install torch_measure
```

With **[uv](https://docs.astral.sh/uv/)** (faster; drop-in replacement for pip):

```bash
uv pip install torch_measure        # into the active environment
uv add torch_measure                # into a uv-managed project
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details, or drop by our [Discord](https://discord.gg/F6xbEwvvhb) to chat.
