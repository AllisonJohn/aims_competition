# Best Submission Snapshot

This folder collects the files for the strongest Exact-Root K3 submission used in the Predictive Evaluation Competition report.

## Primary Submission Zip

```text
submission_zip/ridge-k3-labeled-type-residual-exact-root.zip
```

This is the self-contained competition upload. It contains:

```text
artifacts/baseline_stats.json
artifacts/bge_kfactor_ridge_artifact.json
artifacts/bge_kfactor_ridge_summary.json
model.py
models.txt
requirements.txt
```

## Exact-Root Model Package

```text
exact_root_model/
```

This is the unpacked/source copy of the exact-root inference package:

```text
exact_root_model/model.py
exact_root_model/models.txt
exact_root_model/requirements.txt
exact_root_model/artifacts/
```

The important inference behavior lives in `model.py`: K=3 BGE ridge IRT prediction followed by same-benchmark labeled residual calibration.

## Training Scripts

```text
training_scripts/modal_train_kfactor_ridge.py
training_scripts/modal_sweep_leave_one_benchmark.py
```

These are the relevant base K=3 training and sweep scripts.

`modal_train_kfactor_ridge.py` is the K3 training script we still have. It was copied from:

```text
competition/douglas_ridge_kfactor_k3/modal_train_kfactor_ridge.py
```

`modal_sweep_leave_one_benchmark.py` is the leave-one-benchmark sweep script used for hyperparameter checks.

Exact-root itself does not have a separate training script because it is a post-hoc inference-time calibration layer on top of the trained K=3 ridge artifact.

## Validation Sandbox

```text
validation_sandbox/
```

This contains the core scripts we used for local/sandbox validation of labeling and calibration ideas:

```text
validation_sandbox/README.md
validation_sandbox/validate_labeling.py
validation_sandbox/tune_meta_calibration.py
validation_sandbox/modal_validate_labeling.py
```

## Method Writeup

### Overview

Our best submission was a **three-stage statistical model** with a small test-time calibration layer:

1. Fit a K=3 IRT/factor model on historical model-item responses.
2. Use ridge regression to predict the item-side IRT vectors from item text embeddings.
3. Use logistic regression as the final probability model over IRT logits and empirical-rate features.
4. At test time, apply exact-root labeled residual calibration using the few labeled examples supplied by the competition.

In other words, the core model is not just a text embedding model and not just an IRT model. It first learns an IRT latent space, then learns how to map unseen item text into that latent space, then learns a logistic calibration layer that turns those components into probabilities.

### 1. Fit the K=3 IRT Latent Space

The first stage fits a K=3 factorized IRT model on public response data. Each training example is a binary label for whether a subject/model succeeded on an item. The fitted model is:

```text
logit P(Y_ij = 1) = item_bias_j + subject_factor_i dot item_factor_j
```

Here:

- `subject_factor_i` is a three-dimensional latent capability vector for a model/subject.
- `item_factor_j` is a three-dimensional latent skill/difficulty loading vector for an item.
- `item_bias_j` is an item-level bias/difficulty term.

The K3 training script optimizes binary cross entropy with a small L2 penalty on the subject and item factors:

```text
loss = BCEWithLogitsLoss(logit_ij, y_ij)
     + irt_l2 * (mean(subject_factor_i^2) + mean(item_factor_j^2))
```

The submitted artifact used:

```text
latent_dim = 3
irt_epochs = 10
irt_lr = 0.003
irt_l2 = 0.001
optimizer = AdamW(weight_decay=0)
train_rows = 1,086,835
subjects = 909
items = 88,886
```

This stage gives the model a compact interaction representation. A subject can be strong on one latent factor and weak on another, and each item can load differently on those factors.

### 2. Ridge Regression From Item Text to IRT Item Parameters

At test time, hidden benchmark items are unseen, so we cannot directly look up their learned IRT item parameters. To solve that, the training script embeds item text using:

```text
BAAI/bge-large-en-v1.5
```

For each known training item, we already have fitted IRT targets:

```text
item_factor_1
item_factor_2
item_factor_3
item_bias
```

The model trains one ridge regression head per target:

```text
ridge_factor_1: BGE(item_text) -> standardized item_factor_1
ridge_factor_2: BGE(item_text) -> standardized item_factor_2
ridge_factor_3: BGE(item_text) -> standardized item_factor_3
ridge_bias:     BGE(item_text) -> standardized item_bias
```

The ridge targets are standardized before fitting and unstandardized at inference time:

```text
target_z = (target - target_mean) / target_std
prediction = ridge(BGE(item_text)) * target_std + target_mean
```

The submitted artifact used:

```text
ridge_alpha = 300
ridge_heads = 4
text_item_targets = 88,886
max_length = 256
```

This is the key amortization step: unseen item text is converted into approximate IRT item vectors and an item bias. Once we have those, we can compute an IRT-style logit even for a new benchmark item:

```text
factor_logit = predicted_item_bias
             + subject_factor dot predicted_item_factors
```

### 3. Smoothed Empirical Baselines

Alongside the IRT/ridge path, the model stores smoothed empirical rates from the public data:

- subject rate
- condition rate
- benchmark rate

The fallback baseline logit is:

```text
base_logit =
  0.68 * subject_logit
+ 0.22 * condition_logit
+ 0.10 * benchmark_logit
+ item_text_adjustment
```

This provides a stable non-neural baseline. It is especially useful for subjects or conditions whose historical average is already highly informative, and it gives a fallback if the text encoder cannot load.

### 4. Logistic Regression Final Layer

The final base probability is produced by logistic regression. Its inputs are the smoothed empirical logits, the item-text adjustment, the K3 IRT factor logit, and interactions between the empirical logits and the factor logit:

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

Before logistic regression, these nine features are standardized with `StandardScaler`. Then an L2 logistic regression is fit:

```text
p_base = sigmoid(
    intercept
  + beta dot standardized_features
)
```

The submitted artifact used:

```text
LogisticRegression(C=1.0, penalty="l2", solver="lbfgs")
calibrator_features = 9
logit_cap = 4.0
```

This layer is important because the raw IRT logit is not used blindly. The logistic regression learns how much to trust subject history, condition history, benchmark history, item text difficulty, and the K3 interaction signal.

### 5. Exact-Root Labeled Residual Calibration

The competition provides a small labeled set at prediction time. Exact-root uses those labels only as a residual correction. It does not retrain the base model.

For each target item, the model looks at labeled examples from the same benchmark and computes three residual shifts:

1. `benchmark_delta`: all labeled examples from the same benchmark.
2. `category_delta`: same benchmark and same inferred category.
3. `subject_delta`: same benchmark and same parsed subject name.

Categories are inferred with lightweight benchmark and keyword rules:

```text
math
code
visual
medical
agent_tool
preference
general
```

Each residual shift uses an empirical-Bayes update:

```text
prior_rate = mean(raw_model_prediction on matching labeled examples)
observed_rate = mean(observed labels on matching labeled examples)

posterior_rate =
  (prior_rate * prior_count + observed_rate * n_labels)
  / (prior_count + n_labels)

delta = logit(posterior_rate) - logit(prior_rate)
delta = clamp(delta, -cap, cap)
```

The exact-root settings were:

```text
benchmark_delta: prior_count = 11, cap = 0.35
category_delta:  prior_count = 8,  cap = 0.18
subject_delta:   prior_count = 6,  cap = 0.16
```

The final logit correction is:

```text
total_delta = benchmark_delta + 0.65 * category_delta + 0.45 * subject_delta
final_p = sigmoid(logit(raw_p) + total_delta)
```

### Why Exact-Root Helped

The base K=3 model captured broad model ability and item difficulty, but hidden benchmarks could still be locally shifted. Some benchmarks were globally easier or harder than expected; others were specifically shifted for categories such as code, visual reasoning, agent/tool use, or preference judging.

Exact-root treats the five labeled examples as a tiny same-benchmark survey. The empirical-Bayes shrinkage keeps the update conservative, while the category and subject matches let the correction be more specific than a single global benchmark shift.

In live submissions, exact-root variants reached about `-0.58`, beating the simpler benchmark-only Bayes calibration around `-0.59` and the original K3 ridge baseline around `-0.60`.

### What Did Not Help As Much

The following variants were tried and were generally worse or unstable:

- category-only residual calibration
- stronger category/subject weights
- neural residual heads
- ensemble blends with teammate or alternate models
- active/diverse label acquisition
- item refit/freeze-subject refit attempts
- global post-hoc scale or bias sweeps

The main lesson was that the label signal was useful but high variance. It worked best when used as a conservative same-benchmark residual, with category and subject corrections as smaller add-ons.

### Reproducibility Checks

These files compiled successfully when this snapshot was created:

```bash
python -m py_compile exact_root_model/model.py
python -m py_compile training_scripts/modal_train_kfactor_ridge.py
python -m py_compile training_scripts/modal_sweep_leave_one_benchmark.py
```

The primary submission zip can be inspected with:

```bash
unzip -l submission_zip/ridge-k3-labeled-type-residual-exact-root.zip
```

## Internal Ablation Plan

This section is for report preparation and teammate coordination. The goal is to identify which parts of the method were actually responsible for the final improvement, and which claims need evidence before we put them in the report.

### Priority 1. IRT Latent Dimension

Question: did K=3 matter, or was any low-dimensional IRT model enough?

Suggested ablations:

- K=1 Rasch-style model with the same ridge/logistic pipeline.
- K=2 factor model.
- K=3 final model.
- K=4 factor model.

Report angle: show whether the extra latent factors improved hidden-item transfer, or whether higher K overfit/noised up item-side regression.

Why this is highest priority: K is the central modeling choice. If K=3 beats K=1/K=2/K=4, it justifies the core factor-IRT framing of the submission.

### Priority 2. Item-Side Ridge Amortization

Question: how much did predicting item vectors from BGE text help beyond empirical rates?

Suggested ablations:

- Remove BGE/ridge item heads entirely; use only subject/condition/benchmark rates.
- Predict only `item_bias`, no item factors.
- Predict item factors but no `item_bias`.
- Use unstandardized vs standardized ridge targets.
- Sweep ridge alpha, especially around `100`, `300`, `1000`.

Report angle: this isolates the contribution of text-to-IRT amortization.

Why this is high priority: this is the bridge from training items to hidden items. Without this step, the IRT item factors do not transfer to unseen benchmark questions.

### Priority 3. Logistic Regression Final Layer

Question: did the logistic regression layer improve calibration over raw IRT blending?

Suggested ablations:

- Raw K3 IRT factor logit only.
- Fixed blend of `base_logit` and `factor_logit`.
- Logistic regression with only:
  - `subject_logit`
  - `condition_logit`
  - `benchmark_logit`
  - `factor_logit`
- Full 9-feature logistic regression.
- Remove interaction terms:
  - `subject_logit * factor_logit`
  - `condition_logit * factor_logit`
  - `benchmark_logit * factor_logit`
- Remove `abs(factor_logit)`.

Report angle: argue that logistic regression acts as a calibration/stacking layer that learns when to trust empirical history versus text-derived IRT.

Why this is high priority: this determines whether the final model is truly benefiting from a learned stacking/calibration layer rather than a hand-chosen blend.

### Priority 4. Exact-Root Labeled Residual Calibration

Question: what part of exact-root produced the jump from roughly `-0.59/-0.60` to `-0.58`?

Suggested ablations:

- No labeled calibration.
- Benchmark-only Bayes shift.
- Benchmark + category residual.
- Benchmark + subject residual.
- Benchmark + category + subject residual.
- Category-only residual.
- Subject-only residual.
- Sweep residual weights:
  - category weight around `0.45`, `0.65`, `0.85`
  - subject weight around `0.25`, `0.45`, `0.65`
- Sweep prior counts:
  - benchmark prior around `8`, `10`, `11`
  - category prior around `6`, `8`
  - subject prior around `4`, `6`
- Sweep caps:
  - benchmark cap around `0.25`, `0.35`, `0.45`
  - category/subject caps around `0.12`, `0.18`, `0.24`

Report angle: labels are useful but high variance, so the main contribution is a conservative empirical-Bayes residual rather than aggressive refitting.

Why this is high priority: exact-root was the final leaderboard improvement. It is the most important post-hoc component to justify, even though it sits after the base model.

### Priority 5. Smoothed Empirical Baselines

Question: which empirical rates matter most?

Suggested ablations:

- Subject rate only.
- Subject + condition.
- Subject + benchmark.
- Subject + condition + benchmark.
- Vary smoothing constants:
  - subject alpha
  - condition alpha
  - benchmark alpha

Report angle: subject identity/model history is likely the strongest prior, while condition and benchmark provide weaker but useful context.

Why this is medium priority: empirical rates are strong baselines and should be shown, but they are less novel than K-factor IRT and text-to-item-vector amortization.

### Priority 6. Category Inference Rules

Question: did inferred categories help, or did they just add variance?

Suggested ablations:

- No categories.
- Coarse categories:
  - math
  - code
  - visual
  - medical
  - agent/tool
  - preference
  - general
- More granular categories, such as splitting `agentdojo`, `bfcl`, and `androidworld`.
- Keyword-only category inference.
- Benchmark-name-only category inference.

Report angle: category residuals help only if the category bins are broad enough for five labels to be meaningful.

Why this is medium priority: category matching is part of exact-root, but the exact rule details are less central than showing that category/subject residuals help at all.

### Priority 7. Label Acquisition Strategy

Question: did selecting labels actively help, or was random safer?

Suggested ablations:

- No custom `labeling.py`; use default/random acquisition.
- Uniform random with fixed seed.
- Uncertainty sampling near `p = 0.5`.
- Category-balanced sampling.
- Diversity sampling over inferred category/subject.
- Extreme-residual sampling.

Observed lesson so far: active/diverse label acquisition often hurt, likely because it biased the five labels away from the benchmark average. For the final report, be careful not to claim active learning helped unless validation supports it.

Why this is lower priority: our strongest results came from default/random-style labeled examples rather than active acquisition. This is still worth documenting because negative results explain why exact-root stayed conservative.

### Priority 8. Post-Hoc Probability Calibration

Question: could small global transformations improve log loss?

Suggested ablations:

- Global logit scale: `logit(p) * a`, with `a` near `0.95`, `0.97`, `1.03`.
- Global logit bias: `logit(p) + b`, with small `b` such as `+/- 0.02` or `+/- 0.05`.
- Apply scale/bias before exact-root.
- Apply scale/bias after exact-root.

Observed lesson so far: most global scale/bias sweeps did not improve live scores, suggesting the base calibration was already close and errors were more local.

Why this is lower priority: useful as a sanity check, but live experiments suggested global calibration was not the main bottleneck.

### Priority 9. Failed or Risky Directions to Mention Carefully

These are useful for internal context but probably should not be central claims unless we have clean validation:

- Neural residual heads: added complexity but did not beat exact-root.
- Refitting item/subject parameters from the tiny labeled set: unstable and some submissions failed.
- Pairwise-specific adapters: conceptually plausible but hard to activate reliably from hidden item format.
- Ensemble with teammate or alternate models: did not consistently improve.
- Qwen/LLM-judge-style features: expensive/fragile and did not beat the simpler BGE ridge setup in time.

### Minimum Ablation Table for Report

If we only have space/time for one ablation table, use:

| Variant | Purpose |
| --- | --- |
| Empirical baseline only | Tests non-IRT baseline strength |
| K3 IRT raw/fixed blend | Tests latent factor value |
| K3 + BGE ridge item heads | Tests text-to-IRT amortization |
| K3 + ridge + logistic calibrator | Tests final stacking/calibration |
| + benchmark Bayes labels | Tests simple labeled calibration |
| + exact-root category/subject residuals | Tests final contribution |

This sequence tells the cleanest story: empirical priors -> IRT interactions -> text amortization -> logistic calibration -> labeled residual adaptation.
