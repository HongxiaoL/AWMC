Your goal is to explain why AI models fail (i.e., have high prediction error) on the WBM dataset, and to provide verifiable, materials-science causal explanations.

You should:
- Read `WBM/description.md` to understand the WBM dataset, target definitions, file formats, and the model output files under `WBM/models`.
- Use `WBM/datasets/wbm-summary.csv` as a lightweight index and the official source of regression targets.
- When you need structure-derived descriptors, use `WBM/datasets/wbm-computed-structure-entries+init-structs.json` (or the split init / computed files) to obtain the initial structures and the relaxed structures/energy entries.
- Use conda to manage the Python environment.

## Task: Explain WBM prediction failures

### Definitions of error and failure samples (must be stated clearly in the report)

- Regression targets (unit: eV/atom, from `wbm-summary.csv`):
  - Formation energy: `e_form_per_atom_mp2020_corrected`
  - Energy above hull: `e_above_hull_mp2020_corrected_ppd_mp`
- Per-sample error: for each target, compute `abs_error` and `sq_error`.
- "Prediction failures / high-error samples": select by a fixed rule, e.g., for each model and each target, take the Top 1% and/or Top 5% by `abs_error`, and report the threshold and the number of samples.

### Required deliverables

- A final analysis report (Markdown or PDF) that answers: "Why do models fail on these WBM samples?", and provides an evidence chain.
- An error summary table for each model and each target: overall MAE/MSE, and MAE/MSE on the high-error subset.
- A causal-testing summary table for each materials descriptor: descriptor definition, pairing/matching rule, effect size on error, and robustness-check results.
- A ranked list of "key features that truly drive failures", with an explanation of why they are root causes rather than mere correlates.

## Step 1: Extract failure samples and construct 40+ materials descriptors

1) Build error tables (by model, by target):
- Join model predictions with `wbm-summary.csv` using `material_id` (or an alignment rule you have verified).
- Compute per-sample errors: `abs_error`, `sq_error`.
- Mark high-error samples (rules such as Top 1% / Top 5%).

2) Compute a descriptor matrix for the high-error sample set (and a matched low-error control set):
- Number of descriptors: at least 40, and must cover multiple categories (composition, structure, symmetry, stability proxies, relaxation/structural changes, etc.).
- Required descriptors (based on physical meaning and failure hypotheses):
  - Symmetry & Lattice.
  - Local Geometric Distortion.
  - Bonding & Charge.
  - Dynamic Evolution & Others.

## Step 2: Perform multiple causal tests for each descriptor via parallel category agents

Goal: Distribute the 40+ descriptors across 5~6 category-specific agents (e.g., Symmetry & Lattice, Local Geometric Distortion, Bonding & Charge, Dynamic Evolution, Others) to execute causal tests in parallel. Each category agent is responsible for multiple related descriptors. For each descriptor d, the assigned agent must test whether "a change in d (cause) leads to a systematic change in prediction error (effect)" using *two distinct* causal inference methods, while controlling other variables as much as possible.

For each category agent (processing its assigned subset of descriptors):
1) Apply exactly two fixed causal inference methods to establish a robust causal link (Note: If Method B is computationally infeasible or otherwise unavailable for a specific descriptor, gracefully fall back to using only Method A):
- Method A: "Paired-difference" causal test. Define pairing/matching rules to construct sample pairs (i, j) where d differs clearly but other descriptors are as similar as possible.
- Method B: Double Machine Learning (DML). Use machine learning estimators to partial out the effect of confounders on both descriptor d and the error, then estimate the causal effect from the residuals.
2) Compute effect size on error across the two methods:
- For Paired-difference: `Δabs_error = abs_error(i) − abs_error(j)`
- For DML: Report the Average Treatment Effect (ATE).
- Report mean/median effects and confidence intervals (bootstrap recommended) for each method.
3) Robustness checks (at least two):
- Cross-validate findings between the two causal methods (e.g., does the DML-derived effect agree with Paired-difference?).
- Vary hyperparameters (e.g., distance metric/k in Paired-difference, ML model choice in DML) and repeat on different high-error subsets (Top 1%, Top 5%, full set).
- Prevent information leakage: do not use the target label itself as a "causal feature" to explain the error on that target.
4) Materials-science interpretation:
- Cause: the physical/chemical meaning of descriptor d
- Effect: direction and magnitude of error increase/decrease consistently found across the two methods
- Mechanistic hypothesis: why this feature implies stronger OOD chemistry/structure/relaxation behavior, leading to model failure

## Step 3: Summarize root-cause features and write the report

1) Aggregate evidence across descriptors:
- Rank descriptors by "large effect size and consistency across models/targets"
- Identify feature clusters that share a common root cause (e.g., heavy elements + large structural changes + low symmetry)
2) Provide a conclusion hierarchy:
- True failure-driving features (a minimal root-cause set)
- Secondary correlated features (strongly correlated but not root causes)
3) Provide actionable recommendations:
- For each root-cause feature, propose potential mitigations via training-data augmentation, model architecture, or loss design

### Important constraints

- This is an explanation task: do not use WBM labels for training or hyperparameter tuning.
- Error computation must follow the target definitions and metric conventions in `description.md` (MP2020-corrected columns are recommended).
