# TASK 05 — Post-Hoc Analyses (τ Sensitivity, Statistical Comparison, Results Tables)

## Project Context

SLIM-Diff is a shared-bottleneck diffusion model for joint FLAIR MRI + lesion mask synthesis. The ICIP 2026 camera-ready rebuttal commits to three post-hoc analyses that require NO new training or generation — only re-computation on existing outputs:

1. **τ sensitivity analysis** (R1.5) — How does the mask binarisation threshold affect MMD-MF?
2. **Cross-fold statistical comparison** (R1.3 + R1.1/R2.2) — Is the shared bottleneck consistently better than the decoupled variant across all 3 folds?
3. **Camera-ready results tables** — LaTeX-formatted tables for the paper.

You consume the outputs of TASK-04 (per-cell metrics CSVs) and the raw continuous masks stored during generation (TASK-03).

---

## Scope

### Files you OWN (create)

```
src/diffusion/scripts/similarity_metrics/posthoc/__init__.py                  # NEW subpackage
src/diffusion/scripts/similarity_metrics/posthoc/tau_sensitivity.py           # NEW — τ sweep
src/diffusion/scripts/similarity_metrics/posthoc/cross_fold_comparison.py     # NEW — statistical tests
src/diffusion/scripts/similarity_metrics/posthoc/latex_tables.py              # NEW — table generation
src/diffusion/scripts/similarity_metrics/posthoc/cli.py                       # NEW — CLI entry point
```

### Files you READ but do NOT modify

```
src/diffusion/scripts/similarity_metrics/metrics/mask_morphology.py    # MaskMorphologyDistanceComputer
src/diffusion/scripts/similarity_metrics/statistics/comparison.py      # compute_cliffs_delta, existing tests
src/diffusion/scripts/similarity_metrics/fold_evaluation.py            # TASK-04 output structure
```

### Interface contracts

**From TASK-03 (generation):**
- Raw continuous masks (pre-binarisation) stored at:
  ```
  {results_root}/fold_{k}/{architecture}/replicas/replica_{r}.npz
  ```
  Each `.npz` contains `masks` array in continuous `[-1, 1]` range.

**From TASK-04 (evaluation):**
- `{output_dir}/fold_metrics.csv` — per-cell metrics:
  ```
  fold,architecture,kid_mean,kid_std,lpips_mean,lpips_std,mmd_mf_mean,mmd_mf_std
  ```
- `{output_dir}/summary_metrics.csv` — cross-fold aggregation.
- `{output_dir}/wasserstein_per_feature.csv` — per-feature Wasserstein distances.
- `{output_dir}/eval_sample_counts.json` — sample counts per cell.

**From TASK-02 (data pipeline):**
- Per-fold test CSVs at `{cache_dir}/fold_{k}/test.csv` for loading real masks.

---

## Detailed Requirements

### 1. τ Sensitivity Analysis (`tau_sensitivity.py`)

The rebuttal states:

> *"A sensitivity analysis of MMD-MF w.r.t. τ will be included in the camera-ready."*

Generated masks are in continuous `[-1, 1]`. The default binarisation threshold is `τ = 0.0` (corresponding to 0.5 after rescaling to `[0, 1]`). We sweep τ to show MMD-MF stability.

#### Specification

```python
@dataclass
class TauSensitivityResult:
    """Results for one (fold, architecture) cell across τ values."""
    fold: int
    architecture: str
    tau_values: list[float]
    mmd_mf_values: list[float]       # MMD-MF at each τ
    mmd_mf_stds: list[float]         # std from kernel subsampling
    n_lesions_detected: list[int]    # number of connected components found at each τ
    optimal_tau: float               # τ minimising MMD-MF
```

**Algorithm:**

1. Load raw continuous synthetic masks for a given `(fold, architecture)` cell.
2. Load real binary test masks for the same fold.
3. For each τ in `{-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3}`:
   a. Binarise synthetic masks: `binary = (continuous > τ).astype(float32)`, then map to `{-1, +1}` to match the real mask convention, OR keep as `{0, 1}` — be consistent with what `MaskMorphologyDistanceComputer` expects. **Check the existing module's convention.**
   b. Extract morphological features from binarised synthetic masks using `MorphologicalFeatureExtractor`.
   c. Compute MMD-MF against real mask features.
   d. Record the number of connected components detected (this reveals whether extreme τ values eliminate all lesions or create spurious ones).
4. Store results per cell and aggregate.

**Important detail:** The existing `MaskMorphologyDistanceComputer` in `metrics/mask_morphology.py` expects binary masks. Check the `extract_features_from_masks` method to see whether it expects `{0, 1}` or `{-1, +1}` values. The real masks from the dataset are in `{-1, +1}` (since the model output range is `[-1, 1]`), so synthetic masks should match after binarisation.

#### Output files

- `tau_sensitivity.csv`:
  ```
  fold,architecture,tau,mmd_mf_mean,mmd_mf_std,n_lesions
  0,shared,-0.3,...
  0,shared,-0.2,...
  ...
  ```
- `tau_sensitivity_summary.csv` (aggregated across folds):
  ```
  architecture,tau,mmd_mf_mean_across_folds,mmd_mf_std_across_folds,n_lesions_mean
  ```

#### Design note (favorable framing)

If MMD-MF is relatively flat around τ = 0.0 (which we expect — the model was trained with targets in `{-1, +1}`), the result demonstrates robustness of our threshold choice. If the shared model shows flatter sensitivity than the decoupled model, that is additional evidence of better-calibrated mask generation. Report both.

---

### 2. Cross-Fold Statistical Comparison (`cross_fold_comparison.py`)

The rebuttal commits to showing that the shared bottleneck outperforms the decoupled variant. With 3 folds, we have paired observations.

#### Statistical tests

Given the small sample size (n=3 paired observations), the analysis must be primarily descriptive. Implement the following hierarchy:

**Level 1 — Descriptive:**
- Mean ± std of each metric across folds, per architecture.
- Per-fold difference: `Δ_k = metric_decoupled_k - metric_shared_k` for each fold k.
- Mean Δ ± std across folds.

**Level 2 — Effect size:**
- Cliff's δ between the two architectures (use `compute_cliffs_delta` from `statistics/comparison.py`). With n=3, this is coarse but interpretable.
- Cohen's d on the paired differences: `d = mean(Δ) / std(Δ)`.

**Level 3 — Directional consistency:**
- Sign test: are all 3 folds consistent in direction (shared < decoupled for KID/LPIPS/MMD-MF, since lower is better)?
- Under H₀ (random direction), P(all 3 same sign) = 2 × (0.5)³ = 0.25. This is not significant at α=0.05, so frame it as "consistent direction" rather than "statistically significant."
- If all 3 folds show the same direction, report this explicitly.

**Level 4 — Wilcoxon signed-rank (if applicable):**
- With n=3, the minimum achievable p-value for the Wilcoxon signed-rank test is 0.25 (when all 3 differences have the same sign). So this is marginal. Compute and report it, but do NOT claim significance unless p < 0.05.
- For the paper, lean on descriptive statistics + effect sizes + directional consistency.

#### Early-stopping epoch comparison

If TASK-03 logs the early-stopping epoch per run (it should, from the `ModelCheckpoint` callback), load these and report:
- Mean early-stopping epoch per architecture across folds.
- If the decoupled model stops earlier (overfits faster), this supports the regularisation hypothesis.

#### Output files

- `cross_fold_comparison.json`:
  ```json
  {
    "metrics": {
      "kid": {
        "shared": {"mean": 0.013, "std": 0.002, "per_fold": [0.012, 0.014, 0.013]},
        "decoupled": {"mean": 0.045, "std": 0.008, "per_fold": [0.038, 0.052, 0.045]},
        "delta_mean": 0.032,
        "delta_std": 0.007,
        "cliffs_delta": 1.0,
        "cohens_d": 4.57,
        "all_folds_consistent": true,
        "wilcoxon_p": 0.25
      },
      "lpips": { ... },
      "mmd_mf": { ... }
    },
    "early_stopping_epochs": {
      "shared": {"mean": 150, "per_fold": [145, 155, 150]},
      "decoupled": {"mean": 95, "per_fold": [90, 100, 95]}
    }
  }
  ```

---

### 3. LaTeX Table Generation (`latex_tables.py`)

Generate publication-ready LaTeX tables for the camera-ready paper. Two tables:

#### Table A — Ablation: Shared vs. Decoupled (NEW for camera-ready)

```latex
\begin{table}[t]
\centering
\caption{Shared vs.\ decoupled bottleneck ($x_0$-prediction, $L_{\gamma=1.5}$).
Mean $\pm$ std across 3 stratified folds. Lower is better.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Architecture & KID $\downarrow$ & LPIPS $\downarrow$ & MMD-MF $\downarrow$ \\
\midrule
Decoupled   & $X.XXX \pm X.XXX$ & $X.XXX \pm X.XXX$ & $X.XX \pm X.XX$ \\
Shared (ours) & $\mathbf{X.XXX \pm X.XXX}$ & $\mathbf{X.XXX \pm X.XXX}$ & $\mathbf{X.XX \pm X.XX}$ \\
\bottomrule
\end{tabular}
\end{table}
```

**Logic:** bold the better value per column (lower is better). If the decoupled is accidentally better on some metric, bold that one honestly.

#### Table B — Updated Table 1 with cross-fold stability (replaces original Table 1)

Same structure as the original Table 1 (prediction type × Lp norm), but now showing `mean ± std` where the `±` reflects cross-fold variance rather than cross-replica variance. Only for the best config (x₀, L₁.₅) since only that config is retrained across folds. For the remaining 8 configurations (ε × 3 Lp + v × 3 Lp), keep the original single-split numbers with a footnote.

#### Table C — τ sensitivity (inline or small table)

```latex
% Inline version for space:
MMD-MF varies from X.XX ($\tau{=}{-}0.2$) to X.XX ($\tau{=}0.2$),
with the minimum at $\tau{=}0.0$ (X.XX), confirming low sensitivity
to the binarisation threshold.

% Or as a small table if space allows.
```

#### Output

Write `.tex` files to `{output_dir}/tables/`:
- `table_ablation.tex`
- `table_main_updated.tex`
- `table_tau_sensitivity.tex`

Each file is a standalone `\begin{table}...\end{table}` block that can be `\input{}` into the paper.

---

## CLI Entry Point (`posthoc/cli.py`)

```bash
# Run all post-hoc analyses
python -m src.diffusion.scripts.similarity_metrics.posthoc.cli \
    --fold-metrics /path/to/fold_metrics.csv \
    --results-root /path/to/results \
    --cache-dir /path/to/slice_cache \
    --output-dir /path/to/posthoc_output

# Run only τ sensitivity
python -m src.diffusion.scripts.similarity_metrics.posthoc.cli \
    --only tau_sensitivity \
    --results-root /path/to/results \
    --cache-dir /path/to/slice_cache \
    --output-dir /path/to/posthoc_output

# Run only statistical comparison (no GPU needed)
python -m src.diffusion.scripts.similarity_metrics.posthoc.cli \
    --only comparison \
    --fold-metrics /path/to/fold_metrics.csv \
    --output-dir /path/to/posthoc_output

# Generate LaTeX tables only
python -m src.diffusion.scripts.similarity_metrics.posthoc.cli \
    --only tables \
    --fold-metrics /path/to/fold_metrics.csv \
    --output-dir /path/to/posthoc_output
```

The `comparison` and `tables` sub-commands require only the CSV from TASK-04 and run on CPU. The `tau_sensitivity` sub-command needs access to raw `.npz` files and real masks.

---

## Acceptance Criteria (Testable on 8 GB VRAM)

### AC-1: τ sensitivity on dummy data
Create 50 synthetic continuous masks in `[-1, 1]` with known lesion regions.
Create 50 binary real masks with matching lesion statistics.
Run τ sweep for `{-0.2, -0.1, 0.0, 0.1, 0.2}`.
Assert: `tau_sensitivity.csv` has `5 × 1 × 1 = 5` rows.
Assert: MMD-MF at τ=0.0 is finite and ≥ 0.
Assert: `n_lesions` is > 0 for τ ∈ `{-0.1, 0.0, 0.1}` (extreme thresholds may eliminate all lesions, which is acceptable).

### AC-2: Cross-fold comparison with synthetic CSV
Create a dummy `fold_metrics.csv` with 6 rows (3 folds × 2 architectures).
Set shared values consistently lower than decoupled.
Run comparison.
Assert: `cross_fold_comparison.json` exists and contains all expected fields.
Assert: `all_folds_consistent` is `true` for all metrics.
Assert: `cliffs_delta` is 1.0 (perfect separation) for the dummy data.
Assert: `cohens_d` is positive and finite.

### AC-3: Cross-fold comparison with tied data
Create a dummy `fold_metrics.csv` where shared wins on 2 folds but loses on 1.
Run comparison.
Assert: `all_folds_consistent` is `false`.
Assert: no crash, all fields still populated.

### AC-4: LaTeX table generation
Run `latex_tables.py` on the dummy `fold_metrics.csv`.
Assert: `table_ablation.tex` exists and contains `\begin{table}` and `\end{table}`.
Assert: `\mathbf` appears on the row with lower values.
Assert: the file compiles without error when wrapped in a minimal LaTeX document (use `pdflatex` if available, or just check syntax with regex for matched braces).

### AC-5: τ sweep consistency
The MMD-MF at τ=0.0 from the τ sensitivity analysis must match (within floating-point tolerance, `atol=0.01`) the MMD-MF reported in `fold_metrics.csv` for the same `(fold, architecture)` cell. Add an explicit assertion for this.

### AC-6: No modification of existing modules
Run `git diff --stat` after all changes. Assert that only files under `posthoc/` are new. No existing files are modified.

---

## Anti-Patterns

- **Do NOT re-implement Cliff's δ.** Import from `statistics/comparison.py`.
- **Do NOT re-implement morphological feature extraction.** Import from `metrics/mask_morphology.py`.
- **Do NOT hardcode metric values.** All numbers come from CSVs or computation.
- **Do NOT claim statistical significance with n=3.** Frame as "directional consistency" and effect sizes. The paper already uses Cliff's δ, so the framework is established.
- **Do NOT run τ sweep on training data.** Only test fold masks are used as reference.
- **Do NOT tune τ per architecture.** The same sweep range applies to both. If the optimal τ differs, report it as an observation, not as a method choice.
- **Do NOT generate tables with more decimal places than the original Table 1.** KID: 3 decimals. LPIPS: 3 decimals. MMD-MF: 2 decimals. Match the existing paper.
