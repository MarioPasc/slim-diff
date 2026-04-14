# ICIP 2026 Camera-Ready Rebuttal — Progress Tracker

## TASK-01: Decoupled-Bottleneck U-Net (COMPLETE)

**Status**: Done  
**Date**: 2026-04-13  
**Addresses**: R1.1 (missing baseline), R2.2 (shared-bottleneck ablation), R2.3 (within-framework comparison)

### Summary

Implemented `DecoupledMiddleBlock` as a drop-in replacement for MONAI's shared middle block. The decoupled variant splits the bottleneck into two independent half-channel paths (`path_a`, `path_b`) with no shared parameters, isolating the effect of the shared-vs-decoupled representation.

### Files Created/Modified

| File | Action |
|------|--------|
| `src/diffusion/model/decoupled_unet.py` | Created — core `DecoupledMiddleBlock` implementation |
| `src/diffusion/model/factory.py` | Modified — `bottleneck_mode` switch after model construction |
| `src/diffusion/model/__init__.py` | Modified — re-exports |
| `src/diffusion/config/validation.py` | Modified — validation for decoupled config |
| `src/diffusion/tests/test_decoupled_unet.py` | Created — 9 acceptance tests (all pass) |

### Parameter Matching

Best configuration for camera-ready (within 1% target):

```yaml
model:
  bottleneck_mode: "decoupled"
  decoupled_bottleneck:
    channels_per_path: null      # resolves to 128 (= channels[-1] // 2)
    extra_resnet_blocks: 2       # compensates param loss from halving
```

| Variant | Params | Delta |
|---------|--------|-------|
| Shared (baseline) | 26,894,210 | — |
| Decoupled (tuned) | 27,029,378 | +0.503% |

### Test Results

All 9 tests pass:
- AC-1: Forward pass shapes match
- AC-2: Parameter count within tolerance
- AC-3: Gradient flows through both independent paths
- AC-4: `path_a` and `path_b` hold disjoint parameters
- AC-5: Default config (omitting `bottleneck_mode`) yields stock MONAI block
- AC-6: Explicit `bottleneck_mode="shared"` preserves state_dict keys
- Bonus: Cross-attention path works with `with_conditioning=True`
- Bonus: Numerical sanity (no NaN/Inf)
- Bonus: Unknown `bottleneck_mode` rejected

### Usage (Picasso)

Add to any existing SLURM YAML:

```yaml
model:
  bottleneck_mode: "decoupled"
  decoupled_bottleneck:
    extra_resnet_blocks: 2
```

Run training as normal — no other changes required.

---

## TASK-02: K-Fold Data Pipeline (COMPLETE)

**Status**: Done  
**Date**: 2026-04-13  
**Addresses**: R1.3 (stability-across-splits evidence)

### Summary

Implemented `KFoldManager` for 3-fold stratified patient-level cross-validation. The test set is held **FIXED** across all folds (only train+val gets 3-folded), providing cross-fold variance from the training partition against a stable gold test set — which is what R1.3 actually requires.

Stratification uses binary `has_lesion` at the patient level (patient is positive if ≥1 slice has `has_lesion=True`). Since the cache was built with `drop_healthy_patients: true`, only epilepsy patients are present.

### Files Created/Modified

| File | Action |
|------|--------|
| `src/diffusion/data/kfold.py` | Created — `FoldAssignment`, `KFoldManager`, argparse CLI |
| `src/diffusion/data/splits.py` | Modified — added `create_kfold_splits()` bridge function |
| `src/diffusion/tests/test_kfold.py` | Created — 12-test pytest suite (all pass) |
| `pyproject.toml` | Modified — added `slimdiff-kfold` CLI entrypoint |

### Key Design Decisions

1. **Fixed test set**: Original spec had rotating test; user override keeps `test.csv` subjects identical across folds.
2. **No master.csv**: Pool built in-memory by concatenating `train.csv` + `val.csv`.
3. **Atomic writes**: Fold tree written to `folds.tmp/` then `os.replace` to `folds/` — no half-written state on crash.
4. **Split column rewritten**: Rows landing in a different split than their original have `row["split"]` updated for consistency.
5. **Symlinked priors**: `zbin_priors_brain_roi.npz` symlinked into each fold dir for training pipeline compatibility.

### Output Layout

```
{cache_dir}/folds/
├── folds_meta.json         # Metadata + reproducibility info
├── fold_0/
│   ├── train.csv           # Pool patients assigned to train for fold 0
│   ├── val.csv             # Pool patients assigned to val for fold 0
│   ├── test.csv            # FIXED across all folds
│   └── zbin_priors_brain_roi.npz -> ../../zbin_priors_brain_roi.npz
├── fold_1/{train,val,test}.csv
└── fold_2/{train,val,test}.csv
```

### Test Results

All 12 tests pass:
- `test_no_patient_leakage` — train/val/test pairwise disjoint per fold
- `test_fixed_test_set_across_folds` — test subjects identical across folds
- `test_every_pool_patient_in_exactly_one_val` — each pool patient in exactly one val
- `test_stratification_balance` — per-fold val lesion ratio within 20% of global
- `test_csv_columns_match` — 10 source columns preserved in fold CSVs
- `test_split_column_rewritten` — split column updated correctly
- `test_determinism` — same seed → identical folds
- `test_slice_counts_sum` — slice counts partition correctly
- `test_json_roundtrip` — `write_meta_json` → `from_meta_json` roundtrips
- `test_cli_idempotent` — re-running CLI exits 0, no-op
- `test_no_slice_appears_twice` — no duplicate slices within a fold
- `test_pool_derivation` — pool = all_subjects - test_subjects

### Usage (Picasso)

```bash
# Generate fold CSVs from existing cache
slimdiff-kfold \
    --cache-dir /mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/epilepsy/slice_cache \
    --n-folds 3 \
    --seed 42

# Train on a specific fold (TASK-03 will create SLURM scripts for this)
# Just point cfg.data.cache_dir at the fold directory:
# cfg.data.cache_dir = "{original_cache}/folds/fold_0"
```

No changes needed to `dataset.py`, `create_dataloader`, or training runners — each fold dir is a drop-in replacement for the original cache.

---

## TASK-03: Training Orchestration (COMPLETE)

**Status**: Done  
**Date**: 2026-04-14  
**Addresses**: R1.1 (missing baseline), R1.3 (stability across splits), R2.2 (shared-bottleneck ablation), R2.3 (within-framework comparison) — via orchestrating TASK-01 + TASK-02 into a 6-cell grid.

### Summary

Created the full SLURM scaffolding for the 3-fold × 2-architecture camera-ready grid (`shared_fold_{0,1,2}` and `decoupled_fold_{0,1,2}`). Each cell has its own `config.yaml` and `train_generate.sh` under `slurm/camera_ready/`, derived from two canonical templates at `configs/camera_ready/base_{shared,decoupled}.yaml`. A single `launch_camera_ready.sh` submits all 6 jobs independently.

### Files Created

| File | Action |
|------|--------|
| `configs/camera_ready/base_shared.yaml` | Created — canonical template (shared bottleneck) |
| `configs/camera_ready/base_decoupled.yaml` | Created — canonical template (decoupled bottleneck) |
| `slurm/camera_ready/shared_fold_0/config.yaml` | Created |
| `slurm/camera_ready/shared_fold_0/train_generate.sh` | Created — executable |
| `slurm/camera_ready/shared_fold_1/config.yaml` | Created |
| `slurm/camera_ready/shared_fold_1/train_generate.sh` | Created — executable |
| `slurm/camera_ready/shared_fold_2/config.yaml` | Created |
| `slurm/camera_ready/shared_fold_2/train_generate.sh` | Created — executable |
| `slurm/camera_ready/decoupled_fold_0/config.yaml` | Created |
| `slurm/camera_ready/decoupled_fold_0/train_generate.sh` | Created — executable |
| `slurm/camera_ready/decoupled_fold_1/config.yaml` | Created |
| `slurm/camera_ready/decoupled_fold_1/train_generate.sh` | Created — executable |
| `slurm/camera_ready/decoupled_fold_2/config.yaml` | Created |
| `slurm/camera_ready/decoupled_fold_2/train_generate.sh` | Created — executable |
| `slurm/camera_ready/launch_camera_ready.sh` | Created — executable |
| `slurm/camera_ready/README.md` | Created |
| `src/diffusion/tests/test_camera_ready_configs.py` | Created — 40-test acceptance suite (all pass) |

### Key Design Decisions

1. **Two canonical templates + 6 resolved per-cell configs**: `base_{arch}.yaml` carries the literal string `fold_K`; per-cell configs are produced by `cp` + `sed "s|fold_K|fold_{k}|g"`. Committing both makes future rebuttal sub-experiments trivial to derive.
2. **`decoupled_bottleneck` block in both templates**: the subtree is carried by the shared template too (where the factory ignores it). This makes Test 3 hold strictly — the two configs are byte-identical after popping `model.bottleneck_mode` + the allowed per-cell metadata fields.
3. **`SEED_BASE=42` shared across all 6 cells**: `generate_replicas.py` seeds x_T via SHA256 of `(seed_base, replica_id, zbin, lesion_present, domain_int, sample_index)`. Architecture is **not** in the hash, so shared-fold-k and decoupled-fold-k produce byte-identical x_T per matched (replica, zbin, domain, sample). TASK-06's paired qualitative figure depends on this.
4. **DDP fix**: camera-ready uses `training.devices: 2, strategy: "ddp"` (reference ICIP ablation had `devices: 1, strategy: "auto"` despite requesting 2 GPUs via SLURM). This halves wall-clock at the cost of doubling global batch size — flagged as a residual risk if strict hyperparameter-identical comparison to completed ablation runs becomes required later.
5. **Kfold CSVs: in-job idempotent guard**: SLURM script checks `${FOLD_CACHE_DIR}/train.csv`; if missing, runs `slimdiff-kfold --n-folds 3 --seed 42`. The CLI's atomic `folds.tmp/ → folds/` swap prevents races between concurrent jobs (verified by TASK-02's `test_cli_idempotent`).
6. **`slimdiff-*` CLI names**: `slimdiff-train`, `slimdiff-cache`, `slimdiff-kfold` (current `pyproject.toml` `[project.scripts]`). The reference SLURM used legacy `jsddpm-*` names.
7. **`sed`-patched paths at submit time**: committed YAML uses relative paths (`./outputs/...`, `./data/slice_cache/folds/fold_K`). The SLURM script rewrites `output_dir:` and `cache_dir:` to absolute Picasso mount paths on the compute node. Mirrors the reference ablation script's pattern.

### Configuration (all 6 cells share these values)

```yaml
scheduler:
  prediction_type: "sample"       # x0-prediction
  schedule: "cosine"
  num_train_timesteps: 1000
training:
  optimizer: {lr: 1.0e-4, weight_decay: 1.0e-4}
  max_epochs: 1000                # capped by early_stopping.patience=25
  ema.decay: 0.999
  gradient_clip_val: 1.0
loss:
  mode: "mse_lp_norm"
  lp_norm.p: 1.5
sampler:
  type: "DDIM", num_inference_steps: 300, eta: 0.2
```

The only differences across cells are (a) `model.bottleneck_mode` + `model.decoupled_bottleneck`, (b) fold id embedded in `experiment.name`, `experiment.output_dir`, `logging.logger.wandb.{name,tags,notes}`, `data.cache_dir`.

### Generation phase (held constant)

| Parameter | Value |
|---|---|
| `NUM_REPLICAS` | 20 |
| `N_SAMPLES_PER_MODE` | 150 (= 150 × 30 zbins × 2 domains = 9 000 / replica) |
| `GEN_BATCH_SIZE` | 32 |
| `SEED_BASE` | 42 (identical across all 6 cells) |
| `GEN_DTYPE` | `float16` |

Total generated samples per cell: **20 × 150 × 30 × 2 = 180 000** (90 000 per domain).

### Test Results

All 40 acceptance tests pass:

- `test_config_exists_and_parses[*]` (6) — all cell configs load via OmegaConf.
- `test_hyperparameter_consistency[*]` (11) — scheduler, optimizer, patience, EMA, clip, Lp, DDIM all identical across cells.
- `test_arch_configs_minimal_diff[*]` (3) — per fold, shared vs decoupled configs are byte-identical after popping the allowed fields.
- `test_slurm_script_syntax[*]` (6) + `test_launcher_script_syntax` (1) — bash `-n` passes, executable bit set.
- `test_output_dirs_unique` (1) — all 6 `experiment.output_dir` values distinct.
- `test_bottleneck_mode_present[*]` (6) — each cell's `model.bottleneck_mode` matches its arch label.
- `test_seed_base_is_42[*]` (6) — paired-sample guarantee enforced at test level.

### Usage (Picasso)

```bash
# From the repo root on Picasso (conda env `jsddpm` activated)
bash slurm/camera_ready/launch_camera_ready.sh
# Submits 6 independent jobs, all with --gres=gpu:2 --constraint=dgx --time=3-00:00:00.
# Monitor with: squeue -u $USER
```

The first job to run builds the base slice cache (if missing) and the 3-fold CSVs (idempotent, atomic). All subsequent jobs skip these steps.

### Residual risks

- GPU-hours budget: ~36 GPU-days training + ~60–120 GPU-hours generation for the full grid.
- `devices: 2` changes effective batch size vs the completed ICIP ablation (per-GPU batch doubles global batch). If strict hyperparameter-identical comparison to those runs is needed, halve `training.batch_size` in the camera-ready templates.
- If `generate_replicas.py` SHA256 ever changes to include architecture, the paired-sample guarantee (Design Decision 3) breaks — TASK-06 should re-verify before producing the qualitative figure.

---

## TASK-04: Fold-Aware Evaluation Pipeline (COMPLETE)

**Status**: Done
**Date**: 2026-04-14
**Addresses**: R1.3 (stability across splits), R2.3 (within-framework comparison) — consumes TASK-02 fold CSVs and TASK-03 replica outputs to emit cross-fold KID / LPIPS / MMD-MF evidence.

### Summary

Built a fold-aware similarity-metrics pipeline that processes the
`3 folds × 2 architectures = 6 cells` grid sequentially. Each cell loads the
per-fold `test.csv` + its cell's `replicas/replica_*.npz` files and runs KID,
LPIPS, and MMD-MF (+ per-feature Wasserstein). Memory is bounded by one cell's
float16 resident arrays (~9 GB at 180k × 160² samples) with transient float32
casts released between metrics. Partial-file + `--aggregate-only` flow
supports SLURM array execution for the 6 cells.

### Files Created / Modified

| File | Action |
|------|--------|
| `src/diffusion/scripts/similarity_metrics/data/fold_loaders.py` | Created — `FoldEvalData`, `load_fold_eval_data`, `resolve_cell_dir` |
| `src/diffusion/scripts/similarity_metrics/fold_evaluation.py` | Created — `compute_cell_metrics`, `run_single_cell`, `run_grid`, `save_outputs`, `aggregate_partials`, `main` |
| `src/diffusion/scripts/similarity_metrics/config/icip2026_camera_ready.yaml` | Created — nullable-path YAML config |
| `src/diffusion/scripts/similarity_metrics/cli.py` | Modified — `fold-eval` short-circuit to `fold_evaluation.main` |
| `src/diffusion/tests/test_fold_evaluation.py` | Created — 6-test acceptance suite (all pass in ~10 s) |

### Spec discrepancies resolved against TASK-02/03 reality

1. Fold manifest: `{cache_dir}/folds/folds_meta.json` via `KFoldManager.from_meta_json`, not `{results_root}/fold_assignments.json`. Test set is fixed across folds.
2. Fold CSV schema: `filepath, subject_id, z_index, z_bin, pathology_class, token, source, split, has_lesion, lesion_area_px`. A single `.npz` at `filepath` holds both `image` and `mask` keys; `filepath` is relative to the ORIGINAL cache root (not the fold dir).
3. Replica layout: `{results_root}/slimdiff_cr_{arch}_fold_{k}/replicas/replica_{r:03d}.npz` (flat per-cell).
4. Replica NPZ keys: `images, masks, zbin, domain` (singular), `(N,H,W)` float16 in [-1,1], masks continuous.

### Outputs

| File | Rows | Cols |
|------|------|------|
| `fold_metrics.csv` | 6 | `fold, architecture, kid_mean, kid_std, lpips_mean, lpips_std, mmd_mf_mean, mmd_mf_std, n_real, n_synth` |
| `summary_metrics.csv` | 2 | `architecture, kid_mean, kid_std_across_folds, lpips_mean, lpips_std_across_folds, mmd_mf_mean, mmd_mf_std_across_folds` |
| `wasserstein_per_feature.csv` | 6 | `fold, architecture, area, perimeter, circularity, solidity, extent, eccentricity, major_axis_length, minor_axis_length, equivalent_diameter, geometric_mean` |
| `eval_sample_counts.json` | — | `{cells:[{fold, architecture, n_real, n_synth, n_replicas}, ...], generated_at}` |

Aggregation across folds uses `np.nanmean` and `np.nanstd(ddof=0)`; per-fold MMD-MF NaN → logged warning, skipped from aggregation; all-NaN → NaN in summary.

### CLI

```bash
# Full grid
slimdiff-metrics fold-eval \
    --config src/diffusion/scripts/similarity_metrics/config/icip2026_camera_ready.yaml \
    --results-root /path/to/results \
    --cache-dir   /path/to/slice_cache \
    --output-dir  /path/to/eval_output \
    --device cuda:0

# Single cell (for SLURM arrays)
slimdiff-metrics fold-eval ... --fold 0 --architecture shared

# Merge partials
slimdiff-metrics fold-eval --aggregate-only --partial-dir /path/to/partials --output-dir /path/to/final
```

### Test Results

All 6 acceptance tests pass (~10 s on CPU):

- AC-1 `test_ac1_load_fold_eval_data` — loader returns expected shapes/dtypes/`n_replicas`
- AC-2 `test_ac2_metric_smoke` — real KID/LPIPS/MMD-MF on 16 × 64² synthetic arrays
- AC-3 `test_ac3_grid_end_to_end` — 2-fold × 2-arch grid writes the 4 artefacts with correct shapes (KID/LPIPS monkey-patched for speed)
- AC-4 `test_ac4_single_cell_partial` — single-cell run writes `fold_metrics_f0_shared.csv`, Wasserstein + counts partials, and `aggregate_partials` can promote them to the grid artefacts
- AC-5 `test_ac5_eval_sample_counts_schema` — JSON schema validates
- AC-6 `test_ac6_cli_registers_fold_eval` — `slimdiff-metrics --help` still lists `fold-eval` and pre-existing subcommands; no ImportError

### Residual notes

- `cli.py` short-circuits `sys.argv[1] == "fold-eval"` to `fold_evaluation.main(sys.argv[2:])` to avoid argparse REMAINDER conflicts with parent-level `--help`. The subparser registration is kept only for the `--help` listing.
- The pipeline is metric-agnostic: each metric has an `enabled` flag in YAML and can be turned off independently per run.
- Per-feature Wasserstein re-uses the features extracted by `MaskMorphologyDistanceComputer.compute()` via the `real_features` / `synth_features` kwargs, avoiding repeated feature extraction.

---

## TASK-05: Post-Hoc Analyses (COMPLETE)

**Status**: Done
**Date**: 2026-04-14
**Addresses**: R1.5 (τ sensitivity of MMD-MF), R1.1 / R1.3 / R2.2 (shared-vs-decoupled cross-fold comparison), plus the LaTeX deliverables for the camera-ready.

### Summary

Added three post-hoc analyses that consume TASK-04's outputs (`fold_metrics.csv`) and the cached replica `.npz` masks under `{results_root}/slimdiff_cr_{arch}_fold_{k}/replicas/`:

1. **τ-sensitivity sweep** — rebinarises the continuous synth masks at each τ ∈ {−0.3, −0.2, −0.1, −0.05, 0.0, 0.05, 0.1, 0.2, 0.3} and recomputes MMD-MF. Real features are extracted once per cell (the τ loop only re-extracts synth features).
2. **Cross-fold statistical comparison** — paired shared-vs-decoupled deltas per fold, Cliff's δ, Cohen's d on paired differences, sign-consistency flag, and Wilcoxon signed-rank for completeness (no significance claims with n=3).
3. **LaTeX tables** — ablation table (shared vs decoupled, bold winners), updated main Table 1 (cross-fold ± on the camera-ready cell + footnote for single-split legacy cells), and per-architecture τ-sensitivity tables.

All three are reachable through one CLI with four modes (`--only {tau_sensitivity, comparison, tables, all}`).

### Files Created

| File | Action |
|------|--------|
| `src/diffusion/scripts/similarity_metrics/posthoc/__init__.py` | Created — package marker |
| `src/diffusion/scripts/similarity_metrics/posthoc/tau_sensitivity.py` | Created — `compute_tau_sensitivity`, `run_tau_sensitivity_grid`, `save_tau_sensitivity_outputs`, `TauSensitivityResult` dataclass |
| `src/diffusion/scripts/similarity_metrics/posthoc/cross_fold_comparison.py` | Created — `cross_fold_comparison` (returns dict report, writes JSON), paired-delta / Cliff's δ / Cohen's d / Wilcoxon / sign-consistency / optional early-stopping block |
| `src/diffusion/scripts/similarity_metrics/posthoc/latex_tables.py` | Created — `generate_ablation_table`, `generate_main_updated_table`, `generate_tau_sensitivity_table`, `generate_all_tables` |
| `src/diffusion/scripts/similarity_metrics/posthoc/cli.py` | Created — `python -m …posthoc.cli` entrypoint with `--only` dispatch |
| `src/diffusion/tests/test_posthoc.py` | Created — 6 acceptance tests (all pass) |

No existing modules were modified.

### Key Design Decisions

1. **τ binarisation convention**: `np.where(continuous > τ, 1.0, -1.0)` produces `{-1, +1}` float32, which `MorphologicalFeatureExtractor._preprocess_mask` correctly re-binarises at `mask > 0`. This makes τ=0.0 semantically equivalent to TASK-04's default (the reference cell).
2. **Real features cached across τ**: `MaskMorphologyDistanceComputer.extract_features(real_masks)` is called once per cell; the τ loop only extracts synth features. Saves ~Nτ × real-extraction cost.
3. **Reproducibility**: `np.random.seed(seed)` is called immediately before each `computer.compute(...)` call so MMD subsampling is deterministic. AC-5 verifies τ=0 matches TASK-04's `fold_metrics.csv` cell within `atol=0.01`.
4. **Statistical hierarchy with n=3**: the report privileges descriptives + paired deltas + Cliff's δ + Cohen's d + sign consistency. Wilcoxon p is included but accompanied by a module-level docstring noting that the minimum two-sided p with n=3 is 0.25 — no significance claims.
5. **Delta sign convention**: `delta = decoupled - shared`, with "lower is better" for all three supported metrics. Positive delta means shared wins.
6. **Table formatting**: KID/LPIPS at 3 decimals, MMD-MF at 2 decimals (matches the paper's existing Table 1). Winners on ablation table bolded via `\mathbf{...}`. Main Table 1 body prints only the `(x₀, L_{1.5})` row with cross-fold ± and a footnote flagging that the remaining 8 cells of the original ablation retain their single-split numbers.
7. **Early-stopping block is optional**: `cross_fold_comparison` accepts CSV (`fold, architecture, epoch`) or JSON (`{"arch": {"0": epoch, …}}`); if absent, the field is simply omitted from the JSON report.

### Outputs

Written under `--output-dir`:

| File | Produced by | Shape / purpose |
|------|-------------|-----------------|
| `tau_sensitivity.csv` | tau_sensitivity | Long form: one row per `(fold, architecture, tau)` — columns `fold, architecture, tau, mmd_mf_mean, mmd_mf_std, n_lesions` |
| `tau_sensitivity_summary.csv` | tau_sensitivity | Aggregated across folds per `(architecture, tau)` — `mmd_mf_mean_across_folds, mmd_mf_std_across_folds, n_lesions_mean` |
| `cross_fold_comparison.json` | comparison | `{metrics: {kid: {…}, lpips: {…}, mmd_mf: {…}}[, early_stopping_epochs: {…}]}` |
| `tables/table_ablation.tex` | tables | Shared vs decoupled ablation, bold winners |
| `tables/table_main_updated.tex` | tables | Updated Table 1 `(x₀, L_{1.5})` row with cross-fold ± + footnote |
| `tables/table_tau_sensitivity_{shared,decoupled}.tex` | tables (if `tau_sensitivity.csv` available) | Per-architecture τ-sweep table |

### CLI

```bash
python -m src.diffusion.scripts.similarity_metrics.posthoc.cli \
    --only all \
    --fold-metrics /path/to/eval_output/fold_metrics.csv \
    --results-root /path/to/results \
    --cache-dir    /path/to/slice_cache \
    --output-dir   /path/to/posthoc_output \
    [--early-stopping-csv /path/to/early_stopping.csv]
```

Modes: `tau_sensitivity` (needs `--results-root` + `--cache-dir`), `comparison` (needs `--fold-metrics`), `tables` (needs `--fold-metrics`, optionally `--tau-csv`), `all` (default; runs the three in sequence and wires τ-sensitivity into the tables automatically).

### Test Results

All 6 acceptance tests pass (~3.4 s on CPU):

- AC-1 `test_ac1_tau_sensitivity_on_dummy_data` — 50 matched synth/real masks, 5-value τ sweep; asserts CSV shapes, `n_lesions > 0` at τ ∈ {−0.1, 0, 0.1}, finite MMD at τ=0, summary CSV correctly aggregated
- AC-2 `test_ac2_cross_fold_comparison_clear_separation` — synthetic 3-fold CSV with decoupled strictly worse on every fold → `cliffs_delta == 1.0`, `all_folds_consistent == True`, `cohens_d > 0`, `delta_mean > 0`
- AC-3 `test_ac3_cross_fold_comparison_mixed_directions` — mixed sign per-fold → `all_folds_consistent == False`, fields still populated and finite
- AC-4 `test_ac4_latex_ablation_table` — `\begin{table}` / `\begin{tabular}` present, `\mathbf` only on the winning (shared) data row, braces matched, `generate_all_tables` writes both ablation + main-updated tables
- AC-5 `test_ac5_tau0_matches_fold_metrics` — `compute_tau_sensitivity([0.0], seed=42)` vs `fold_evaluation.compute_cell_metrics` (KID/LPIPS disabled) → `|Δ| ≤ 0.01`
- AC-6 `test_ac6_only_posthoc_files_are_new` — `git status --porcelain` shows no modifications to pre-existing module files; new files limited to `posthoc/` + `tests/test_posthoc.py` (acknowledged layout deviation)

### Residual notes

- `cross_fold_comparison` asserts complete `(fold, architecture)` pairs in the pivoted frame; a missing cell raises early rather than silently imputing.
- `generate_main_updated_table` emits only the retrained `(x₀, L_{1.5})` row — the remaining 8 cells of the original ablation must be pasted into the final `.tex` manually (flagged by a placeholder row + footnote).
- `run_tau_sensitivity_grid` does NOT parallelise across cells; one (fold, architecture) is processed at a time. Total wall-clock on CPU for 6 cells × 9 τ × 180 000-sample arrays is tracked as a residual risk (feature extraction dominates).

---

## Camera-Ready Runbook (Picasso, end-to-end)

The full 6-cell camera-ready pipeline is run top-to-bottom as follows. Every command is executed from the repo root with conda env `jsddpm` activated.

### 0) Prerequisites on Picasso

| What | Where | How |
|------|-------|-----|
| Repo checkout | `/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/js-ddpm-epilepsy` | `git pull` on the camera-ready branch |
| Conda env `jsddpm` | user's env | `slimdiff-{cache,kfold,train,generate,generate-spec,metrics}` must resolve on PATH |
| Raw patient data | `/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/epilepsy` | NIfTI cohort (Schuch 2023) |
| Slice cache | `${DATASETS_ROOT}/slice_cache` (`slice_cache_config.yaml` at its root) | Built once via `slimdiff-cache build --config src/diffusion/config/cache/epilepsy.yaml`. SLURM jobs also auto-build if missing. |
| 3-fold CSV tree | `${DATASETS_ROOT}/slice_cache/folds/{fold_0,fold_1,fold_2}/{train,val,test}.csv` | Built automatically by the first SLURM job to run (idempotent atomic `folds.tmp/ → folds/` swap), or manually via `slimdiff-kfold --cache-dir ${DATASETS_ROOT}/slice_cache --n-folds 3 --seed 42` |
| Results root | `${RESULTS_ROOT}/icip2026/camera_ready/` (e.g. `/mnt/home/users/tic_163_uma/mpascual/fscratch/results/icip2026/camera_ready`) | `launch_camera_ready.sh` writes `slimdiff_cr_{arch}_fold_{k}/` here |

### 1) Launch the 6-cell training + generation grid (SLURM)

```bash
cd /mnt/home/users/tic_163_uma/mpascual/fscratch/repos/js-ddpm-epilepsy
bash slurm/camera_ready/launch_camera_ready.sh
# Submits 6 independent jobs: {shared,decoupled} × {fold_0,fold_1,fold_2}
# Each --gres=gpu:2 --constraint=dgx --time=3-00:00:00
squeue -u $USER            # monitor
```

Per cell, the job (a) builds the slice cache if missing, (b) builds `${DATASETS_ROOT}/slice_cache/folds/` if missing, (c) trains to `early_stopping.patience=25` (≤1000 epochs), (d) runs `generate_replicas.py` → `slimdiff_cr_{arch}_fold_{k}/replicas/replica_{000..019}.npz` (20 replicas × 9 000 samples each).

### 2) Fold-aware similarity metrics (post-training)

Once all 6 generation jobs have finished:

```bash
slimdiff-metrics fold-eval \
    --config src/diffusion/scripts/similarity_metrics/config/icip2026_camera_ready.yaml \
    --results-root ${RESULTS_ROOT}/icip2026/camera_ready \
    --cache-dir    ${DATASETS_ROOT}/slice_cache \
    --output-dir   ${RESULTS_ROOT}/icip2026/camera_ready/eval_output \
    --device cuda:0
```

This emits `fold_metrics.csv`, `summary_metrics.csv`, `wasserstein_per_feature.csv`, `eval_sample_counts.json`.
For SLURM arrays, add `--fold K --architecture ARCH` per task + `--aggregate-only --partial-dir …` merge job.

### 3) Post-hoc analyses (tau sweep + stats + LaTeX tables)

```bash
python -m src.diffusion.scripts.similarity_metrics.posthoc.cli \
    --only all \
    --fold-metrics ${RESULTS_ROOT}/icip2026/camera_ready/eval_output/fold_metrics.csv \
    --results-root ${RESULTS_ROOT}/icip2026/camera_ready \
    --cache-dir    ${DATASETS_ROOT}/slice_cache \
    --output-dir   ${RESULTS_ROOT}/icip2026/camera_ready/posthoc_output
```

Outputs `tau_sensitivity{,_summary}.csv`, `cross_fold_comparison.json`, and `tables/*.tex` as documented in the TASK-05 section above.

### 4) (Future) Qualitative figures — TASK-06

Pending. Will consume `${RESULTS_ROOT}/icip2026/camera_ready/slimdiff_cr_{arch}_fold_{k}/replicas/replica_000.npz` + the per-cell `fold_metrics.csv` to pick median-LPIPS samples with paired x_T (Design Decision 3 of TASK-03 guarantees this pairing).

### Required Picasso file layout (summary)

```
/mnt/home/users/tic_163_uma/mpascual/fscratch/
├── repos/js-ddpm-epilepsy/                                    # this repo
├── datasets/epilepsy/
│   ├── <raw NIfTI cohort>
│   └── slice_cache/
│       ├── slice_cache_config.yaml
│       ├── train.csv, val.csv, test.csv                       # legacy split (still read by kfold)
│       ├── zbin_priors_brain_roi.npz
│       └── folds/{fold_0,fold_1,fold_2}/{train,val,test}.csv  # built by first SLURM job or slimdiff-kfold
└── results/icip2026/camera_ready/
    ├── slimdiff_cr_{shared,decoupled}_fold_{0,1,2}/
    │   ├── checkpoints/…                                      # training artefacts
    │   └── replicas/replica_{000..019}.npz                    # generation artefacts (TASK-03)
    ├── eval_output/                                           # TASK-04 (step 2)
    │   ├── fold_metrics.csv
    │   ├── summary_metrics.csv
    │   ├── wasserstein_per_feature.csv
    │   └── eval_sample_counts.json
    └── posthoc_output/                                        # TASK-05 (step 3)
        ├── tau_sensitivity.csv
        ├── tau_sensitivity_summary.csv
        ├── cross_fold_comparison.json
        └── tables/
            ├── table_ablation.tex
            ├── table_main_updated.tex
            ├── table_tau_sensitivity_shared.tex
            └── table_tau_sensitivity_decoupled.tex
```

---

## TASK-06: Qualitative Visualization

**Status**: Not started
