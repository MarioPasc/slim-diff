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

## TASK-03: Training Orchestration

**Status**: Not started  
**Blocked by**: ~~TASK-01~~ (done), ~~TASK-02~~ (done) — **UNBLOCKED**

---

## TASK-04: Fold-Aware Evaluation Pipeline

**Status**: Not started

---

## TASK-05: Post-Hoc Analyses

**Status**: Not started

---

## TASK-06: Qualitative Visualization

**Status**: Not started
