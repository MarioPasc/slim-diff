# TASK 04 — Fold-Aware Evaluation Pipeline (KID, LPIPS, MMD-MF)

## Project Context

SLIM-Diff is a shared-bottleneck diffusion model for joint FLAIR MRI + lesion mask synthesis (26.9M params, MONAI `DiffusionModelUNet`). For the ICIP 2026 camera-ready we committed to two new experiments:

1. **Decoupled-bottleneck ablation** — a parameter-matched variant with independent bottleneck paths (built by TASK-01).
2. **3-fold stratified evaluation** — patient-level stratified splits (built by TASK-02).

This produces a **3 folds × 2 architectures = 6 experiment cells**. Each cell generates synthetic data via the generation pipeline (orchestrated by TASK-03). **Your job is to compute KID, LPIPS, and MMD-MF for each of these 6 cells**, aggregate across folds, and output structured results that downstream tasks (TASK-05, TASK-06) consume.

---

## Scope

### Files you OWN (create or modify)

```
src/diffusion/scripts/similarity_metrics/config/icip2026_camera_ready.yaml   # NEW
src/diffusion/scripts/similarity_metrics/fold_evaluation.py                   # NEW — main entry point
src/diffusion/scripts/similarity_metrics/data/fold_loaders.py                 # NEW — fold-aware data loading
```

### Files you READ but do NOT modify

```
src/diffusion/scripts/similarity_metrics/metrics/kid.py          # KID computation
src/diffusion/scripts/similarity_metrics/metrics/lpips.py        # LPIPS computation
src/diffusion/scripts/similarity_metrics/metrics/mask_morphology.py  # MMD-MF computation
src/diffusion/scripts/similarity_metrics/metrics/fid.py          # FID (optional, low priority)
src/diffusion/scripts/similarity_metrics/cli.py                  # Existing CLI structure
src/diffusion/scripts/similarity_metrics/run_icip2026.py         # Existing single-split pipeline
src/diffusion/scripts/similarity_metrics/run_mask_metrics.py     # Existing mask metrics pipeline
```

### Interface contracts (provided by other tasks)

**From TASK-02 (data pipeline):**
- A JSON file at `{results_root}/fold_assignments.json` with structure:
```json
{
  "n_splits": 3,
  "seed": 42,
  "stratify_by": "has_lesion",
  "folds": [
    {"fold": 0, "train": ["sub-001", ...], "test": ["sub-020", ...]},
    {"fold": 1, "train": [...], "test": [...]},
    {"fold": 2, "train": [...], "test": [...]}
  ]
}
```
- Per-fold slice CSVs at `{cache_dir}/fold_{k}/test.csv` with columns: `subject_id, slice_idx, z_bin, lesion_present, image_path, mask_path`.

**From TASK-03 (generation orchestration):**
- Generated replica `.npz` files at:
```
{results_root}/fold_{k}/{architecture}/replicas/replica_{r}.npz
```
  where `architecture ∈ {"shared", "decoupled"}` and each `.npz` contains arrays `images`, `masks`, `zbins`, `domains` following the existing format in `generate_replicas.py`.

---

## Detailed Requirements

### 1. Fold-aware data loading (`fold_loaders.py`)

Create a module that loads real test data and synthetic replicas for a given `(fold, architecture)` pair.

```python
@dataclass
class FoldEvalData:
    """Container for one (fold, architecture) evaluation cell."""
    fold: int
    architecture: str  # "shared" or "decoupled"
    real_images: NDArray[np.float32]    # (N_real, H, W), in [-1, 1]
    real_masks: NDArray[np.float32]     # (N_real, H, W), binary {-1, +1}
    real_zbins: NDArray[np.int64]       # (N_real,)
    synth_images: NDArray[np.float32]   # (N_synth, H, W), in [-1, 1]
    synth_masks: NDArray[np.float32]    # (N_synth, H, W), continuous [-1, 1]
    synth_zbins: NDArray[np.int64]      # (N_synth,)
    n_replicas: int
```

**Critical requirements:**

- Real test images come from the **test** partition of the fold, never the train partition. Use the per-fold `test.csv` to resolve image/mask file paths.
- Synthetic data comes from all replicas merged. Load each `replica_{r}.npz`, concatenate.
- The `.npz` files from `generate_replicas.py` store images in `[-1, 1]` range. Do NOT renormalize.
- Masks in `.npz` are continuous in `[-1, 1]`. For KID/LPIPS, use images only. For MMD-MF, binarise masks at `τ = 0.0` (the midpoint). Store the raw continuous masks as well — TASK-05 needs them for the τ sweep.

Expose a loader function:

```python
def load_fold_eval_data(
    fold: int,
    architecture: str,
    cache_dir: Path,
    results_root: Path,
    max_replicas: int | None = None,
) -> FoldEvalData:
    ...
```

### 2. Per-cell metric computation (`fold_evaluation.py`)

For each `(fold, architecture)` pair, compute:

#### KID (Kernel Inception Distance)

- Use the existing `kid.py` module. It computes KID via polynomial kernel (degree 3) on InceptionV3 features.
- **Reference set**: real test images from the fold.
- **Generated set**: synthetic images from all replicas of that cell.
- The existing `KIDComputer` class expects single-channel grayscale images repeated to 3 channels for InceptionV3. Follow the same convention as `run_icip2026.py`.
- Report: `kid_mean ± kid_std` (from multiple subsets).
- Config: `subset_size=1000`, `num_subsets=100` (same as existing `icip2026.yaml`).

#### LPIPS (Learned Perceptual Image Patch Similarity)

- Use the existing `lpips.py` module with VGG backbone.
- Compute pairwise LPIPS between random synthetic–real pairs.
- **Reference set**: real test images from the fold.
- Report: `lpips_mean ± lpips_std`.
- Config: `n_pairs=1000` (same as existing).

#### MMD-MF (Maximum Mean Discrepancy on Morphological Features)

- Use the existing `mask_morphology.py` module (`MaskMorphologyDistanceComputer`).
- **Binarise** synthetic masks at `τ = 0.0` before computing features.
- **Reference set**: real test masks from the fold (already binary).
- The module extracts 9 morphological features per connected component (area, perimeter, circularity, solidity, extent, eccentricity, major/minor axis, equivalent diameter) and computes polynomial-kernel MMD.
- Report: `mmd_mf_mean ± mmd_mf_std`, plus per-feature Wasserstein distances.

### 3. Aggregation across folds

After computing metrics for all 6 cells, produce:

**Per-cell CSV** (`fold_metrics.csv`):
```
fold,architecture,kid_mean,kid_std,lpips_mean,lpips_std,mmd_mf_mean,mmd_mf_std
0,shared,0.012,0.001,0.305,0.003,1.43,0.22
0,decoupled,...
1,shared,...
...
```

**Cross-fold summary CSV** (`summary_metrics.csv`):
```
architecture,kid_mean,kid_std_across_folds,lpips_mean,lpips_std_across_folds,mmd_mf_mean,mmd_mf_std_across_folds
shared,...
decoupled,...
```

Where `*_mean` is the mean of per-fold means, and `*_std_across_folds` is the std of per-fold means (i.e., variability across data partitions, which is what R1.3 asks for).

**Per-feature Wasserstein CSV** (`wasserstein_per_feature.csv`):
```
fold,architecture,area,perimeter,circularity,solidity,extent,eccentricity,major_axis,minor_axis,eq_diameter
0,shared,...
...
```

### 4. Configuration file (`icip2026_camera_ready.yaml`)

Create a YAML config extending the existing `icip2026.yaml` structure:

```yaml
experiment:
  name: "icip2026_camera_ready"
  type: "fold_evaluation"  # NEW: signals fold-aware mode

folds:
  n_splits: 3
  assignments_file: null  # Override via CLI: --fold-assignments path/to/fold_assignments.json

architectures:
  - shared
  - decoupled

paths:
  results_root: null   # Override via CLI
  cache_dir: null      # Override via CLI
  output_dir: null     # Override via CLI

# Inherit metric configs from the existing pipeline
metrics:
  kid:
    enabled: true
    subset_size: 1000
    num_subsets: 100
    degree: 3
  lpips:
    enabled: true
    net: vgg
    n_pairs: 1000
  mmd_mf:
    enabled: true
    threshold: 0.0     # τ for mask binarisation
    min_lesion_size_px: 5

compute:
  device: "cuda:0"
  batch_size: 32
  seed: 42
```

### 5. CLI entry point

Add a CLI command callable as:

```bash
python -m src.diffusion.scripts.similarity_metrics.fold_evaluation \
    --config src/diffusion/scripts/similarity_metrics/config/icip2026_camera_ready.yaml \
    --results-root /path/to/results \
    --cache-dir /path/to/slice_cache \
    --output-dir /path/to/eval_output \
    --fold-assignments /path/to/fold_assignments.json \
    --device cuda:0
```

Support `--fold` and `--architecture` flags for running a single cell (useful for SLURM array jobs).

---

## Design Principles (Fair but Favorable)

1. **KID is computed against the test fold, not training fold.** If the decoupled model memorises training data, its KID against test will be higher — this is the correct behaviour and the whole point of the shared-bottleneck hypothesis.

2. **Same τ = 0.0 for both architectures.** No per-architecture threshold tuning. This was the threshold reported in the original paper and committed in the rebuttal.

3. **Same InceptionV3 feature extractor, same subset sizes, same number of pairs.** No evaluation-side advantages.

4. **Merge ALL replicas before computing metrics.** This gives maximum statistical power and matches the original paper's protocol (20 replicas merged).

---

## Acceptance Criteria (Testable on 8 GB VRAM)

### AC-1: Unit test — Fold data loading
Create synthetic dummy data matching the expected directory structure:
```
/tmp/test_eval/fold_0/shared/replicas/replica_0.npz
/tmp/test_eval/fold_0/shared/replicas/replica_1.npz
```
Each `.npz` has `images: (50, 160, 160)`, `masks: (50, 160, 160)`, `zbins: (50,)`, `domains: (50,)`.
Create a dummy `test.csv` with 20 entries.
Call `load_fold_eval_data(fold=0, architecture="shared", ...)`.
Assert: `synth_images.shape[0] == 100`, `real_images.shape[0] == 20`, dtypes are `float32`.

### AC-2: Unit test — Metric computation on tiny data
Generate random images `(32, 160, 160)` as both real and synthetic. Run KID, LPIPS, MMD-MF.
Assert: all three return finite floats, KID ≥ 0, LPIPS ∈ [0, 1], MMD-MF ≥ 0.
This must run on CPU (set `--device cpu`) in < 60 seconds.

### AC-3: Unit test — Output CSV format
Run the full pipeline on dummy data (2 folds × 2 architectures, 32 samples each).
Assert: `fold_metrics.csv` has exactly 4 rows (2 folds × 2 architectures).
Assert: `summary_metrics.csv` has exactly 2 rows (shared, decoupled).
Assert: all numeric columns are present and contain finite values.

### AC-4: Integration test — Single-cell execution
Run `fold_evaluation.py --fold 0 --architecture shared` on dummy data.
Assert: produces one row in `fold_metrics.csv` with `fold=0, architecture=shared`.

### AC-5: Parameter count logging
Log and save the number of real test samples and synthetic samples per cell to a JSON sidecar file `eval_sample_counts.json`. This is needed for the paper.

### AC-6: Backward compatibility
The existing `slimdiff-metrics` CLI must remain functional. Do NOT modify `cli.py` beyond adding an optional import. All new code goes in new files.

---

## Anti-Patterns

- **Do NOT re-implement KID, LPIPS, or MMD-MF.** Reuse the existing modules in `metrics/`. They are tested and match the paper's reported values.
- **Do NOT compute FID unless explicitly requested.** KID is the primary metric; FID requires large sample counts and is less reliable at small N.
- **Do NOT binarise masks at any threshold other than τ = 0.0** for the primary results. The τ sweep is TASK-05's responsibility using the raw continuous masks you expose.
- **Do NOT normalise or rescale images before KID/LPIPS.** The existing modules handle the grayscale-to-RGB conversion internally.
- **Do NOT use training data as the reference set.** Always use the test partition of the current fold.
