# Replica Quality Check Script Usage

## Overview

The `check_replica_quality.py` script performs comprehensive validation of generated synthetic replicas against the test set distribution.

**Location**: `src/diffusion/scripts/check_replica_quality.py`

## What It Does

### 1. Statistical Validation
- **Sample count distribution**: Verifies each condition has correct number of samples
- **Brain fraction**: Compares brain tissue fraction per z-bin
- **Intensity statistics**: Validates mean and std intensity distributions
- **Lesion area**: Checks lesion size distributions

### 2. Outlier Detection
- Identifies samples > 2 standard deviations from expected test values
- Reports outliers per metric and condition
- Highlights problematic replicas

### 3. Visual Inspection
- **6 comprehensive plots**:
  1. Sample count comparison (2x2 panel)
  2. Brain fraction per z-bin (line plot with outliers)
  3. Intensity statistics (2x2 panel: mean + std)
  4. Lesion area distribution (2x1 panel)
  5. Detailed image grid (all z-bins)
  6. Representative image grid (5 evenly-spaced z-bins)

### 4. Reports
- **summary.txt**: Human-readable summary
- **summary.json**: Structured data for programmatic analysis
- **comparison.csv**: Per-condition statistics
- **outliers.csv**: Detailed outlier list

---

## Quick Start

### Basic Usage

```bash
python src/diffusion/scripts/check_replica_quality.py \
    --replicas-dir /media/mpascual/Sandisk2TB/research/epilepsy/results/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/replicas \
    --test-dist-csv docs/test_analysis/test_zbin_distribution.csv \
    --output-dir outputs/quality_check_full
```

### Fast Testing Mode (2 replicas, no images)

```bash
python src/diffusion/scripts/check_replica_quality.py \
    --replicas-dir /media/mpascual/Sandisk2TB/research/epilepsy/results/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/replicas \
    --test-dist-csv docs/test_analysis/test_zbin_distribution.csv \
    --output-dir outputs/quality_check_test \
    --max-replicas 2 \
    --skip-images \
    --verbose
```

### With Custom Thresholds

```bash
python src/diffusion/scripts/check_replica_quality.py \
    --replicas-dir /media/mpascual/Sandisk2TB/research/epilepsy/results/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/replicas \
    --test-dist-csv docs/test_analysis/test_zbin_distribution.csv \
    --output-dir outputs/quality_check_custom \
    --outlier-threshold 2.5 \
    --n-images-per-condition 5 \
    --n-representative-zbins 7
```

### With Config File (for brain mask parameters)

```bash
python src/diffusion/scripts/check_replica_quality.py \
    --replicas-dir /media/mpascual/Sandisk2TB/research/epilepsy/results/replicas_jsddpm_sinus_kendall_weighted_anatomicalprior/replicas \
    --test-dist-csv docs/test_analysis/test_zbin_distribution.csv \
    --output-dir outputs/quality_check_full \
    --config slurm/jsddpm_sinus_kendall_weighted_anatomicalprior/jsddpm_sinus_kendall_weighted_anatomicalprior.yaml
```

---

## Command-Line Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--replicas-dir` | Directory containing `replica_*.npz` files |
| `--test-dist-csv` | Path to `test_zbin_distribution.csv` |
| `--output-dir` | Output directory for reports and plots |

### Optional Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | None | Path to YAML config (for brain mask params) |
| `--outlier-threshold` | 2.0 | Outlier detection threshold (standard deviations) |
| `--max-replicas` | None | Limit replicas to load (for testing) |

### Image Visualization

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-images-per-condition` | 3 | Example images per condition |
| `--n-representative-zbins` | 5 | Z-bins for representative grid |
| `--skip-images` | False | Skip image grids (faster) |

### Brain Mask Computation

| Argument | Default | Description |
|----------|---------|-------------|
| `--gaussian-sigma-px` | 1.0 | Gaussian sigma for brain mask |
| `--min-component-px` | 100 | Min component size for brain mask |

### Output Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--dpi` | 300 | DPI for saved plots |
| `--verbose` | False | Enable verbose logging |

---

## Output Structure

```
output_dir/
├── plots/
│   ├── sample_count_comparison.png       # Sample counts: test vs replicas
│   ├── brain_fraction_comparison.png     # Brain fraction per z-bin
│   ├── intensity_comparison.png          # Intensity statistics
│   ├── lesion_area_comparison.png        # Lesion area distribution
│   ├── image_grid_detailed.png           # All z-bins visualization
│   └── image_grid_representative.png     # 5 z-bins visualization
├── reports/
│   ├── summary.txt                       # Human-readable summary
│   ├── summary.json                      # Structured JSON data
│   ├── comparison.csv                    # Per-condition comparison
│   └── outliers.csv                      # Outlier details (sorted by |z-score|)
└── check_replica_quality.log             # Execution log
```

---

## Understanding the Plots

### 1. Sample Count Comparison
- **Panel 1**: Stacked bars showing n_slices per z-bin (test vs replica)
- **Panel 2**: Total counts by domain (control vs epilepsy)
- **Panel 3**: Deviation from test per condition
- **Panel 4**: Summary statistics

### 2. Brain Fraction Comparison
- **Test data**: Thick colored line (alpha=1.0)
- **Individual replicas**: Thin grey lines (alpha=0.3)
- **Replica mean**: Thick colored line with markers (alpha=1.0)
- **Std bands**: Shaded regions (alpha=0.2)
- **Outliers**: Red X markers (alpha=1.0)

### 3. Intensity Comparison (2x2)
- **Top row**: Mean intensity (control, epilepsy)
- **Bottom row**: Std intensity (control, epilepsy)
- Same color scheme as brain fraction plot

### 4. Lesion Area Comparison
- **Panel 1**: Mean lesion area per z-bin (test vs replica)
- **Panel 2**: Histogram of all lesion areas

### 5. Image Grids
- **Detailed**: Shows all z-bins with test data
- **Representative**: Shows 5 evenly-spaced z-bins
- Each condition shows 3 example images
- Lesion masks overlaid in red (alpha=0.5)

---

## Interpreting Results

### Good Replica Quality Indicators
- ✅ Sample counts match test distribution (< 5% deviation)
- ✅ Brain fraction within ±0.05 of test mean
- ✅ Intensity distributions overlap well
- ✅ Lesion area distributions similar
- ✅ Low outlier rate (< 2%)
- ✅ Images visually similar to test set

### Warning Signs
- ⚠️ Large sample count deviations (> 10%)
- ⚠️ Brain fraction systematically higher/lower
- ⚠️ Intensity distributions shifted
- ⚠️ High outlier rate (> 5%)
- ⚠️ One replica consistently flagged as outlier (may indicate bad checkpoint)

### Critical Issues
- ❌ Sample count completely wrong (missing conditions)
- ❌ Brain fraction way off (> 0.1 difference)
- ❌ Intensity out of expected range
- ❌ Many outliers in specific conditions
- ❌ Generated images don't resemble MRI data

---

## Performance Notes

### Memory Usage
- **15 replicas × 4500 samples × 128×128 × 2 channels × 2 bytes (float16)**: ~6 GB
- All replicas loaded simultaneously (faster processing)
- May need 8-16 GB RAM total

### Computation Time
- **Statistics computation**: ~1-2 hours (depends on CPU)
  - Brain mask computation: ~0.1s per slice
  - 15 replicas × 4500 slices = 67,500 slices
- **Visualization**: ~5-10 minutes
- **Total**: ~1.5-2.5 hours for full analysis

### Tips for Faster Testing
1. Use `--max-replicas 2` to test with 2 replicas only
2. Use `--skip-images` to skip image grid generation
3. Use `--verbose` to monitor progress

---

## Validation Checklist

After running the script, check:

### 1. Console Output
- [ ] No errors or warnings
- [ ] All replicas loaded successfully
- [ ] Statistics computed for all samples
- [ ] Outlier detection completed

### 2. Summary Report (`reports/summary.txt`)
- [ ] Total sample counts match
- [ ] MAE per metric is reasonable
- [ ] Outlier rate is acceptable

### 3. Plots (`plots/*.png`)
- [ ] All 6 plots generated
- [ ] Test and replica lines visible
- [ ] Outliers clearly marked
- [ ] Images look like MRI data

### 4. CSV Files (`reports/*.csv`)
- [ ] comparison.csv has entries for all conditions
- [ ] outliers.csv lists detected outliers
- [ ] No unexpected NaN values

---

## Troubleshooting

### Error: "No replica_*.npz files found"
- Check `--replicas-dir` path is correct
- Ensure replica files follow naming pattern: `replica_000.npz`, `replica_001.npz`, etc.

### Error: "Failed to load replica"
- Replica file may be corrupted
- Check file size (should be ~100 MB per replica)
- Script will skip corrupted files and continue

### Error: "Missing required columns"
- Ensure `test_zbin_distribution.csv` has correct format
- Should have columns: split, zbin, lesion_present, domain, n_slices, mean_brain_frac, etc.

### MemoryError
- Reduce `--max-replicas` (e.g., process 5 at a time)
- Close other applications
- Use machine with more RAM

### Slow computation
- Normal for large replica sets
- Use `--verbose` to monitor progress
- Statistics computation is the slowest part (expected)

---

## Example Workflow

```bash
# 1. Test with 2 replicas first (fast)
python src/diffusion/scripts/check_replica_quality.py \
    --replicas-dir /path/to/replicas \
    --test-dist-csv docs/test_analysis/test_zbin_distribution.csv \
    --output-dir outputs/quality_check_test \
    --max-replicas 2 \
    --skip-images

# 2. Check output
ls -lh outputs/quality_check_test/reports/
cat outputs/quality_check_test/reports/summary.txt

# 3. If looks good, run full analysis
python src/diffusion/scripts/check_replica_quality.py \
    --replicas-dir /path/to/replicas \
    --test-dist-csv docs/test_analysis/test_zbin_distribution.csv \
    --output-dir outputs/quality_check_full \
    --verbose

# 4. Review all outputs
xdg-open outputs/quality_check_full/plots/brain_fraction_comparison.png
xdg-open outputs/quality_check_full/plots/image_grid_representative.png
```

---

## Next Steps

After validating replica quality:

1. **If quality is good**:
   - Use replicas for downstream experiments
   - Document results in paper/presentation

2. **If quality issues found**:
   - Check training convergence
   - Review EMA weights usage
   - Inspect checkpoint quality
   - Consider retraining with different hyperparameters

3. **For further analysis**:
   - Load `comparison.csv` and `outliers.csv` in pandas
   - Create custom visualizations
   - Compute additional metrics

---

## Citation

If you use this quality check tool in your research, please cite:

```
[Your paper citation here]
```
