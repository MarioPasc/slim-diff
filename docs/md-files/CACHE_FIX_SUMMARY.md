# Cache Build Fix Summary

**Date:** 2026-01-19
**Issue:** Epilepsy dataset cache builder failing with "LoadImaged" transform errors

## Problem

The epilepsy dataset was failing during cache building with errors like:
```
Failed to transform MRIe_103: applying transform <monai.transforms.io.dictionary.LoadImaged object...>
```

This resulted in no slices being cached and the final error:
```
ValueError: No valid slices found in cache
```

## Root Cause

In `src/diffusion/data/caching/builders/epilepsy.py`, the `_discover_all_subjects_for_split()` method was incorrectly passing the `split` parameter ("train", "val", "test") directly to `get_image_path()` and `get_label_path()` functions, which expect **directory names** ("imagesTr", "imagesTs", "labelsTr", "labelsTs") instead.

### Code Before Fix

```python
# Line 127-128 (epilepsy builder)
image_path = get_image_path(dataset_path, subject_id, modality_index, split)
label_path = get_label_path(dataset_path, subject_id, split)
```

This was trying to look for files in directories named "train", "val", "test" which don't exist.

## Solution

Added proper mapping from split names to directory names based on the dataset configuration:

1. **For epilepsy/control datasets with predefined test splits:**
   - `split="train"` or `"val"` → use `"imagesTr"` / `"labelsTr"`
   - `split="test"` with `use_predefined_test=true` → use `"imagesTs"` / `"labelsTs"`

2. **Updated two locations in epilepsy.py:**
   - `_discover_all_subjects_for_split()` method (lines 105-161)
   - `auto_detect_z_range()` method (lines 300-312)

### Code After Fix

```python
# Determine directory names based on split and configuration
use_predefined_test = epilepsy_cfg.get("splits", {}).get("use_predefined_test", False)

if split == "test" and use_predefined_test:
    image_dir = "imagesTs"
    label_dir = "labelsTs"
else:
    image_dir = "imagesTr"
    label_dir = "labelsTr"

# Now use correct directory names
image_path = get_image_path(dataset_path, subject_id, modality_index, image_dir)
label_path = get_label_path(dataset_path, subject_id, label_dir)
```

## Files Modified

- `src/diffusion/data/caching/builders/epilepsy.py`
  - Updated `_discover_all_subjects_for_split()` method (lines 105-161)
  - Updated `auto_detect_z_range()` method (lines 300-312)

## Verification

Both datasets now build caches successfully:

### Epilepsy Dataset
```bash
jsddpm-cache --config src/diffusion/config/cache/epilepsy.yaml
```
- ✅ Training: 3,416 slices
- ✅ Validation: 366 slices
- ✅ Test: 976 slices
- ✅ **Total: 4,758 slices**

### BRATS MEN Dataset
```bash
jsddpm-cache --config src/diffusion/config/cache/brats_men.yaml
```
- ✅ Training: 4,433 slices
- ✅ Validation: 956 slices
- ✅ Test: 950 slices
- ✅ **Total: 6,339 slices**

## Impact

Both datasets now generate slice caches with the same structure:
- `train.csv`, `val.csv`, `test.csv` - metadata CSVs
- `slices/` directory - cached .npz files
- `zbin_priors_brain_roi.npz` - computed z-bin priors
- `cache_stats.yaml` - statistics
- `viz_zbin_priors.png` - visualization

This allows both datasets to be used interchangeably for training the diffusion model and segmentation model.

## Notes

The BRATS MEN builder did not have this issue because it uses a different discovery mechanism (`_discover_all_subjects_from_flat_structure()`) that constructs full paths directly rather than calling the utility functions with split parameters.
