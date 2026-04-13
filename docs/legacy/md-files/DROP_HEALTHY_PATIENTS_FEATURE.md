# drop_healthy_patients Feature

**Date:** 2026-01-19

## Overview

The `drop_healthy_patients` configuration flag has **dataset-specific behavior** to accommodate different data characteristics:

- **Epilepsy dataset**: Drops entire control/healthy subjects (patients without lesions)
- **BRATS-MEN dataset**: Drops 50% of non-lesion slices per z-bin (balances lesion vs non-lesion slices)

This design reflects the fundamental difference between the datasets:
- Epilepsy has separate control subjects (healthy patients with no lesions)
- BRATS-MEN has no control subjects (all patients have tumors), but individual slices may not contain visible tumor

---

## Configuration

In `src/diffusion/config/cache/brats_men.yaml`:
```yaml
drop_healthy_patients: false  # BraTS-MEN: Drops 50% of non-lesion slices per z-bin to balance data
                              # (Different from epilepsy, which drops control subjects)
```

In `src/diffusion/config/cache/epilepsy.yaml`:
```yaml
drop_healthy_patients: false  # Drop all control/healthy subjects
```

---

## Behavior by Dataset

### Epilepsy Dataset

**Implementation location**: Base class `SliceCacheBuilder.build_cache()` (line 246-250 in `base.py`)

**When `drop_healthy_patients: true`**:
- Filters subjects **before** slice extraction
- Removes all subjects with `source == "control"`
- Affects entire control cohort from Dataset310_MRIcontrol_none

**Example**:
```python
# Before filtering
subjects = [
    SubjectInfo(subject_id="MRIe_001", source="epilepsy", ...),    # Kept
    SubjectInfo(subject_id="MRIe_002", source="epilepsy", ...),    # Kept
    SubjectInfo(subject_id="MRIcontrol_001", source="control", ...),  # Dropped
    SubjectInfo(subject_id="MRIcontrol_002", source="control", ...),  # Dropped
]
```

**Use case**: Training models exclusively on pathology cases without healthy control data.

---

### BRATS-MEN Dataset

**Implementation location**: `BraTSMenSliceCacheBuilder.filter_collected_slices()` (brats_men.py)

**When `drop_healthy_patients: true`**:
- Filters slices **after** extraction, **before** CSV writing
- Groups slices by z-bin (anatomical level)
- For each z-bin:
  - Keeps **all lesion slices** (has_lesion=True)
  - Randomly drops **50% of non-lesion slices** (has_lesion=False)
- Uses fixed random seed (42) for reproducibility

**Algorithm**:
```python
for z_bin in [0, 1, 2, ..., 29]:
    lesion_slices = slices where has_lesion == True
    non_lesion_slices = slices where has_lesion == False

    # Keep all lesion slices
    filtered_slices += lesion_slices

    # Keep 50% of non-lesion slices
    n_to_keep = len(non_lesion_slices) // 2
    kept_slices = random_sample(non_lesion_slices, n=n_to_keep, seed=42)
    filtered_slices += kept_slices
```

**Why per-z-bin filtering?**
- Preserves anatomical distribution across brain levels
- Prevents bias toward slices from certain brain regions
- Maintains balanced representation across superior/inferior brain sections

**Example results** (from actual run):

| Split | Before Filtering | After Filtering | Reduction |
|-------|-----------------|-----------------|-----------|
| Train | 4,434 slices | 2,799 slices | 36.9% |
| Val | 957 slices | 580 slices | 39.4% |
| Test | 951 slices | 613 slices | 35.5% |
| **Total** | **6,342 slices** | **3,992 slices** | **37.0%** |

Breakdown of train split after filtering:
- Lesion slices: 1,179 (kept all)
- Non-lesion slices: 1,620 (kept 50% of ~3,240 original)

**Use case**: Training models with more balanced lesion/non-lesion distribution when the dataset has many more non-lesion slices than lesion slices.

---

## Implementation Details

### Base Class Hook Method

Added to `src/diffusion/data/caching/base.py`:

```python
def filter_collected_slices(
    self,
    slices: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    """Optional hook to filter slices after collection but before CSV writing.

    Default implementation returns slices unchanged.
    Subclasses can override to implement custom filtering logic.

    Args:
        slices: List of slice metadata dictionaries
        split: Split name ("train", "val", or "test")

    Returns:
        Filtered list of slice metadata dictionaries
    """
    return slices
```

This hook is called in `build_cache()` after all slices are collected but before writing to CSV:

```python
# Apply subclass-specific slice filtering
all_metadata[split] = self.filter_collected_slices(all_metadata[split], split)

# Write CSV index for this split
self.write_index_csv(all_metadata[split], split)
```

### BRATS-MEN Override

In `src/diffusion/data/caching/builders/brats_men.py`:

```python
def filter_collected_slices(
    self,
    slices: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    if not self.drop_healthy_patients:
        return slices

    # Group slices by z-bin
    slices_by_zbin = {}
    for slice_meta in slices:
        z_bin = slice_meta["z_bin"]
        if z_bin not in slices_by_zbin:
            slices_by_zbin[z_bin] = {"lesion": [], "non_lesion": []}

        if slice_meta["has_lesion"]:
            slices_by_zbin[z_bin]["lesion"].append(slice_meta)
        else:
            slices_by_zbin[z_bin]["non_lesion"].append(slice_meta)

    # Filter 50% of non-lesion slices per z-bin
    filtered_slices = []
    rng = np.random.RandomState(42)

    for z_bin in sorted(slices_by_zbin.keys()):
        bin_data = slices_by_zbin[z_bin]

        # Keep all lesion slices
        filtered_slices.extend(bin_data["lesion"])

        # Keep 50% of non-lesion slices
        non_lesion_slices = bin_data["non_lesion"]
        n_to_keep = len(non_lesion_slices) // 2

        if len(non_lesion_slices) > 0:
            indices = rng.choice(
                len(non_lesion_slices), size=n_to_keep, replace=False
            )
            kept_slices = [non_lesion_slices[i] for i in indices]
            filtered_slices.extend(kept_slices)

    logger.info(
        f"BraTS-MEN {split}: Dropped non-lesion slices "
        f"({len(slices)} → {len(filtered_slices)}, "
        f"{100 * (1 - len(filtered_slices) / len(slices)):.1f}% reduction)"
    )

    return filtered_slices
```

### Epilepsy Implementation

No override needed - uses base class default behavior. The subject-level filtering happens earlier in the pipeline (in `build_cache()` method).

---

## Backward Compatibility

✅ **Fully backward compatible**:
- Default value: `false` for both datasets
- Epilepsy behavior unchanged (subject-level filtering in base class)
- BRATS-MEN behavior: no-op when flag is `false` (returns slices unchanged)
- Existing caches remain valid

---

## Testing

### Test with BRATS-MEN

```bash
# Edit config
vim src/diffusion/config/cache/brats_men.yaml
# Set: drop_healthy_patients: true

# Build cache
jsddpm-cache --config src/diffusion/config/cache/brats_men.yaml

# Verify reduction
wc -l /path/to/meningioma/slice_cache/*.csv

# Expected: ~37% reduction in total slices
```

### Test with Epilepsy

```bash
# Edit config
vim src/diffusion/config/cache/epilepsy.yaml
# Set: drop_healthy_patients: true

# Build cache
jsddpm-cache --config src/diffusion/config/cache/epilepsy.yaml

# Verify control subjects removed
cat /path/to/epilepsy/slice_cache/train.csv | grep -c "MRIcontrol"
# Expected: 0
```

---

## Design Rationale

### Why Different Behaviors?

**Epilepsy dataset**:
- Has distinct control subjects (Dataset310_MRIcontrol_none)
- Binary classification: epilepsy patients vs healthy controls
- Subject-level filtering is semantically meaningful

**BRATS-MEN dataset**:
- No control subjects (all patients have meningiomas)
- Imbalance is at the **slice level** (many non-tumor slices per patient)
- Slice-level filtering addresses the data imbalance

### Why Per-Z-Bin?

Alternative approaches considered:

1. **Global random sampling** (drop 50% of all non-lesion slices)
   - ❌ Problem: Could remove all non-lesion slices from some z-levels
   - ❌ Problem: Could create anatomical bias (e.g., mostly superior slices)

2. **Per-subject sampling** (drop 50% per patient)
   - ❌ Problem: Doesn't address within-subject imbalance
   - ❌ Problem: Patients with few lesion slices still contribute many non-lesion slices

3. **Per-z-bin sampling** (current approach)
   - ✅ Maintains anatomical distribution
   - ✅ Balances data at each brain level independently
   - ✅ Preserves spatial coverage across the volume

### Why 50%?

- Empirically chosen based on typical lesion/non-lesion ratios in BRATS-MEN
- Can be made configurable in future versions if needed
- Still provides significant representation of non-lesion slices for model learning

---

## Future Extensions

Potential enhancements:

1. **Configurable drop percentage**:
```yaml
drop_healthy_patients: true
drop_percentage: 0.7  # Drop 70% instead of 50%
```

2. **Alternative balancing strategies**:
```yaml
balancing_strategy: "per_subject"  # or "global", "per_zbin"
```

3. **Lesion-aware oversampling**:
```yaml
oversample_lesion_slices: true
oversample_factor: 2.0
```

4. **Dynamic balancing based on dataset statistics**:
```yaml
auto_balance: true  # Automatically determine drop percentage
target_ratio: 0.5   # Target lesion:non-lesion ratio
```

---

## Related Files

- `src/diffusion/data/caching/base.py`: Base class with hook method
- `src/diffusion/data/caching/builders/brats_men.py`: BRATS-MEN override
- `src/diffusion/data/caching/builders/epilepsy.py`: Epilepsy (uses base class)
- `src/diffusion/config/cache/brats_men.yaml`: BRATS-MEN config
- `src/diffusion/config/cache/epilepsy.yaml`: Epilepsy config
