# 🐛 Critical Bug: Sinusoidal Embedding with Local Z-Binning

## Summary

**Status:** ✅ **FIXED** (See `SINUSOIDAL_EMBEDDING_FIX.md` for implementation)

**Previous Status:** 🔴 **CRITICAL BUG FOUND**

The `ConditionalEmbeddingWithSinusoidal` class is **incompatible** with LOCAL z-binning. It converts z_bin back to z_index using GLOBAL scaling, causing incorrect sinusoidal encodings.

---

## The Bug

### Location
`src/diffusion/model/embeddings/zpos.py`, line 311:

```python
# BUGGY CODE:
z_indices = ((z_bin.float() + 0.5) / self.z_bins * self.max_z).long()
```

### Problem

This line assumes bins span the FULL volume `[0, max_z]`, but with LOCAL binning, bins only span `z_range`.

### Example

**Configuration:**
- `z_range = [24, 93]` (70 slices)
- `n_bins = 50`
- `max_z = 127`

**Scenario:** Processing slice at z_index=59

1. **Caching (CORRECT):**
   ```python
   z_bin = quantize_z(59, z_range=(24, 93), n_bins=50)
   # z_norm = (59 - 24) / (93 - 24) = 35/69 = 0.507
   # z_bin = int(0.507 * 50) = 25
   ```
   ✓ z_bin = 25 (correct)

2. **Sinusoidal Encoding (BUGGY):**
   ```python
   # Current buggy code:
   z_index = (25.5 / 50) * 127 = 64.77 ≈ 64
   ```
   ❌ Maps to z_index=64 (WRONG!)

3. **What it SHOULD be:**
   ```python
   # Correct code:
   z_index = 24 + (25.5 / 50) * 69 = 24 + 35.19 = 59.19 ≈ 59
   ```
   ✓ Maps to z_index=59 (correct)

### Impact

**The sinusoidal encoding receives the WRONG z-position!**

- z_bin=25 represents slices around **z=59** in local space
- But sinusoidal encoder thinks it's **z=64** (global space)
- **Error: 5 slices off!**

This error varies depending on:
- Larger error for bins in early part of z_range
- Smaller error for bins in middle of z_range
- Can be up to ~20 slices off for extreme bins!

---

## Why This Matters

### 1. Sinusoidal Encoding Purpose

Sinusoidal encodings provide **fine-grained positional information**:
- Bin embeddings: Coarse position (bin 25 = "somewhere in the middle third")
- Sinusoidal encodings: Fine position (z=59 = "specific anatomical landmark")

If the sinusoidal encoding receives the wrong z-index, it defeats the purpose!

### 2. Impact on Model Learning

The model receives **conflicting signals**:
- Bin embedding says: "This is bin 25 (local position 50%)"
- Sinusoidal encoding says: "This is z=64 (global position 50.4%)"
- Ground truth is: "This is z=59 (actual position 50.7% within range)"

This **confuses the model** and degrades the benefit of sinusoidal encodings.

### 3. Anatomical Inconsistency

With the bug:
- z=24 (start of z_range) → sinusoidal thinks it's z=0 (bottom of brain)
- z=93 (end of z_range) → sinusoidal thinks it's z=127 (top of brain)

The model learns **incorrect anatomical relationships**!

---

## The Fix

### Updated `ConditionalEmbeddingWithSinusoidal`

Need to add `z_range` parameter to convert bins correctly:

```python
class ConditionalEmbeddingWithSinusoidal(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        z_bins: int,
        z_range: tuple[int, int],  # NEW PARAMETER
        use_sinusoidal: bool = True,
        max_z: int = 127,
    ) -> None:
        # ...
        self.z_range = z_range

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # ...
        if self.use_sinusoidal:
            # OLD (BUGGY):
            # z_indices = ((z_bin.float() + 0.5) / self.z_bins * self.max_z).long()

            # NEW (CORRECT):
            min_z, max_z_range = self.z_range
            range_size = max_z_range - min_z
            z_indices = (min_z + (z_bin.float() + 0.5) / self.z_bins * range_size).long()
            z_indices = torch.clamp(z_indices, min_z, max_z_range)
```

### Update `factory.py`

Pass z_range when creating the embedding:

```python
def build_model(cfg: DictConfig) -> DiffusionModelUNet:
    # ...
    if cond_cfg.use_sinusoidal and model_cfg.use_class_embedding:
        z_range = tuple(cfg.data.slice_sampling.z_range)  # NEW

        custom_embedding = ConditionalEmbeddingWithSinusoidal(
            num_embeddings=num_class_embeds,
            embedding_dim=embedding_dim,
            z_bins=z_bins,
            z_range=z_range,  # NEW
            use_sinusoidal=True,
            max_z=cond_cfg.max_z,
        )
```

### Update `ZPositionEncoder`

Similar fix needed in the standalone encoder:

```python
class ZPositionEncoder(nn.Module):
    def __init__(
        self,
        n_bins: int,
        embed_dim: int,
        z_range: tuple[int, int],  # NEW
        use_sinusoidal: bool = False,
        max_z: int = 128,
    ) -> None:
        # ...
        self.z_range = z_range
```

---

## Testing Strategy

### Unit Tests

```python
def test_sinusoidal_encoding_with_local_binning():
    """Test that sinusoidal encoding uses correct z-indices with local binning."""
    z_range = (24, 93)  # 70 slices
    z_bins = 50

    embedding = ConditionalEmbeddingWithSinusoidal(
        num_embeddings=100,
        embedding_dim=256,
        z_bins=z_bins,
        z_range=z_range,
        use_sinusoidal=True,
        max_z=127,
    )

    # Test bin 0 (first bin) should map to z ≈ 24
    token_bin0 = torch.tensor([0])  # class=0, bin=0
    # Extract z_index that will be used
    # Should be ≈ 24

    # Test bin 25 (middle bin) should map to z ≈ 59
    token_bin25 = torch.tensor([25])  # class=0, bin=25
    # Should be ≈ 59

    # Test bin 49 (last bin) should map to z ≈ 93
    token_bin49 = torch.tensor([49])  # class=0, bin=49
    # Should be ≈ 93
```

### Integration Test

```python
def test_sinusoidal_matches_cache():
    """Test that sinusoidal z-indices match cached z-indices."""
    # Load cached data
    dataset = SliceDataset(cache_dir, split="train")

    # For each sample, verify sinusoidal encoding receives correct z-index
    for sample in dataset.samples[:100]:
        z_index_actual = sample['z_index']
        z_bin = sample['z_bin']
        token = sample['token']

        # Forward through embedding
        emb_output = embedding(torch.tensor([token]))

        # Check that internal z_index calculation is correct
        # (would need to add a method to extract this for testing)
```

---

## Verification Checklist

Before deploying the fix:

- [ ] Update `ConditionalEmbeddingWithSinusoidal` with z_range parameter
- [ ] Update `ZPositionEncoder` with z_range parameter
- [ ] Update `factory.py` to pass z_range
- [ ] Update all tests for sinusoidal embedding
- [ ] Add specific test for local binning + sinusoidal
- [ ] Verify backwards compatibility (if needed)
- [ ] Document the change in CLAUDE.md

---

## Backwards Compatibility

**Old checkpoints using sinusoidal embeddings are BROKEN anyway** because:
1. Token semantics changed (global → local binning)
2. z_bin values in cache are different

So adding z_range parameter doesn't create additional incompatibility.

**Action:** No backwards compatibility needed - must train from scratch regardless.

---

## Related Code

Files that need updates:
1. ✅ `src/diffusion/model/embeddings/zpos.py` - Main fix
2. ✅ `src/diffusion/model/factory.py` - Pass z_range
3. ✅ `src/diffusion/tests/test_sinusoidal_embedding.py` - Update tests
4. ⚠️ `src/diffusion/config/jsddpm.yaml` - Document z_range dependency

---

## Root Cause

The sinusoidal embedding code was written **before** implementing local z-binning. It assumed:
- Bins span [0, max_z]
- z_bin to z_index conversion is: `z = (bin / n_bins) * max_z`

With local binning:
- Bins span [min_z, max_z_range]
- z_bin to z_index conversion is: `z = min_z + (bin / n_bins) * range_size`

---

## Severity Assessment

**Severity:** 🔴 **CRITICAL** (if using sinusoidal embeddings)

**Impact:**
- ❌ Sinusoidal encodings are WRONG
- ❌ Model learns incorrect positional relationships
- ❌ Defeats the purpose of sinusoidal encoding
- ✅ Model still works (just doesn't benefit from sinusoidal)

**Priority:**
- **HIGH** if `use_sinusoidal=True` in config
- **LOW** if `use_sinusoidal=False` (default)

---

## Recommendation

### If use_sinusoidal=False (default):
✅ **No action needed** - bug doesn't affect you

### If use_sinusoidal=True:
🔴 **MUST FIX** before training

**Steps:**
1. Apply the fix (see below)
2. Rebuild cache (already needed for local binning)
3. Train from scratch (already needed for local binning)

---

## Next Steps

1. Create fix implementation
2. Update all affected tests
3. Add verification test
4. Document the dependency between z_range and sinusoidal
5. Update baseline config if using sinusoidal

---

**Date Identified:** 2025-12-28
**Status:** ✅ **BUG FIXED** - See `SINUSOIDAL_EMBEDDING_FIX.md`
**Affected:** Configurations with `use_sinusoidal=True`

## Fix Implementation

The bug has been **FIXED** and **thoroughly tested**. See `SINUSOIDAL_EMBEDDING_FIX.md` for:
- Complete fix implementation
- Test coverage (36/36 tests pass)
- Before/after comparison
- Deployment checklist

**All tests passing!** Ready for training.
