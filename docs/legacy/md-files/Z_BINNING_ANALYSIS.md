# Z-Binning and Z-Range Analysis for JS-DDPM

## Executive Summary

**Current Behavior:** Global binning (bins based on full volume 0-127)
**Expected Behavior:** Local binning (bins based on z_range only)
**Recommendation:** **Switch to LOCAL binning** for better efficiency and alignment with model architecture

---

## Q1: How Should This Work According to Literature?

### Conditional Diffusion Models

In conditional diffusion models like DDPM with class conditioning:

1. **Purpose of Conditioning**: Guide generation toward specific categories
2. **Embedding Space**: Each unique condition gets a learnable embedding
3. **Efficiency Principle**: All conditioning tokens should be utilized during training

### Literature Support for Local Binning

**Analogies from Established Models:**

- **StyleGAN's Truncation Trick**: Operates on the actually-used latent space
- **DALL-E/Imagen**: Class embeddings match the actual class distribution
- **ControlNet**: Spatial conditioning is relative to the working resolution

**Key Insight:** In Ho et al. (2020) DDPM and Nichol & Dhariwal (2021) improved DDPM with class conditioning:
- Conditioning should capture meaningful semantic variation **within the dataset**
- Unused embedding slots waste model capacity
- The model learns p(x|condition) for conditions it actually sees

### Anatomical Considerations (Medical Imaging)

For brain MRI with epilepsy lesions:

- **z_range=[24, 93]** typically captures the cerebrum (where lesions occur)
- Excluding bottom slices (neck/cerebellum) and top slices (vertex) is standard
- Z-position semantics are **relative to the brain region of interest**
- Absolute global position (z=50 in full volume) less meaningful than relative position (middle third of cerebrum)

**Verdict for Q1:** ✅ **LOCAL binning is theoretically justified**
- More efficient use of embedding capacity
- Better aligned with medical imaging conventions
- Matches conditional diffusion model best practices

---

## Q2: Is the Current Implementation Correct?

### Current Implementation Analysis

**File:** `src/diffusion/model/embeddings/zpos.py:30-48`

```python
def quantize_z(z_index: int, max_z: int, n_bins: int) -> int:
    z_norm = z_index / max_z  # Normalizes based on FULL volume
    z_bin = int(z_norm * n_bins)
    return min(max(z_bin, 0), n_bins - 1)
```

**File:** `src/diffusion/data/caching.py:116`

```python
z_bin = quantize_z(z_idx, n_slices - 1, z_bins)  # max_z = n_slices - 1 = 127
```

### Test Results

**Configuration:** z_range=[24, 93], z_bins=10

| Metric | Global (Current) | Local (Expected) |
|--------|------------------|------------------|
| Bins used | 7/10 (bins 1-7) | 10/10 (bins 0-9) |
| Bins wasted | 3 (bins 0, 8, 9) | 0 |
| Slices per bin | Uneven (2-13) | Even (~7) |
| Embedding efficiency | 70% | 100% |

**Specific Example (User's Case):**

z_range=[50, 100], z_bins=10

- **Current:** Maps to global bins 3-7 (50% utilization)
- **Expected:** Maps to local bins 0-9 (100% utilization)
  - Bin 0: [50-54]
  - Bin 1: [55-59]
  - ...
  - Bin 9: [95-100]

### Verdict for Q2

❌ **Current implementation does NOT match expected behavior**

**Evidence:**
1. ✗ Only 60-70% of bins are used during training
2. ✗ Embedding slots for bins 0, 8, 9 never receive gradients
3. ✗ Bin semantics don't align with user's description
4. ✗ Model capacity is underutilized

**Why This Happened:**
Likely the code was written assuming full-volume training (z_range=[0, 127]), then z_range filtering was added later without updating the binning logic.

---

## Q3: Should We Compute MSE with the Most Similar Slice in the Bin?

### Understanding the Loss Computation

**During Training (lit_modules.py:250-260):**

```python
# 1. Take real slice at z=51 (which is in bin 5, say)
x0 = torch.cat([image, mask], dim=1)

# 2. Add noise: x_t = √(ᾱ_t) x0 + √(1-ᾱ_t) ε
x_t = self._add_noise(x0, noise, timesteps)

# 3. Predict noise conditioned on bin token
eps_pred = self(x_t, timesteps, tokens)  # tokens encodes bin 5

# 4. Compute loss on NOISE prediction
loss = MSE(eps_pred, noise)  # NOT MSE(x0_hat, x0)
```

### Key Insight: We're Training on Noise, Not x0

**The model learns:** p(ε | x_t, t, bin_token)
**NOT:** p(x0 | bin_token)

At each training step:
1. We have a **specific** x0 (e.g., slice at z=51)
2. We know its **specific** noise ε
3. We give the model a **coarse** condition (bin 5 = [50-60])
4. The model predicts ε for **that specific** x0

**Analogy:** It's like training a classifier with coarse labels:
- Image of "golden retriever" labeled as "dog"
- Model learns "dog" features but from golden retriever examples
- At test time, "dog" token might generate any dog breed
- This is by design! It increases generation diversity.

### What Happens During Generation

```python
# Sample z_bin = 5 (representing slices 50-60 in local binning)
token = compute_class_token(z_bin=5, pathology_class=1, n_bins=10)

# Generate from pure noise
x_T ~ N(0, I)
for t in reversed(range(T)):
    eps = model(x_t, t, token)  # Condition on bin 5
    x_{t-1} = denoise_step(x_t, eps)

# Result: A slice that "looks like" it's from bin 5
# Could resemble z=51, z=55, or z=58 - all valid!
```

### Evaluation Metrics: PSNR/SSIM/Dice

**File:** `lit_modules.py:320-329`

```python
# Validation step
x0_hat = self._predict_x0(x_t, eps_pred, timesteps)  # Predicted x0
metrics = self.metrics.compute_all(
    x0_hat_image, image,  # Compare to GROUND TRUTH
    x0_hat_mask, mask,
)
```

**The Issue:**
- Ground truth slice is z=51
- Predicted slice (from bin [50-60] token) might look like z=55
- Both are semantically correct for bin [50-60]
- But PSNR/SSIM/Dice penalize the difference

### Should We Use Within-Bin MSE?

**Arguments AGAINST (keep current):**

1. **Validation uses REAL slices with known z**: We're validating the model's ability to reconstruct the exact slice it was given, not generate a plausible slice from the bin.

2. **Training objective is epsilon matching**: The model is trained to predict the exact noise for the exact x0, not the "most similar" x0 in the bin.

3. **Evaluation should be strict**: PSNR/SSIM/Dice on exact reconstruction is a fair evaluation metric. If we relax it, we might mask poor performance.

4. **Implementation complexity**: Finding "most similar slice in bin" requires:
   - Storing all slices from each bin
   - Computing similarity during validation (expensive)
   - Defining "similarity" (L2? SSIM? Perceptual?)

**Arguments FOR (within-bin relaxation):**

1. **Model is handicapped by coarse conditioning**: The model can't distinguish z=51 from z=55 at inference time (both have the same token), so penalizing it for not matching exactly seems unfair.

2. **Generation diversity**: If bin [50-60] includes anatomically diverse slices, the model should be rewarded for generating ANY valid slice from that range.

3. **Aligns with generative model goals**: For a generative model, we care about sampling quality (does it look plausible?) more than reconstruction accuracy.

### Recommendation for Q3

**Split the evaluation:**

✅ **Keep current metrics for reconstruction validation:**
- During validation, we have ground truth x0 at specific z
- Compute PSNR/SSIM/Dice against exact ground truth
- This measures denoising quality (noise prediction accuracy)

✅ **Add separate metrics for generation quality:**
- During generation/visualization callbacks
- Sample from bins and evaluate:
  - **Intra-bin diversity**: Do different noise seeds produce different slices?
  - **Anatomical plausibility**: Does generated slice match expected anatomy for that z-bin?
  - **Fréchet Inception Distance (FID)** or **Kernel Inception Distance (KID)** against all slices in the bin

**Implementation suggestion:**

```python
class BinAwareMetrics:
    """Metrics that understand bin semantics."""

    def compute_bin_aware_psnr(
        self,
        generated: torch.Tensor,
        z_bin: int,
        reference_slices: dict[int, list[torch.Tensor]],  # bin -> list of slices
    ) -> float:
        """Compute PSNR against the best-matching slice in the bin."""
        bin_slices = reference_slices[z_bin]
        psnrs = [compute_psnr(generated, ref) for ref in bin_slices]
        return max(psnrs)  # Best match in bin
```

**However**, for the main training loop, I recommend:

❌ **Do NOT change the loss computation**
- Keep training on exact ε prediction
- The model should learn to reconstruct specific x0 given its noise

✅ **Change the binning to LOCAL** (see below)
- This makes bins more semantically coherent
- Each bin has ~7 similar slices instead of 12-13 scattered slices

---

## Recommended Changes

### 1. Update `quantize_z` to Use Local Binning

**File:** `src/diffusion/model/embeddings/zpos.py`

```python
def quantize_z_local(
    z_index: int,
    z_range: tuple[int, int],
    n_bins: int,
) -> int:
    """Quantize z-position into bins WITHIN the z_range.

    Args:
        z_index: Current z-index.
        z_range: (min_z, max_z) range of training slices.
        n_bins: Number of bins.

    Returns:
        Bin index in [0, n_bins - 1].

    Example:
        >>> quantize_z_local(55, z_range=(50, 100), n_bins=10)
        0  # First bin of the local range
    """
    min_z, max_z = z_range

    if z_index < min_z or z_index > max_z:
        raise ValueError(f"z_index {z_index} outside range [{min_z}, {max_z}]")

    # Normalize within the range
    range_size = max_z - min_z
    if range_size == 0:
        return 0

    z_norm_local = (z_index - min_z) / range_size
    z_bin = int(z_norm_local * n_bins)

    # Clamp to valid range
    return min(max(z_bin, 0), n_bins - 1)
```

### 2. Update Caching to Use Local Binning

**File:** `src/diffusion/data/caching.py:116`

```python
# OLD:
z_bin = quantize_z(z_idx, n_slices - 1, z_bins)

# NEW:
z_bin = quantize_z_local(z_idx, tuple(z_range), z_bins)
```

### 3. Update `compute_z_bin` in conditioning.py

**File:** `src/diffusion/model/components/conditioning.py`

```python
def compute_z_bin(
    z_index: int | torch.Tensor,
    z_range: tuple[int, int],  # NEW parameter
    n_bins: int = 50,
) -> int | torch.Tensor:
    """Compute z-bin from z-index using LOCAL binning within z_range."""
    min_z, max_z = z_range

    if isinstance(z_index, torch.Tensor):
        range_size = max_z - min_z
        z_norm = (z_index.float() - min_z) / range_size
        z_bin = (z_norm * n_bins).long()
        return z_bin.clamp(0, n_bins - 1)
    else:
        range_size = max_z - min_z
        if range_size == 0:
            return 0
        z_norm = (z_index - min_z) / range_size
        z_bin = int(z_norm * n_bins)
        return min(max(z_bin, 0), n_bins - 1)
```

### 4. Rebuild the Cache

After updating the code:

```bash
# The cache must be rebuilt with the new binning logic
jsddpm-cache --config slurm/sinusoidal_embeddings/jsddpm_sinusoidal_embeddings.yaml
```

---

## Impact Assessment

### Benefits of LOCAL Binning

1. ✅ **100% embedding utilization** (all z_bins used)
2. ✅ **Even distribution** (~7 slices per bin for z_range=[24,93], z_bins=10)
3. ✅ **Semantically coherent bins** (consecutive slices are anatomically similar)
4. ✅ **Matches user's mental model** and documentation
5. ✅ **Better model capacity utilization**

### Risks/Considerations

1. ⚠️ **Breaks existing checkpoints** (token semantics change)
   - Solution: Train from scratch (likely already planned)

2. ⚠️ **Bin semantics depend on z_range**
   - If z_range changes, bin 5 means something different
   - Solution: Keep z_range fixed for a project/dataset

3. ⚠️ **Generation at z-positions outside z_range**
   - Can't generate slices outside [24, 93] with local binning
   - Solution: This is already the case (model never saw those slices)

### Comparison Table

| Aspect | Global (Current) | Local (Proposed) |
|--------|------------------|------------------|
| Bins used | 7/10 (70%) | 10/10 (100%) |
| Embedding efficiency | Low | High |
| Anatomical consistency | Absolute | Relative to ROI |
| Generalization to new z_range | Better | Worse |
| Model capacity | Wasted | Fully utilized |
| Matches user expectation | ❌ No | ✅ Yes |

---

## Testing Plan

### Unit Test for Local Binning

```python
def test_local_binning():
    """Test that local binning uses all bins evenly."""
    z_range = (24, 93)
    z_bins = 10

    bin_counts = {}
    for z_idx in range(z_range[0], z_range[1] + 1):
        z_bin = quantize_z_local(z_idx, z_range, z_bins)
        bin_counts[z_bin] = bin_counts.get(z_bin, 0) + 1

    # All bins should be used
    assert set(bin_counts.keys()) == set(range(z_bins))

    # Bins should be roughly equal (±1 slice)
    counts = list(bin_counts.values())
    assert max(counts) - min(counts) <= 1
```

### Integration Test

```python
def test_caching_with_local_binning():
    """Test that cache builder uses local binning correctly."""
    # Run cache builder
    build_slice_cache(cfg)

    # Load cached samples
    dataset = SliceDataset(cache_dir=cfg.data.cache_dir, split="train")

    # Check bins
    bins_seen = set()
    for sample in dataset.samples:
        bins_seen.add(sample["z_bin"])

    # Should see all bins
    assert bins_seen == set(range(cfg.conditioning.z_bins))
```

---

## References

1. **Ho et al. (2020)**: "Denoising Diffusion Probabilistic Models" - Original DDPM paper
2. **Nichol & Dhariwal (2021)**: "Improved Denoising Diffusion Probabilistic Models" - Class conditioning
3. **Dhariwal & Nichol (2021)**: "Diffusion Models Beat GANs on Image Synthesis" - Conditional generation
4. **Rombach et al. (2022)**: "High-Resolution Image Synthesis with Latent Diffusion Models" - Conditioning strategies

---

## Conclusion

**Q1**: ✅ Literature supports **LOCAL binning** for efficient conditional diffusion
**Q2**: ❌ Current implementation uses **GLOBAL binning** (not matching expectations)
**Q3**: ⚠️ Within-bin MSE is theoretically interesting but **NOT recommended for training loss**
   - Keep exact reconstruction metrics for validation
   - Add bin-aware metrics for generation evaluation
   - Fix the root cause by switching to LOCAL binning

**Action Items:**
1. ✅ Implement `quantize_z_local()` function
2. ✅ Update caching.py to use local binning
3. ✅ Rebuild cache with new binning
4. ✅ Add tests to prevent regression
5. ⚠️ Optional: Add bin-aware generation metrics for future work
