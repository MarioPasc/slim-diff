# JS-DDPM Training Failure Investigation

**Date**: 2026-01-02
**Status**: In Progress
**Issue**: Model produces noisy outputs despite loss decreasing

---

## Problem Summary

After code changes to add anatomical prior conditioning, both baseline (no features) and full-featured experiments produce garbage outputs (pure noise shaped like brain regions). The working snapshot is at commit `73d6968a4caba1c84eccd04150273af33ccaef2f`.

### Key Symptoms
1. **Loss decreases normally**: train/loss goes from ~2.0 to ~0.22 over 500 epochs
2. **PSNR is ALWAYS negative**: ranges from -3.07 (epoch 0) to -0.20 (epoch 449)
3. **SSIM is very low**: ~0.05-0.08, often NaN
4. **Visualizations show noise**: Brain-shaped regions filled with pure noise (post-processing masks the background)
5. **Both baseline and full experiments fail identically**

---

## Files Compared (Working vs Current)

### 1. `src/diffusion/training/lit_modules.py`
- **Status**: Minor changes, mostly additions for anatomical conditioning
- **Key changes**:
  - Added z-bin priors loading for post-processing
  - Added `_use_anatomical_conditioning` flag
  - Added anatomical prior concatenation in training/validation steps
  - Changed logging to conditional: `if "loss_image" in loss_details:`
- **Conclusion**: Changes are conditional and disabled in baseline; NOT the cause

### 2. `src/diffusion/losses/diffusion_losses.py`
- **Status**: Refactored but functionally equivalent
- **Key changes**:
  - Renamed `self.uncertainty_loss` to `self.loss`
  - Added `clamp_range` parameter
  - Interface unchanged: `forward(eps_pred, eps_target, x0_mask)`
- **Tests**: All 4 loss regression tests PASS
- **Conclusion**: Loss computation is correct; NOT the cause

### 3. `src/diffusion/model/factory.py`
- **Status**: Added anatomical conditioning support
- **Key changes**:
  - `in_channels += 1` when `anatomical_conditioning=True`
  - DiffusionSampler now accepts `anatomical_mask` argument
- **Conclusion**: Changes are conditional; NOT the cause for baseline

### 4. YAML Configs
- **Old working baseline**: No EMA, no postprocessing section, max_epochs=200
- **New failing baseline**: EMA enabled, postprocessing.zbin_priors.enabled=true, max_epochs=500

---

## Critical Commits Since Working Snapshot

```
9f70e60 Fix callback bad input when anatomical prior was being used
b3a6550 Solve jsddpm baseline config file
2193db3 Found critical issue regarding MSE loss computation, solved  <-- KEY COMMIT
6fecfa1 Reconfigure experiments
0a97bbb Refactorization of anatomical priors as channel concatenation in model
871963f Solve gradient explosion by normalizing MSE with pixels  <-- KEY COMMIT
```

### Commit `871963f` Analysis
- Changed MSE normalization from `weight_sum` to `total_pixels`
- Aimed to prevent gradient explosion with sparse weights
- This was in INTERMEDIATE code, may have been buggy

### Commit `2193db3` Analysis
- "Found critical issue regarding MSE loss computation, solved"
- Completely refactored loss interface
- Added regression tests
- Test comments describe the bug: "Computing single MSE on full (B, 2, H, W) tensor instead of separate losses"

---

## Verified Working Correctly

1. **Scheduler Consistency**
   - Training and inference schedulers have IDENTICAL `alphas_cumprod`
   - Cosine schedule values verified at t=0,100,250,500,750,999
   - Max difference: 0.0

2. **Loss Function Tests**
   - `test_loss_computes_separate_channel_losses`: PASS
   - `test_loss_image_and_mask_are_independent`: PASS
   - `test_loss_with_lesion_weighting`: PASS
   - `test_loss_forward_signature`: PASS

3. **Smoke Tests**
   - All 24 tests PASS including training step tests

4. **Training Step Logic**
   - Identical between working snapshot and current code (except anatomical conditioning which is disabled)

---

## Performance Data Analysis

### From `/media/mpascual/Sandisk2TB/research/epilepsy/results/jsddpm_baseline/performance.csv`

**Loss Values (decreasing = good)**:
- Epoch 0: val/loss = 1.99, train/loss_image = 0.997, train/loss_mask = 1.001
- Epoch 449: val/loss = 0.22, train/loss_image = 0.133, train/loss_mask = 0.096

**PSNR Values (should be positive 20-30dB for good images)**:
- Epoch 0: -3.07 dB
- Epoch 99: -1.57 dB
- Epoch 199: -0.75 dB
- Epoch 399: -0.10 dB
- Epoch 449: -0.20 dB (NEVER BECOMES POSITIVE)

**Prediction MSE at Different Timesteps** (diagnostics/pred_mse_image_t*):
- Epoch 0: ~1.0 at all timesteps (random)
- Epoch 399: t50=0.295, t250=0.165, t500=0.139, t950=0.125
- Pattern is correct (easier to predict noise at high t)

---

## PSNR Calculation Verification

```python
# PSNR = 10 * log10(max_val^2 / MSE)
# For max_val=1.0:
MSE = 0.13  → PSNR = 8.88 dB   # Should see this if x0_hat is close to x0
MSE = 1.58  → PSNR = -1.97 dB  # What we're seeing (x0_hat is garbage)
MSE = 0.67  → PSNR = 1.77 dB   # Random predictions
```

**Key Insight**: Noise prediction MSE is ~0.13 (good), but x0_hat vs x0 MSE is ~1.58 (terrible). This means `_predict_x0()` is producing garbage despite good noise predictions.

---

## Key Differences in Config

| Parameter | Working (73d6968) | Current Baseline |
|-----------|-------------------|------------------|
| EMA | Not present | enabled=true, decay=0.9999 |
| postprocessing.zbin_priors | Not present | enabled=true |
| scheduler.schedule | "linear_beta" (in base config) | "cosine" |
| max_epochs | 200 | 500 |
| clamp_range | Not present | [-5.0, 5.0] |

---

## Hypotheses to Test

### 1. EMA Callback Issue (HIGH PRIORITY)
- EMA is NEW in current baseline, not in working baseline
- With decay=0.9999, EMA updates very slowly
- Early epochs would use near-initial weights for validation
- **BUT**: Problem persists at epoch 449, so EMA has had time to converge
- **Test**: Disable EMA and re-run

### 2. Cosine vs Linear Beta Schedule
- Working config used "linear_beta"
- Current uses "cosine"
- Both are valid schedules, but different behavior
- **Test**: Change to linear_beta schedule

### 3. Z-bin Priors Post-processing Corruption
- Enabled for validation in current config
- Applied AFTER x0_hat prediction, BEFORE metrics
- Could corrupt predictions before PSNR computation
- **BUT**: Should only affect regions outside brain ROI
- **Test**: Disable postprocessing.zbin_priors

### 4. _predict_x0 Bug
- The function looks correct mathematically
- Uses registered alphas_cumprod buffer
- **Verified**: Same code in working and current versions
- **Unlikely** to be the cause

### 5. Cache Data Corruption
- Cache built at different time than training
- z_bins verified: 0-29 (correct for z_bins=30)
- tokens verified: 0-59 (correct for 2*30=60 classes)
- **Unlikely** to be the cause

---

## Files to Check Next

1. **EMACallback full implementation**: `/home/mpascual/research/code/js-ddpm-epilepsy/src/diffusion/training/callbacks/epoch_callbacks.py` lines 493-630
2. **Visualization callback sampling**: Same file, `VisualizationCallback` class
3. **DiagnosticsCallback**: Check if any callback modifies model state incorrectly

---

## Recommended Next Steps

1. **Quick Test**: Run baseline with `ema.enabled: false` to rule out EMA
2. **Quick Test**: Run baseline with `scheduler.schedule: "linear_beta"`
3. **Quick Test**: Run baseline with `postprocessing.zbin_priors.enabled: false`
4. **Deep Dive**: Add debug logging to `_predict_x0()` to inspect actual values
5. **Compare**: Run the EXACT working snapshot config on current code

---

## Working Snapshot Location

```
/home/mpascual/Downloads/js-ddpm-epilepsy-73d6968a4caba1c84eccd04150273af33ccaef2f/
```

**Good output example**: User-provided image shows proper brain MRI with lesion overlays at various z-positions.

---

## Cache Location

```
/media/mpascual/Sandisk2TB/research/epilepsy/data/slice_cache/
```
- train.csv, val.csv, test.csv
- z_bins: 0-29, tokens: 0-59

---

## Result Locations

- Baseline (failing): `/media/mpascual/Sandisk2TB/research/epilepsy/results/jsddpm_baseline/`
- Full experiment (failing): `/media/mpascual/Sandisk2TB/research/epilepsy/results/jsddpm_sinus_kendall_weighted_anatomicalprior/`
