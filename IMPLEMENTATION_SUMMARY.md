# Implementation Summary

This document summarizes the recent improvements to the JS-DDPM codebase, covering both the uncertainty-weighted loss logging fixes and the implementation of sinusoidal position encoding.

---

## Part 1: Uncertainty Loss Logging Fixes

### Problem
The uncertainty-weighted loss logging was inconsistent between what was optimized and what was logged. The model optimized using **clamped** log variance values (clamped to [-5, 5]), but the logging used **unclamped** values.

### Solution

#### 1. Added `get_log_vars_clamped()` Method
**File**: `src/diffusion/losses/uncertainty.py`

- Added method that returns clamped log_vars matching the optimization
- Updated `get_log_vars()` docstring to clarify it returns unclamped values
- Stored `learnable` flag in the class

#### 2. Fixed Logging in Lightning Module
**Files**: `src/diffusion/training/lit_modules.py`

- Replaced re-computed weighted losses with values directly from `loss_details`
- All logged metrics now use clamped values from the forward pass
- Added proper separation between weighted loss metrics and uncertainty-specific metrics
- Metrics logged:
  - `train/loss`, `train/loss_image`, `train/loss_mask`
  - `train/weighted_loss_image`, `train/weighted_loss_mask`
  - `train/log_var_image`, `train/log_var_mask` (only when uncertainty enabled)
  - `train/sigma_image`, `train/sigma_mask` (only when uncertainty enabled)
  - `train/precision_image`, `train/precision_mask` (only when uncertainty enabled)

#### 3. Configuration Validation
**File**: `src/diffusion/config/validation.py`

Created validation module that checks:
- `use_sinusoidal` is properly configured (now accepts true after implementing it)
- `max_z` is positive and warns if set to 128 instead of 127
- CFG null token is in valid range when CFG is enabled
- `initial_log_vars` has exactly 2 values for image and mask

Integration: Validation runs in `JSDDPMLightningModule.__init__()` before building components.

#### 4. Clarified Optimizer Configuration
**File**: `src/diffusion/training/lit_modules.py`

- Enhanced docstring explaining that log_vars are optimized when learnable=True
- Added logging showing parameter counts including uncertainty log_vars

#### 5. Comprehensive Testing
**Files**:
- `src/diffusion/tests/test_uncertainty_loss.py` (7 tests)
- Updated `src/diffusion/tests/test_smoke.py`

Tests verify:
- Loss details match optimization objective
- Log_vars receive gradients when learnable
- Clamped values are used for logging
- Configuration validation works correctly
- Uncertainty parameters actually change during training

### Why Negative Weighted Losses Are Expected

The Kendall uncertainty formulation uses:
```
weighted_loss_i = exp(-log_var_i) * L_i + 0.5 * log_var_i
```

This **can become negative** when:
- Task MSE is very small (high model confidence)
- log_var is negative (learned high precision)

Example: MSE=0.01, log_var=-4.0 → weighted_loss = 0.546 - 2.0 = **-1.454** ✓

This is **expected behavior** that prevents the model from becoming overconfident (driving uncertainties to zero).

---

## Part 2: Sinusoidal Position Encoding Implementation

### Problem
The `conditioning.use_sinusoidal` configuration parameter existed but was:
1. Never actually implemented or wired up
2. Rejected by validation with an error message
3. The `ZPositionEncoder` class was implemented but never used

### Solution

#### 1. Created Custom Embedding Module
**File**: `src/diffusion/model/embeddings/zpos.py`

Implemented `ConditionalEmbeddingWithSinusoidal` class that:
- Handles both pathology class (control/lesion) and z-position
- Decodes composite tokens: `token = z_bin + pathology_class * z_bins`
- Supports sinusoidal encoding for z-position when enabled
- Handles null tokens for classifier-free guidance (CFG)
- Can operate with or without sinusoidal encoding

**Architecture**:
- Pathology embedding: Learned embeddings for control (class=0) and lesion (class=1)
- Z-position encoding:
  - When `use_sinusoidal=True`: Combines learned bin embeddings with sinusoidal position encodings
  - When `use_sinusoidal=False`: Uses standard learned embeddings
- Combines both embeddings via linear layer

#### 2. Wired to Model Factory
**File**: `src/diffusion/model/factory.py`

After building the MONAI `DiffusionModelUNet`:
1. Checks if `conditioning.use_sinusoidal = True`
2. Replaces MONAI's standard `class_embedding` layer with `ConditionalEmbeddingWithSinusoidal`
3. Preserves the embedding dimension from the original model
4. Logs the replacement for transparency

#### 3. Removed Validation Rejection
**File**: `src/diffusion/config/validation.py`

- Removed error that rejected `use_sinusoidal=True`
- Added validation that `max_z` must be positive when using sinusoidal encoding
- Sinusoidal encoding is now fully supported

#### 4. Comprehensive Testing
**File**: `src/diffusion/tests/test_sinusoidal_embedding.py` (5 tests)

Tests verify:
- Custom embedding forward pass works correctly
- Null tokens are handled properly
- Works with and without sinusoidal encoding
- Model builds correctly with `use_sinusoidal=True`
- Model forward pass produces correct output shapes

### How to Use Sinusoidal Encoding

Set in your configuration YAML:
```yaml
conditioning:
  z_bins: 50
  use_sinusoidal: true  # Enable sinusoidal encoding
  max_z: 127  # Maximum z-index (0-indexed: 0-127 for 128 slices)
  cfg:
    enabled: false
    null_token: 100
```

The model will automatically replace the standard class embedding with sinusoidal position encoding that combines:
- Learned pathology class embeddings (control vs. lesion)
- Sinusoidal position encodings for z-position
- Learned bin embeddings

---

## Part 3: Uncertainty Weighting Disabled Mode

### What Loss Is Used When `uncertainty_weighting.enabled = False`?

**File**: `src/diffusion/losses/diffusion_losses.py`

When uncertainty weighting is **disabled**, the model uses `SimpleWeightedLoss` with equal weights [1.0, 1.0]:

```python
if loss_cfg.uncertainty_weighting.enabled:
    self.uncertainty_loss = UncertaintyWeightedLoss(...)
else:
    self.uncertainty_loss = SimpleWeightedLoss(weights=[1.0, 1.0])
```

**Loss Formula**:
```
total_loss = 1.0 * loss_image + 1.0 * loss_mask
           = loss_image + loss_mask
```

This is a simple **unweighted sum** of the per-channel MSE losses.

### Logged Metrics When Uncertainty Disabled

- `train/loss`: Total loss (sum of image and mask losses)
- `train/loss_image`: MSE for image channel
- `train/loss_mask`: MSE for mask channel
- `train/weighted_loss_image`: Same as `loss_image` (weight=1.0)
- `train/weighted_loss_mask`: Same as `loss_mask` (weight=1.0)

**NOT logged** (only available when uncertainty enabled):
- `train/log_var_image`, `train/log_var_mask`
- `train/sigma_image`, `train/sigma_mask`
- `train/precision_image`, `train/precision_mask`

### Testing

**File**: `src/diffusion/tests/test_loss_configurations.py` (4 tests)

Tests verify:
- Correct loss class is used based on configuration
- Logging works correctly in both modes
- Loss computation produces finite values
- Compatible with Lightning module

---

## Test Results

All tests pass successfully:
```
test_smoke.py: 21 tests ✓
test_uncertainty_loss.py: 7 tests ✓
test_sinusoidal_embedding.py: 5 tests ✓
test_loss_configurations.py: 4 tests ✓

Total: 37 tests passed
```

---

## Files Modified/Created

### Modified Files:
1. `src/diffusion/losses/uncertainty.py` - Added `get_log_vars_clamped()` method
2. `src/diffusion/training/lit_modules.py` - Fixed logging, added validation
3. `src/diffusion/config/validation.py` - Created validation module
4. `src/diffusion/model/factory.py` - Wire sinusoidal embedding
5. `src/diffusion/model/embeddings/zpos.py` - Added `ConditionalEmbeddingWithSinusoidal`
6. `src/diffusion/model/embeddings/__init__.py` - Export new classes
7. `src/diffusion/tests/test_smoke.py` - Added test, fixed config

### Created Files:
1. `src/diffusion/tests/test_uncertainty_loss.py` - Uncertainty loss tests
2. `src/diffusion/tests/test_sinusoidal_embedding.py` - Sinusoidal encoding tests
3. `src/diffusion/tests/test_loss_configurations.py` - Loss configuration tests

---

## Configuration Examples

### Standard Configuration (No Uncertainty, No Sinusoidal):
```yaml
conditioning:
  z_bins: 50
  use_sinusoidal: false
  max_z: 127
  cfg:
    enabled: false

loss:
  uncertainty_weighting:
    enabled: false  # Use simple unweighted sum
```

### Uncertainty Weighting Enabled:
```yaml
loss:
  uncertainty_weighting:
    enabled: true
    initial_log_vars: [0.0, 0.0]
    learnable: true  # Log_vars will be optimized
```

### Sinusoidal Encoding Enabled:
```yaml
conditioning:
  z_bins: 50
  use_sinusoidal: true  # Use sinusoidal position encoding
  max_z: 127
```

### Full Featured Configuration:
```yaml
conditioning:
  z_bins: 50
  use_sinusoidal: true
  max_z: 127
  cfg:
    enabled: true
    null_token: 100
    dropout_prob: 0.1

loss:
  uncertainty_weighting:
    enabled: true
    initial_log_vars: [0.0, 0.0]
    learnable: true
```

---

## Key Benefits

1. **Logging Consistency**: All logged metrics now match the actual optimization objective
2. **Configuration Safety**: Invalid configurations fail fast with clear error messages
3. **Verified Learning**: Tests confirm uncertainty parameters are actually being optimized
4. **Better Transparency**: Enhanced documentation and logging
5. **Sinusoidal Encoding**: Fully implemented and tested sinusoidal position encoding
6. **Flexible Loss**: Can easily toggle between weighted and unweighted losses
7. **Comprehensive Testing**: 37 tests ensuring correctness and preventing regressions

---

## Acceptance Criteria Met

✅ **Criterion A**: Logging matches optimization
- `train/loss ≈ train/weighted_loss_image + train/weighted_loss_mask` (within floating-point tolerance)

✅ **Criterion B**: Uncertainty parameters learn
- log_vars change during training when learnable=True (verified by tests)

✅ **Sinusoidal Implementation**: Fully working
- Can enable via config, properly wired, tested end-to-end

✅ **Uncertainty Disabling Works**: Verified
- Uses `SimpleWeightedLoss` with weights=[1.0, 1.0]
- Provides simple unweighted sum of losses
- Properly tested and documented
