# Comprehensive Diagnostic Analysis Report: x0_lp_1.5

**Experiment**: x0_lp_1.5 (sample prediction + Lp norm p=1.5)
**Analysis Date**: 2026-01-25
**Overall Severity Score**: 0.185 (Grade C)
**Classification AUC**: 0.996 (pooled: 0.973, 95% CI: 0.965-0.979)

---

## Executive Summary

The x0_lp_1.5 model represents the best configuration from the ICIP 2026 ablation study, achieving the lowest artifact severity (0.025 in normalized cross-experiment comparison). However, a binary classifier trained to distinguish real from synthetic images still achieves near-perfect AUC (0.996), indicating persistent discriminable artifacts.

**Primary Finding**: The dominant artifact is a **21% high-frequency energy deficit** in the finest wavelet scale (HH L1 ratio = 0.787, target > 0.85). This manifests as subtle over-smoothing detectable through:
- GLCM texture features (dissimilarity d=0.47, homogeneity d=-0.30)
- Gradient magnitude distributions (d=0.45)
- Frequency band power ratios (band 4 ratio = 0.86)

**XAI Attribution**: The classifier primarily exploits **image channel** artifacts (56% gradient attribution, 10x ablation impact vs. mask). Spectral attribution reveals **negative concordance** (-0.70), meaning the classifier learns low-mid frequency texture features that correlate with the high-frequency deficit.

---

## 1. Spectral Analysis

### 1.1 Power Spectral Density (PSD)

| Metric | Real | Synthetic | Difference |
|--------|------|-----------|------------|
| Spectral slope | 3.25 | 3.33 | +0.081 (steeper) |
| JS divergence | - | - | 2.4e-05 (excellent) |

**Interpretation**: The synthetic images have a slightly steeper spectral slope, indicating faster power falloff at high frequencies (over-smoothing). The extremely low JS divergence (0.00002) shows overall spectral shape is well-preserved.

### 1.2 Per-Band Power Ratios (Image Channel)

| Band | Frequency Range | Power Ratio | Significance |
|------|-----------------|-------------|--------------|
| 0 | 0.006-0.015 | 1.07 | *** (excess) |
| 1 | 0.015-0.036 | 1.07 | *** (excess) |
| 2 | 0.036-0.087 | 1.00 | ns |
| 3 | 0.087-0.208 | 0.92 | *** (deficit) |
| 4 | 0.208-0.500 | **0.86** | *** (deficit) |

**Key Finding**: Low frequencies show 7% excess power while high frequencies show 14% deficit. This cross-over pattern is characteristic of over-smoothing: the model slightly exaggerates low-frequency structure while failing to reproduce fine-scale texture.

**Evidence**: `spectral/spectral_results.json`, `spectral/psd_overlay_image.png`

---

## 2. Wavelet Analysis (Critical Finding)

### 2.1 Multi-Scale Energy Distribution

The 4-level db4 wavelet decomposition reveals the HF deficit is **concentrated at Level 1** (finest scale):

| Level | Subband | Energy Ratio | KS Statistic |
|-------|---------|--------------|--------------|
| L1 (finest) | LH | 0.863 | 0.228 *** |
| L1 | HL | 0.889 | 0.222 *** |
| L1 | **HH** | **0.787** | 0.227 *** |
| L2 | LH | 0.962 | 0.232 *** |
| L2 | HL | 0.908 | 0.229 *** |
| L2 | HH | 0.902 | 0.228 *** |
| L3 | LH | 1.005 | 0.201 *** |
| L3 | HL | 0.993 | 0.209 *** |
| L3 | HH | 0.945 | 0.190 *** |
| L4 (coarsest) | LH | 1.038 | 0.156 *** |
| L4 | HL | 1.058 | 0.156 *** |
| L4 | HH | 1.037 | 0.105 *** |

**Critical Metric**: `wavelet_L1_HH_ratio = 0.787` (target > 0.85)

**Interpretation**: The 21% deficit in the finest diagonal subband (HH L1) represents missing high-frequency diagonal texture details. This is the fingerprint the classifier exploits. Coarser levels (L3-L4) show near-unity or excess energy, consistent with the spectral analysis showing LF excess.

**Evidence**: `wavelet_analysis/wavelet_analysis_results.json`, `wavelet_analysis/wavelet_energy_per_level.png`

---

## 3. Texture Analysis

### 3.1 GLCM Features

| Property | Real Mean | Synth Mean | Cohen's d | Significance |
|----------|-----------|------------|-----------|--------------|
| Contrast | 69.0 | 67.0 | 0.15 | *** |
| **Dissimilarity** | 3.30 | 3.05 | **0.47** | *** |
| **Homogeneity** | 0.649 | 0.667 | **-0.30** | *** |
| Energy | 0.542 | 0.539 | 0.03 | ** |
| Correlation | 0.944 | 0.948 | -0.39 | *** |
| Entropy | 8.91 | 8.75 | 0.18 | *** |

**Key Finding**: Synthetic images are more homogeneous (d=-0.30) with lower dissimilarity (d=0.47), confirming the over-smoothing hypothesis. The medium effect sizes indicate these differences are perceptually subtle but statistically robust.

### 3.2 Gradient Magnitude

| Metric | Real | Synthetic | Cohen's d |
|--------|------|-----------|-----------|
| Mean | 0.576 | 0.534 | **0.45** |
| Std | 1.113 | 1.108 | - |
| P95 | 3.13 | 3.05 | - |

**Interpretation**: Lower mean gradient magnitude in synthetic images confirms reduced edge sharpness and texture contrast.

**Evidence**: `texture/texture_results.json`, `texture/glcm_violins_image.png`

---

## 4. Distribution Analysis

### 4.1 Overall Intensity Distribution

| Metric | Image | Mask |
|--------|-------|------|
| KS statistic | 0.342 | 0.241 |
| Wasserstein | 0.022 | 0.001 |
| Mean shift | +0.020 | +0.001 |

### 4.2 Per-Tissue Wasserstein Distance

| Region | Wasserstein | Significance |
|--------|-------------|--------------|
| All | 0.022 | *** |
| **Lesion** | **0.062** | *** |
| Background | 0.021 | *** |

**Key Finding**: Lesion regions show 3x higher Wasserstein distance (0.062 vs 0.022) than overall, indicating the model has more difficulty reproducing lesion-specific intensity distributions. This justifies increased lesion weighting in training.

**Evidence**: `distribution_tests/distribution_tests_results.json`

---

## 5. Boundary Analysis

| Metric | Real | Synthetic | Difference |
|--------|------|-----------|------------|
| Sharpness (gradient peak) | 0.155 | 0.154 | -0.008 (ns) |
| Transition width | 5.26 px | 5.02 px | -0.24 px *** |

**Interpretation**: Boundary quality is excellent. Lesion edges have correct sharpness and slightly sharper transitions than real images (narrower width). This is not a significant artifact source.

**Evidence**: `boundary_analysis/boundary_analysis_results.json`

---

## 6. Background Analysis

| Metric | Real | Synthetic | Ratio |
|--------|------|-----------|-------|
| Mean deviation from -1.0 | 0.000096 | **0.00110** | 11.4x |
| Std | 0.0018 | 0.0031 | 1.7x |
| Unique values | 208,314 | 103 | 0.0005x |

**Interpretation**: Synthetic background shows 11x more deviation from the target -1.0 value, though absolute magnitude is small (0.001). The dramatically fewer unique values in synthetic (103 vs 208K) reflects the float16 generation + dithering pipeline working correctly to remove quantization artifacts.

**Evidence**: `full_image/background/background_analysis_results.json`

---

## 7. Spatial Correlation

| Metric | Real | Synthetic | Ratio |
|--------|------|-----------|-------|
| Correlation length | 7.35 px | 7.34 px | 0.998 |

**Interpretation**: Near-perfect spatial correlation structure. The synthetic images preserve the correct autocorrelation decay, indicating appropriate spatial smoothness at macroscopic scales.

**Evidence**: `full_image/spatial_correlation/spatial_correlation_results.json`

---

## 8. XAI Analysis

### 8.1 Channel Decomposition

| Attribution Method | Image | Mask |
|-------------------|-------|------|
| Gradient magnitude fraction | **56.4%** | 43.6% |
| Ablation delta (real) | 4.57 | 0.32 |
| Ablation delta (synth) | 1.91 | 0.34 |

**Critical Finding**: The classifier primarily uses the **image channel** (56% gradient attribution) to discriminate. Ablation confirms this: removing the image channel changes the logit by 4.57 (real) vs only 0.32 for mask ablation. This means the artifact is in the **FLAIR image texture**, not the lesion mask geometry.

### 8.2 Spectral Attribution

| Band | Frequency Range | Attribution Fraction | Power Ratio |
|------|-----------------|---------------------|-------------|
| B0 | 0.016-0.036 | 34.7% | 1.07 |
| **B1** | 0.036-0.082 | **54.7%** | 1.07 |
| B2 | 0.082-0.189 | 10.1% | 0.99 |
| B3 | 0.189-0.435 | 0.5% | 0.92 |
| B4 | 0.435-1.000 | 0.0% | 0.86 |

**Concordance**: -0.70 (strong negative correlation)

**Interpretation**: The classifier focuses on **low-mid frequencies** (B0-B1, 89% attribution) where power ratios are near 1.0, NOT on high frequencies where the actual deficit exists. This negative concordance (-0.70) means the classifier learned **indirect features**: low-frequency texture patterns that correlate with the HF deficit rather than detecting the HF deficit directly.

**Implication**: Simply adding FFL to boost HF will help, but the model also needs to improve LF texture fidelity for the classifier to lose discriminative signal.

### 8.3 Feature Space Analysis

| Metric | Value |
|--------|-------|
| t-SNE silhouette | 0.448 |
| Top-5 Fisher discriminant dimensions | [56, 92, 57, 46, 109] |
| Max Fisher ratio | 6.42 |
| Significant features (FDR) | 121/128 (94.5%) |
| PCA 3D cumulative variance | 98.3% |

**Interpretation**: High t-SNE silhouette (0.448) indicates clear cluster separation. 121 of 128 learned features show significant differences between real and synthetic - the classifier has many discriminative signals to exploit.

### 8.4 Integrated Gradients

| Metric | Value |
|--------|-------|
| IG-GradCAM correlation | -0.41 |
| Attribution concentration | 0.463 |
| Image channel fraction | 52.6% |

**Interpretation**: The negative IG-GradCAM correlation (-0.41) suggests IG and GradCAM highlight different regions, indicating the classifier uses both local texture features (IG) and global structural features (GradCAM).

### 8.5 Confusion Matrix Analysis

| Category | Count | Image Fraction | Mean Prob |
|----------|-------|----------------|-----------|
| TN (real correctly classified) | 287 | 60.0% | 0.895 |
| TP (synth correctly classified) | 119 | 61.6% | 0.051 |
| FP (synth misclassified as real) | **0** | - | - |
| FN (real misclassified as synth) | 158 | 61.2% | 0.284 |

**Critical Finding**: **Zero false positives** (0% FP rate) - no synthetic image fooled the classifier into thinking it was real. This confirms the synthetic images have a universal detectable signature.

**Evidence**: `channel_decomposition/`, `spectral_attribution/`, `feature_space/`, `integrated_gradients/`, `confusion_stratified/`

---

## 9. Artifact Ranking (by Category Score)

| Rank | Category | Score | Key Metric |
|------|----------|-------|------------|
| 1 | Texture Quality | 0.362 | dissimilarity d=0.47 |
| 2 | Distribution Accuracy | 0.297 | lesion Wasserstein=0.062 |
| 3 | High Frequency | 0.213 | wavelet HH L1=0.787 |
| 4 | Background Integrity | 0.157 | deviation ratio=11.4x |
| 5 | Boundary Quality | 0.054 | width deficit=0.24px |
| 6 | Spectral Fidelity | 0.027 | slope diff=0.081 |
| 7 | Spatial Coherence | 0.003 | correlation ratio=0.998 |

---

## 10. Root Cause Analysis

The diagnostics reveal a **coherent artifact pattern** with a single root cause:

### Primary Cause: High-Frequency Energy Deficit

1. **Wavelet HH L1 = 0.787** (21% deficit at finest scale)
2. **Spectral band 4 ratio = 0.86** (14% deficit at highest frequencies)
3. **Spectral slope = +0.081** steeper (faster HF rolloff)

### Secondary Manifestations:

1. **GLCM texture changes** (dissimilarity d=0.47, homogeneity d=-0.30)
   - Missing HF texture makes images more homogeneous

2. **Gradient magnitude reduction** (d=0.45)
   - Fewer sharp edges due to HF deficit

3. **Negative spectral concordance** (-0.70)
   - LF texture patterns correlate with HF deficit
   - Classifier exploits this indirect signal

### Why This Happens:

The Lp norm loss (p=1.5) with x0 prediction is excellent for preserving overall structure but:
- Lp norm operates in spatial domain, not frequency domain
- High frequencies have low per-pixel magnitude (spread across many coefficients)
- Lp gradient ∝ |x|^(p-1) = |x|^0.5 gives equal weight to all errors regardless of frequency
- MSE would be worse (quadratic penalty on large LF errors), but neither explicitly targets HF

**The model learns to minimize spatial loss by slightly smoothing - reducing HF noise also reduces spatial error**.

---

## 11. Recommendations

### R1: Add Focal Frequency Loss (FFL) - Priority: HIGH

**Rationale**: FFL explicitly penalizes frequency-domain errors with adaptive weighting. The focal mechanism (|error|^α weighting) emphasizes hard-to-reconstruct frequencies (high frequencies where ratio=0.86).

**Suggested Parameters**:
```yaml
loss:
  mode: "mse_lp_norm_ffl_groups"
  ffl:
    enabled: true
    alpha: 1.35       # Slightly focal (>1) to emphasize HF errors
    patch_factor: 1   # No patching - full 160x160 FFT for best frequency resolution
    log_matrix: true  # Numerical stability for MRI's wide dynamic range
```

**Expected Impact**: wavelet_L1_HH_ratio: 0.787 → >0.85

**Literature**: Jiang et al. "Focal Frequency Loss for Image Reconstruction and Synthesis" (ICCV 2021)

### R2: Enable Group Uncertainty Weighting - Priority: HIGH

**Rationale**: Kendall uncertainty weighting learns optimal balance between Lp (spatial) and FFL (frequency) losses. Initial conservative FFL weighting allows spatial structure to establish first.

**Suggested Parameters**:
```yaml
loss:
  group_uncertainty_weighting:
    enabled: true
    initial_log_vars: [0.0, 1.0]  # FFL starts at 0.37 precision (exp(-1.0))
    learnable: true
```

**Literature**: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)

### R3: Increase Self-Conditioning Probability - Priority: MEDIUM

**Rationale**: Spectral attribution concordance=-0.70 shows the classifier detects LF texture patterns. Higher self-conditioning (0.5→0.8) provides iterative refinement of the x0 estimate, improving fine texture consistency.

**Suggested Parameters**:
```yaml
training:
  self_conditioning:
    probability: 0.8  # Up from 0.5
```

**Literature**: Chen et al. "Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning" (2022)

### R4: Reduce Sampling Stochasticity - Priority: MEDIUM

**Rationale**: Lower eta preserves learned texture structure by reducing noise injection during DDIM sampling. The LF texture anomaly (concordance=-0.70) may benefit from more deterministic trajectories.

**Suggested Parameters**:
```yaml
sampler:
  eta: 0.1  # Down from 0.2
```

### R5: Increase Lesion Weighting - Priority: MEDIUM

**Rationale**: Lesion Wasserstein distance (0.062) is 3x higher than overall (0.022). Upweighting lesion regions ensures the model attends to lesion texture quality.

**Suggested Parameters**:
```yaml
loss:
  lesion_weighted_image:
    lesion_weight: 3.0  # Up from 1.0
  lesion_weighted_mask:
    lesion_weight: 3.0
```

### R6: Extended Training with Patience - Priority: LOW

**Rationale**: Dual-loss (spatial + frequency) optimization has a more complex landscape requiring longer convergence.

**Suggested Parameters**:
```yaml
training:
  max_epochs: 1000      # Up from 500
  early_stopping:
    patience: 100       # Up from 25
```

---

## 12. Summary

The x0_lp_1.5 model achieves excellent overall quality but contains a detectable high-frequency texture deficit. The artifact is:

1. **Quantified**: 21% HF energy deficit (wavelet HH L1 = 0.787)
2. **Localized**: Image channel, not mask (56% attribution)
3. **Manifested**: As reduced texture dissimilarity and gradient magnitude
4. **Indirectly learned**: Classifier exploits correlated LF patterns (concordance=-0.70)

**Recommendation Summary**:
| Change | Parameter | From | To | Evidence |
|--------|-----------|------|-----|----------|
| Add FFL | loss.mode | mse_lp_norm | mse_lp_norm_ffl_groups | HH_L1=0.787 |
| FFL alpha | loss.ffl.alpha | - | 1.35 | Spectral band4=0.86 |
| Group weighting | group_uncertainty_weighting | false | true | Multi-task balancing |
| Self-conditioning | self_conditioning.probability | 0.5 | 0.8 | Concordance=-0.70 |
| Sampling eta | sampler.eta | 0.2 | 0.1 | LF texture refinement |
| Lesion weight | lesion_weight | 1.0 | 3.0 | Lesion W=0.062 |

**Expected Outcome**: Reduce discriminability from AUC=0.996 toward AUC<0.90 (non-trivial classification) by addressing the HF deficit.
