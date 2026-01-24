# JS-DDPM Diagnostic Report: What Makes Synthetic Images Differ From Real Ones?

## Executive Summary

Across 9 experiments (3 prediction types x 3 Lp norms), the dominant source of
real-vs-synthetic distinguishability is the **prediction type parameterization**.
The x0 (sample) prediction type produces images nearly indistinguishable from real
data, while epsilon prediction generates catastrophic spectral artifacts.

**Updated finding from XAI analysis**: The remaining x0 deficit has **two
components**: (1) a measurable HF energy deficit (wavelet HH ratio=0.68,
addressable via FFL) and (2) subtle LF texture anomalies that the classifier
actually exploits (spectral attribution concordance=-0.9, classifier focuses
69.6% of attention on lowest-frequency band). The LF anomalies represent
distributed per-pixel texture incoherence (IG-GradCAM correlation=-0.04,
Fisher max ratio only 3.49) that requires **iterative refinement via
self-conditioning (80%) and near-deterministic sampling (eta=0.05)** in addition
to FFL.

---

## Part I: Data-Driven Findings

### 1. Prediction Type Dominates All Other Factors

| Metric | x0 (best) | velocity | epsilon | Ideal |
|--------|-----------|----------|---------|-------|
| Spectral slope diff | 0.030 | 0.113 | 1.750 | 0.0 |
| JS divergence | 0.0002 | 0.008 | 0.114 | 0.0 |
| Boundary sharpness | 0.972 | 0.818 | 0.595 | 1.0 |
| Spatial corr ratio | 0.998 | 0.918 | 0.360 | 1.0 |
| Background std | 0.003 | 0.017 | 0.016 | 0.002 |
| Wasserstein (all) | 0.029 | 0.240 | 0.588 | 0.0 |
| Pixel frac signif | 0.019 | 0.250 | 0.607 | 0.0 |
| HF ratio diff | -0.15 | 0.79 | 40.14 | 0.0 |
| Texture KS | 0.145 | 0.542 | 0.942 | 0.0 |
| Overall severity | 0.025 | 0.254 | 0.915 | 0.0 |

The **prediction type alone accounts for >95% of the variance** in artifact severity.
Lp norm has a secondary effect within each prediction type group.


### 2. Lp Norm Effect: Higher p Causes Over-Smoothing

Within x0 prediction, increasing p from 1.5 to 2.5 systematically degrades
high-frequency content:

| Metric | p=1.5 | p=2.0 | p=2.5 |
|--------|-------|-------|-------|
| Wavelet HH L1 ratio | 0.68 | 0.52 | 0.30 |
| GLCM contrast | 75.98 | 70.54 | 63.87 |
| GLCM dissimilarity | 3.76 | 3.42 | 2.93 |
| Gradient magnitude | 0.667 | 0.606 | 0.518 |
| Spectral slope diff | 0.030 | 0.051 | 0.069 |

Real data reference: contrast=81.67, dissimilarity=4.18, gradient=0.736.

The mathematical explanation is straightforward: the Lp norm loss `L(x) = |x|^p`
has gradient `p|x|^{p-1}sign(x)`. For p > 2, the gradient magnitude grows
super-linearly with error, creating a steep penalty landscape that forces the
model toward the conditional mean (over-smoothing). For p < 2, the sub-linear
gradient growth tolerates large pointwise errors, preserving sharp features.


### 3. Epsilon Prediction: Catastrophic Spectral Failure

Epsilon prediction exhibits:
- **39x excess high-frequency power** (global HF ratio)
- Spectral slope 1.15 vs real 2.91 (nearly flat spectrum instead of 1/f^beta)
- GLCM correlation 0.22 vs real 0.93 (texture is noise-like)
- 95.6% of pixels significantly different from real (FDR-corrected)

This is NOT a model capacity issue. The explanation lies in the signal recovery
math. With epsilon prediction, x0 recovery requires division by `sqrt(alpha_bar_t)`:

```
x0_hat = (x_t - sqrt(1-alpha_t) * eps_hat) / sqrt(alpha_t)
```

At high t (e.g., t=900 with cosine schedule), `sqrt(alpha_t) ~ 0.05`. A 5% error
in epsilon prediction becomes a 100% error in x0. The model compensates by
learning to predict noise that, when the formula is applied, produces reasonable
x0 — but this forces it to implicitly encode x0 structure into the noise
prediction, creating spectral artifacts.


### 4. Velocity Prediction: Intermediate but Background-Noisy

Velocity prediction achieves reasonable spectral slopes (0.11-0.17 diff) but has:
- Background std 0.015-0.017 (5x higher than x0)
- Background deviation from -1.0: 0.025-0.030 (25x higher than x0)
- Boundary sharpness ratio 0.67-0.90 (variable across Lp norms)

The background issue stems from the v-prediction formulation:
`v = sqrt(alpha_t)*eps - sqrt(1-alpha_t)*x0`. At intermediate t, both terms are
significant and errors can leak into background regions where x0 should be
exactly -1.0. With x0 prediction, the model directly learns the -1.0 background,
making it trivially correct.


### 5. The Remaining x0 Deficit: High-Frequency Under-Generation

Even the best model (x0_lp_1.5) has measurable deficits:
- Wavelet HH (diagonal, finest level): energy ratio 0.68 (32% deficit)
- Wavelet HL (horizontal, finest level): energy ratio 0.83 (17% deficit)
- GLCM dissimilarity: 3.76 vs 4.18 (10% deficit)
- Gradient magnitude: 0.667 vs 0.736 (9% deficit)

This is the well-documented **spectral bias of neural networks** (Rahaman et al.,
2019): neural networks learn low-frequency functions first, and diffusion models
inherit this tendency. The UNet architecture compounds it through upsampling
layers that struggle to generate high-frequency content (Durall et al., 2020).


### 6. XAI Evidence: The Classifier Uses Image Channel, Not Mask

Channel decomposition analysis reveals the decisive role of each input channel
in the classifier's decision:

| Experiment | Image Fraction | Ablation Ratio (img/mask) | Dominant |
|-----------|---------------|---------------------------|----------|
| x0_lp_1.5 | 0.586 | **8.87** | image |
| velocity_lp_2.0 | 0.419 | 0.36 | mask |
| epsilon_lp_1.5 | 0.510 | 0.41 | image (barely) |

For x0_lp_1.5, ablating the image channel causes 8.87x more logit change than
ablating the mask channel (delta 5.96 vs 0.67). This means:

1. **The mask is already near-perfect for x0** — ablating it barely affects
   classification. The joint synthesis correctly couples mask and image.
2. **Image texture is the sole remaining distinguishing factor** — all remaining
   artifacts are in the FLAIR channel, not the lesion segmentation.
3. **Velocity/epsilon have primarily mask artifacts** — ablation ratio 0.26-0.49
   and mask_ablation_delta 4.0-4.4 (vs x0's 0.67). Their masks are trivially
   distinguishable from real masks.

Per-class breakdown for x0_lp_1.5:
- Real images: ablation delta 8.86 (image), 0.56 (mask)
- Synthetic images: ablation delta 3.06 (image), 0.78 (mask)

The asymmetry (8.86 vs 3.06) indicates the classifier relies more on image-channel
cues when classifying real images correctly than when classifying synthetic ones.


### 7. XAI Evidence: The Classifier Focuses on LOW Frequencies (Not HF)

This is the most critical new finding. Spectral attribution measures which
frequency bands the classifier uses for its decision:

| Experiment | Band 0 (LF) | Band 1 | Band 2 | Band 3 | Band 4 (HF) | Concordance |
|-----------|-------------|--------|--------|--------|-------------|-------------|
| x0_lp_1.5 | **69.6%** | 23.8% | 6.2% | 0.3% | 0.05% | **-0.9** |
| velocity_lp_2.0 | 49.8% | 36.3% | 13.1% | 0.7% | 0.07% | +0.9 |
| epsilon_lp_1.5 | 61.9% | 24.9% | 12.3% | 0.9% | 0.04% | 0.0 |

The **concordance** between "where the classifier pays attention" and "where the
real-synthetic power ratios differ from 1.0":
- **x0: concordance = -0.9** (strongly anti-correlated!)
- **velocity: concordance = +0.9** (positively correlated)
- **epsilon: concordance = 0.0** (no alignment — everything is wrong)

**Interpretation for x0_lp_1.5**: The classifier pays 69.6% of its attribution
to band 0 (lowest frequencies, f < 0.036), yet the statistical power ratios at
those frequencies are near-perfect (band 0 ratio = 0.994, band 1 = 1.023).
The classifier is detecting **structural/textural anomalies in low-frequency
content that are invisible to simple energy metrics**.

This means the remaining x0 deficit has two components:
1. **HF energy deficit** (wavelet HH=0.68) — detectable by energy statistics
   but NOT what the classifier uses
2. **LF texture anomalies** — undetectable by energy statistics but IS what the
   classifier exploits (phase structure, local GLCM-type patterns, spatial
   correlations at coarse scales)

**Consequence**: FFL will correct the HF energy deficit (component 1) but likely
will NOT eliminate classifier discriminability (component 2). The LF anomalies
require a different approach: perceptual losses, self-conditioning refinement,
or architectural changes that improve local texture coherence.

For velocity models, the positive concordance (+0.9) means the classifier uses
exactly the frequency bands where power differs — a simpler pattern where
spectral correction would be directly beneficial.


### 8. XAI Evidence: Feature Space Separability Quantifies the Gap

The classifier's 128-dimensional GAP features reveal how easily real and
synthetic samples are separated:

| Experiment | Fisher Max | Fisher Mean | Silhouette | Cosine Distance |
|-----------|-----------|-------------|------------|-----------------|
| x0_lp_1.5 | **3.49** | **1.08** | **0.356** | **0.035** |
| velocity_lp_2.0 | 139.7 | 35.3 | 0.492 | 0.323 |
| epsilon_lp_1.5 | 341.5 | 56.3 | 0.509 | 0.358 |

The x0 model achieves **100x less separability** than epsilon. Key observations:

1. **Fisher ratio max 3.49** — only 5 of 128 dimensions have Fisher ratio > 3.0.
   For epsilon, ALL 128 dimensions have Fisher ratio > 20. This means x0's
   artifacts occupy a tiny subspace of the representation.

2. **Cosine distance 0.035** — real and synthetic feature vectors point in nearly
   identical directions (angle ~ 2 degrees). For epsilon the angle is ~21 degrees.

3. **PCA structure differs fundamentally**:
   - x0: PC1=74.9%, PC2=19.4% — the representation has rich structure but the
     classes aren't separated along principal directions
   - epsilon: PC1=97.0% — nearly all variance is in one discriminative direction

4. **Fraction significant features: 99.2% for ALL experiments** — even x0 has
   127/128 features significantly different between classes (FDR-corrected).
   The separation is real but minute in magnitude (Fisher ratio 1.08 mean).

These results quantify the target: reducing Fisher max from 3.49 to <2.0 would
make the representation nearly indistinguishable.


### 9. XAI Evidence: Integrated Gradients Reveal Distributed, Per-Pixel Artifacts

Integrated Gradients provides per-pixel, axiom-satisfying attributions:

| Experiment | IG Image Frac | Concentration | IG-CAM Corr | Completeness |
|-----------|--------------|---------------|-------------|-------------|
| x0_lp_1.5 | 0.526 | **0.186** | **-0.04** | **1080** |
| velocity_lp_2.0 | 0.420 | 0.101 | +0.016 | 17.1 |
| epsilon_lp_1.5 | 0.452 | 0.118 | -0.014 | 10.5 |

Key findings for x0_lp_1.5:

1. **Completeness = 1080**: The sum of attributions equals F(x) - F(baseline).
   The x0 model produces very confident classifier logits despite subtle artifacts.
   This is 100x larger than epsilon/velocity (10-17), confirming the classifier
   is highly certain even for the best model.

2. **IG-GradCAM correlation = -0.04**: Per-pixel attributions (IG) are
   uncorrelated with spatial saliency maps (GradCAM). This means the classifier
   doesn't rely on specific spatial regions but on **distributed, fine-grained
   per-pixel texture** patterns across the entire patch.

3. **Concentration = 0.186** (Gini coefficient): Higher than velocity/epsilon
   (0.09-0.12), suggesting x0's artifacts are slightly more spatially focused
   than the globally-distributed artifacts of other models. The artifacts have
   some spatial structure (possibly lesion-boundary related).

4. **IG image fraction = 0.526**: More balanced than channel decomposition
   (0.586). The difference arises because IG measures pixel-level contribution
   whereas gradient magnitude measures sensitivity. The mask carries more
   subtle per-pixel information than gradient magnitude alone suggests.

**Synthesis**: The classifier exploits fine-grained, spatially-distributed
texture patterns in the image channel, concentrated in low-frequency bands.
These are not correctable by simple spectral energy matching (FFL) alone.

---

## Part II: Proposed Changes

### Revised Understanding from XAI

The XAI analyses reveal that the remaining x0_lp_1.5 deficit has **two distinct
components** with different remedies:

| Component | Evidence | Remedy |
|-----------|----------|--------|
| HF energy deficit (32%) | Wavelet HH ratio=0.68, spectral slope diff=0.030 | FFL (frequency-domain loss) |
| LF texture anomaly | Concordance=-0.9, classifier uses band 0 (69.6%), distributed per-pixel | Self-conditioning, longer training, reduced eta |

The HF deficit is **measurable but not the classifier's main discriminative cue**.
The LF texture anomaly is what the classifier actually exploits. Both must be
addressed for optimal results.


### Parameter Changes (in `slurm/proposal.yaml`)

| # | Parameter | Baseline | Proposed | Justification |
|---|-----------|----------|----------|---------------|
| C1 | prediction_type | sample | sample | Confirmed best; 37x better severity than epsilon |
| C2 | loss.mode | mse_lp_norm | mse_lp_norm_ffl_groups | FFL corrects measurable HF energy deficit |
| C3 | lp_norm.p | 1.5 | 1.5 | Confirmed best HF preservation by data |
| C4 | ffl.alpha | - | 1.0 | Moderate focal weighting; avoids HF noise artifacts |
| C4 | ffl.patch_factor | - | 4 | Local frequency analysis at lesion-relevant scales |
| C5 | group_uncertainty | off | on | Auto-balances spatial vs frequency loss |
| C6 | lesion_weight | 1.0 | 2.0 | Focus gradient on rare lesion pixels |
| C7 | anatomical_cond | off | cross_attention | Spatial guidance for texture coherence |
| C8 | batch_size | 16 | 32 | Stabilizes frequency-domain gradients |
| C9 | num_inference_steps | 300 | 500 | More refinement steps for texture detail |
| C9 | eta | 0.2 | 0.05 | Near-deterministic sampling preserves LF texture |
| C10 | ema.decay | 0.999 | 0.9995 | Smoother weights with dual-loss landscape |
| C11 | self_cond.probability | 0.5 | 0.8 | Primary mechanism for LF texture refinement |
| C12 | max_epochs | 500 | 1000 | Combined loss + self-cond refinement needs time |
| C12 | patience | 25 | 60 | Combined loss converges more slowly |


### Methodological/Architectural Changes

#### M1: Focal Frequency Loss Integration

**Problem**: The Lp norm loss treats all spatial frequencies equally. The wavelet
analysis shows a 32% HF energy deficit at the finest scale (HH ratio=0.68).

**Solution**: Add FFL (Jiang et al., 2021) as a complementary loss operating in
frequency space. FFL computes the 2D FFT of predicted and target x0, then
adaptively weights the frequency-domain MSE to focus on components with the
largest errors.

**Math**:
```
L_FFL = sum_{u,v} w(u,v) * |F_pred(u,v) - F_target(u,v)|^2

where w(u,v) = [|F_pred - F_target| / max(|F_pred - F_target|)]^alpha
```

**XAI-informed nuance**: The spectral attribution analysis shows the classifier
focuses 69.6% of its attention on band 0 (lowest frequencies) with concordance
= -0.9. This means FFL will correct the measurable HF energy deficit but will
**not eliminate classifier discriminability**. The remaining LF texture anomalies
require complementary approaches (see M4, M6).

FFL is still essential because:
1. The wavelet HH deficit (0.68) is a real physical artifact visible in the data
2. Even if the classifier doesn't primarily use it, a reader/clinician might
3. Correcting HF content improves visual quality and downstream task performance
4. FFL indirectly improves phase coherence at high frequencies through the
   complex-valued FFT loss (it penalizes both magnitude AND phase errors)

**Expected impact**: Wavelet HH ratio 0.68 -> 0.85+. Partial reduction in
classifier separability (Fisher ratio 3.49 -> ~2.5), but LF anomalies will persist
until self-conditioning and sampling refinements also take effect.

**Why alpha=1.0**: Our deficit is moderate (ratio 0.68, not 0.0). Linear focal
weighting avoids overshooting into HF noise (cf. epsilon's 39x excess).

**Why patch_factor=4**: Lesions are 10-50px structures in 160x160 images. Patches
of 40x40 isolate lesion-scale texture from global anatomy, matching the spatial
scale where the classifier finds distributed per-pixel artifacts (IG concentration
= 0.186, moderate spatial focus).

**Literature**:
- Jiang et al. (2021), "Focal Frequency Loss for Image Reconstruction and
  Synthesis", ICCV.
- Durall et al. (2020), "Watch your Up-Convolution: CNN Based Generative Deep
  Models Fail to Reproduce Spectral Distributions".
- Chandrasegaran et al. (2021), "A Closer Look at Fourier Spectrum
  Discrepancies for CNN-generated Images Detection".


#### M2: x0 Prediction as Optimal Parameterization for Medical Imaging

**Problem**: Standard diffusion model implementations default to epsilon (noise)
prediction. Our data shows this is catastrophically wrong for medical image
synthesis.

**Explanation**: Medical images have well-defined structure (anatomical prior,
bounded intensities, binary masks). With x0 prediction, the model can be directly
evaluated on sample quality at each timestep. The training signal is:

```
L = E_{t,x0,eps}[||x0 - f_theta(sqrt(alpha_t)*x0 + sqrt(1-alpha_t)*eps, t)||^p]
```

This is a direct image reconstruction objective. The model receives the clearest
possible gradient signal about what the output should look like.

With epsilon prediction, the loss is on noise, and x0 quality is only indirectly
optimized. The model must solve an inverse problem (predict noise whose removal
gives good x0), which introduces the spectral artifacts we observe.

**Data evidence**: The transition epsilon -> velocity -> x0 shows monotonic
improvement on EVERY metric. This is not random; it reflects the signal recovery
chain becoming more direct.

**When x0 prediction can be problematic**: At very high noise levels (t ~ T),
x_t is nearly pure noise and predicting x0 directly is ill-conditioned. However,
with the cosine schedule and self-conditioning, this issue is mitigated because:
1. Cosine schedule maintains higher SNR at all t compared to linear
2. Self-conditioning provides the model with its previous x0 estimate as input

**Literature**:
- Salimans & Ho (2022), "Progressive Distillation for Fast Sampling of Diffusion
  Models". Shows x0 prediction is more numerically stable.
- Ramesh et al. (2022), "Hierarchical Text-Conditional Image Generation with
  CLIP Latents" (DALL-E 2). Uses x0 prediction.
- Chen et al. (2022), "Analog Bits: Generating Discrete Data using Diffusion
  Models with Self-Conditioning". x0 prediction + self-conditioning.


#### M3: Kendall Uncertainty Weighting for Multi-Objective Optimization

**Problem**: The spatial loss (Lp norm) and frequency loss (FFL) have different
magnitudes and convergence rates. Fixed weighting requires manual tuning and
may not be optimal throughout training.

**Solution**: Kendall et al. (2018) homoscedastic uncertainty weighting learns
the optimal relative weighting automatically:

```
L_total = (1/2*sigma_1^2) * L_spatial + (1/2*sigma_2^2) * L_freq
          + log(sigma_1) + log(sigma_2)
```

The regularization terms `log(sigma_i)` prevent the model from trivially
minimizing all losses by setting sigma -> infinity.

**Initialization**: We set `log_var_0 = 0.0` (spatial, precision=1.0) and
`log_var_1 = 1.0` (FFL, precision=0.37). This means the spatial loss dominates
early training (when the model is still learning basic structure), and FFL
gradually takes effect as spectral errors become the dominant remaining issue.

**Literature**:
- Kendall et al. (2018), "Multi-Task Learning Using Uncertainty to Weigh
  Losses for Scene Geometry and Semantics", CVPR.


#### M4: Self-Conditioning as Primary LF Texture Refinement Mechanism

**Problem**: The XAI analysis reveals the classifier detects LF texture anomalies
(band 0 carries 69.6% of attribution, concordance=-0.9). These are subtle
phase/structure artifacts that spectral energy losses cannot address. The current
50% self-conditioning probability limits the model's ability to iteratively refine
these fine-grained texture patterns.

**Proposed change**: Increase to **80%** (up from the previously proposed 70%).

**Rationale**: Self-conditioning is the most direct mechanism for addressing
distributed per-pixel texture artifacts because:

1. **Iterative refinement corrects local texture**: The model's first x0 estimate
   captures global structure. With self-conditioning, the second pass receives this
   estimate as additional input and can focus on correcting local pixel-level
   texture patterns — exactly the LF anomalies the spectral attribution identifies.

2. **The IG analysis supports this**: IG concentration=0.186 and IG-GradCAM
   correlation=-0.04 indicate the artifacts are fine-grained and per-pixel, not
   region-specific. Self-conditioning addresses per-pixel refinement by providing
   the model with spatial context about its own output quality.

3. **Channel decomposition shows image-only focus needed**: With ablation ratio
   8.87 for x0, the mask is already good. Self-conditioning refinement should
   primarily improve the image channel's local texture coherence.

4. **Training at 80% means**: 80% of steps see the previous x0 estimate, training
   the model to be an excellent refiner. At inference, DDIM always uses the
   previous x0 estimate (100% self-conditioning), so higher training probability
   better matches the inference regime.

**Literature**:
- Chen et al. (2022), "Analog Bits". Introduced self-conditioning at p=0.5.
- Jabri et al. (2022), "Scalable Adaptive Computation for Iterative Generation".
  Shows iterative refinement improves fine detail.
- Watson et al. (2022), "Learning to Efficiently Sample from Diffusion
  Probabilistic Models". Demonstrates that refinement-focused training improves
  sample quality in fewer steps.


#### M5: Anatomical Conditioning via Cross-Attention

**Problem**: Without spatial guidance, the model must independently learn where
the brain boundary is, what the background should be, and where lesions can
appear. The XAI analysis reveals the classifier uses distributed per-pixel
texture patterns (IG concentration=0.186) — providing spatial context via
anatomical priors should help the model produce locally coherent texture.

**Proposed change**: Enable cross-attention anatomical conditioning with the
existing `AnatomicalPriorEncoder`.

**XAI-informed rationale**: The z-bin prior provides a spatial map of valid
brain regions. Cross-attention (rather than concatenation) allows the model to
selectively attend to this map at multiple UNet resolution levels. This is
particularly relevant because:

1. The channel decomposition shows per-zbin variation in image_fraction (0.579
   at z=0 to 0.589 at z=19). Different anatomical levels have different texture
   characteristics; providing the anatomical prior helps the model produce
   level-appropriate texture.
2. The IG concentration (0.186) suggests moderately spatially-structured artifacts.
   Cross-attention provides resolution-matched spatial guidance at exactly the
   scale where artifacts concentrate.
3. Background is already near-perfect for x0 (std=0.003), so the main benefit
   is improved texture coherence within brain parenchyma, not background correction.

**Expected impact**: Primarily improves LF texture coherence by providing the
model with spatial context for texture generation. Secondary benefits:
- Boundary sharpness (0.97 -> 0.99+)
- Spatial correlation structure (0.998 -> closer to 1.0)
- Reduced per-pixel texture artifacts in anatomically-guided regions


#### M6: Near-Deterministic Sampling for LF Texture Preservation

**Problem**: The current eta=0.2 in DDIM sampling introduces stochastic noise
at each step. The XAI analysis shows the remaining artifacts are subtle LF texture
patterns (band 0 attribution 69.6%, IG concentration 0.186). Stochastic noise at
each step corrupts the very texture coherence the model learns to produce.

**Proposed change**: eta=**0.05**, inference_steps=500.

**XAI-informed rationale**: The spectral attribution concordance=-0.9 indicates
the classifier detects LF structure anomalies. These are precisely the features
most sensitive to sampling noise injection:

1. **LF texture = spatial correlations over 5-20px**: Adding noise (eta > 0)
   at each step disrupts local spatial correlations, introducing exactly the
   type of texture incoherence the classifier detects.

2. **Near-deterministic (eta=0.05) preserves learned structure**: The model
   learns correct LF texture through self-conditioning. Deterministic sampling
   preserves this learned texture through the full denoising trajectory.

3. **500 steps with low eta**: More steps reduce discretization error. With
   eta=0.05, the small residual stochasticity across 500 steps provides minimal
   diversity while maintaining texture coherence.

4. **The IG completeness (1080)** tells us the model produces very confident
   predictions. Near-deterministic sampling respects this confidence rather than
   perturbing it with noise.

**Diversity concern**: For medical imaging, precision matters more than diversity.
The primary use case is generating realistic training augmentation, where quality
is more important than novelty.

---

## Part III: Risk Analysis

| Change | Risk | Mitigation |
|--------|------|------------|
| FFL addition | Over-correction -> HF noise | alpha=1.0 (moderate), monitor wavelet HH ratio stays < 1.2 |
| FFL alone insufficient | Won't fix LF texture (concordance=-0.9) | Combined with self-cond (80%) + low eta |
| Higher self-cond prob (80%) | Mode collapse, slower convergence | Longer training (1000 epochs), diversity monitoring |
| Very low eta (0.05) | Near-deterministic = less augmentation diversity | Sample many seeds, verify KID stability |
| Anatomical conditioning | Additional parameters | Well-tested in prior experiments, small encoder |
| Higher EMA decay | Slow adaptation | Offset by longer training |
| Combined changes | Hard to attribute improvements | Ablation study already planned in slurm/ |


## Part IV: Validation Protocol

After training the proposal config, run the full diagnostics pipeline including
XAI analyses:

```bash
# Run full diagnostics (includes XAI: channel-decomp, spectral-attr,
# feature-space, integrated-gradients, reports)
python -m src.classification.diagnostics run-all \
    --config src/classification/diagnostics/config/diagnostics.yaml \
    --experiment proposal_x0_ffl --gpu 0

# Aggregate including XAI metrics
python -m src.classification.diagnostics aggregate \
    --config src/classification/diagnostics/config/diagnostics.yaml

# Paired comparison with next-experiment recommendation
python -m src.classification.diagnostics paired-comparison \
    --config src/classification/diagnostics/config/diagnostics.yaml
```

### Success Criteria (compare to x0_lp_1.5 baseline)

**Tier 1: Statistical Metrics** (HF correction via FFL):
1. Wavelet HH L1 energy ratio > 0.85 (baseline: 0.68)
2. GLCM dissimilarity > 3.9 (baseline: 3.76, real: 4.18)
3. Gradient magnitude > 0.700 (baseline: 0.667, real: 0.736)
4. Spectral slope diff < 0.025 (baseline: 0.030)

**Tier 2: XAI Metrics** (LF texture improvement via self-cond + low eta):
5. Fisher max ratio < 2.5 (baseline: 3.49) — less separable features
6. Spectral attribution concordance closer to 0.0 (baseline: -0.9) — less
   exploitable LF structure
7. IG concentration < 0.15 (baseline: 0.186) — less spatially focused artifacts
8. Image ablation delta < 4.0 (baseline: 5.96) — less image-channel reliance

**Tier 3: Overall Quality**:
9. Overall diagnostic severity < 0.015 (baseline: 0.025)
10. Classification AUC after dithering < 0.70 (non-trivial quality threshold)
11. t-SNE silhouette < 0.30 (baseline: 0.356) — weaker clustering

**Tier 4: Non-Regression**:
12. Background std < 0.005 (baseline: 0.003) — no noise introduction
13. Mask ablation delta remains < 1.0 (baseline: 0.67) — mask quality preserved
14. Boundary sharpness ratio > 0.95 (baseline: 0.97) — edges not degraded


## Part V: References

1. Chen, T., Zhang, R., & Hinton, G. (2022). Analog Bits: Generating Discrete
   Data using Diffusion Models with Self-Conditioning. arXiv:2208.04202.

2. Chandrasegaran, K., et al. (2021). A Closer Look at Fourier Spectrum
   Discrepancies for CNN-generated Images Detection. CVPR.

3. Durall, R., et al. (2020). Watch your Up-Convolution: CNN Based Generative
   Deep Models Fail to Reproduce Spectral Distributions. CVPR.

4. Jiang, L., Dai, B., Wu, W., & Loy, C.C. (2021). Focal Frequency Loss for
   Image Reconstruction and Synthesis. ICCV.

5. Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using
   Uncertainty to Weigh Losses for Scene Geometry and Semantics. CVPR.

6. Nichol, A. & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic
   Models. ICML.

7. Rahaman, N., et al. (2019). On the Spectral Bias of Neural Networks. ICML.

8. Ramesh, A., et al. (2022). Hierarchical Text-Conditional Image Generation
   with CLIP Latents (DALL-E 2). arXiv:2204.06125.

9. Salimans, T. & Ho, J. (2022). Progressive Distillation for Fast Sampling
   of Diffusion Models. ICLR.

10. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic
    Models. NeurIPS.

11. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for
    Deep Networks. ICML.

12. Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). Deep Inside Convolutional
    Networks: Visualising Image Classification Models and Saliency Maps. ICLR
    Workshop.

13. Yosinski, J., et al. (2015). Understanding Neural Networks Through Deep
    Visualization. ICML Workshop.

14. Selvaraju, R.R., et al. (2017). Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization. ICCV.

15. Watson, D., Ho, J., Norouzi, M., & Chan, W. (2022). Learning to Efficiently
    Sample from Diffusion Probabilistic Models. arXiv:2106.03802.

16. Jabri, A., Fleet, D., & Chen, T. (2022). Scalable Adaptive Computation for
    Iterative Generation. arXiv:2212.11972.
