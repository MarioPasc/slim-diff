# Study 1.1 — Zero-Coupling Baseline (Two Independent DDPMs)

## Objective

Train a **zero-coupling control** consisting of two independent single-channel DDPMs (one for the FLAIR image, one for the lesion mask) that share only the conditioning embedding. At inference time, sample both with the same conditioning token and concatenate the outputs. The combined parameter count must equal that of the shared and decoupled variants (≈26.9M) to ensure capacity is matched across the three configurations.

This produces the **third point of the coupling continuum**:

> zero coupling (this study) → bottleneck-only coupling (decoupled, existing) → full coupling (shared, existing)

## Scientific rationale

The shared-bottleneck and decoupled-bottleneck variants both perform joint training on a 2-channel sample `x₀ = [I, M]ᵀ` and share the encoder/decoder. They differ only in whether the bottleneck representation is shared. A reviewer can legitimately object that *neither* variant tests whether **joint training itself** is necessary — the shared-vs-decoupled comparison is *internal* to a single architectural family.

The zero-coupling baseline isolates the contribution of joint training. Formally, it factorises the modelled distribution as
$$p_\theta(I, M \mid c) \;\approx\; p_{\theta_I}(I \mid c)\, p_{\theta_M}(M \mid c),$$
i.e. **conditional independence given `c`**. Any improvement of the shared/decoupled variants over this baseline must come from the cross-channel coupling that joint training enables, not from the conditioning signal alone. A useful reference for this style of factorised baseline is Esser et al., *Taming Transformers* (CVPR 2021), §4 (independent VQ-codebooks ablation), and the joint-vs-separate factorisation discussion in Sohl-Dickstein et al., *Deep Unsupervised Learning Using Nonequilibrium Thermodynamics* (ICML 2015).

## Phase 0 — Codebase exploration (mandatory before any coding)

Read, do not skim. Produce a short `EXPLORATION.md` summarising what you found before writing any new code. Do **not** reimplement anything that already exists.

### Required reading

1. **Diffusion model definition.** Locate the `DiffusionModelUNet` instantiation site (likely `src/diffusion/model/` or imported from `monai.networks.nets`). Identify:
   - How `in_channels=2`, `out_channels=2` are wired.
   - Where conditioning embeddings (`cemb`, `temb`) are added (AdaGN injection points).
   - Whether the model exposes a `forward(x, t, c)` signature you can call directly.

2. **Training entrypoint.** The CLI command `slimdiff-train` is registered. Find its setup (`pyproject.toml` or `setup.py` console_scripts) and follow the call chain to the training loop. Identify:
   - The optimizer/EMA configuration.
   - The loss function (Lγ implementation).
   - How prediction-target swapping (ε / v / x₀) is configured.

3. **Subject-level k-fold splitter.** Confirmed location: `src/diffusion/data/splits.py` and `src/segmentation/data/splits.py`. Use the diffusion-side splitter (`stratified_split` / equivalent) for consistency with prior experiments. Verify it produces the same 3-fold partitioning as used in Tables 1–2 of the current manuscript.

4. **Generation pipeline.** Locate `slimdiff-generate`. Identify:
   - Output format (NPZ? per-slice? what naming convention?).
   - How conditioning tokens are iterated over the 60 (zbin × pathology) combinations.
   - The DDIM sampler invocation (`num_inference_steps`, `eta`).

5. **Similarity metrics pipeline.** Path: `src/diffusion/scripts/similarity_metrics/`. Identify:
   - The CLI / config that produces the KID, LPIPS, MMD-MF table.
   - The expected directory layout it consumes (per-replica synthetic slices).

### Decisions to document in `EXPLORATION.md`

- Whether to introduce a new model class `IndependentTwinDDPM` (preferred — clean OOP) or reuse `DiffusionModelUNet` twice as a wrapper.
- Whether the existing training loop can be invoked with `in_channels=1, out_channels=1` directly, or whether a small adapter is needed.
- Where to plug into the existing generation script so that KID/LPIPS/MMD-MF computation works without modification.

If any of the above is unclear after reading, **stop and ask the user**.

## Phase 1 — Implementation

### 1.1 Architecture

Design `IndependentTwinDDPM` as an OOP wrapper that owns two `DiffusionModelUNet` instances, plus the shared conditioning embedding module:

- `self.image_unet`: `in_channels=1, out_channels=1`, channel progression `[C, 2C, 4C, 4C]` with `C` chosen so total params ≈ 26.9M (see §1.2 below).
- `self.mask_unet`: identical configuration, separate weights.
- `self.cond_embed`: a single shared module producing `cemb` from `(zbin, cp)`. Both U-Nets receive the **same** conditioning embedding at training and inference. This is the only shared component.
- Timestep embedding `temb`: each U-Net has its own (per the standard DDPM recipe), as `temb` is properly local to each denoising trajectory.

### 1.2 Parameter matching

The shared variant has 26.9M params at `[64, 128, 256, 256]` channel progression with 2-channel input/output. To match this with two independent single-channel U-Nets, solve for the per-network channel base `C` such that `2 × params(C) ≈ 26.9M`. Empirically this is `C ≈ 46–48` for the same depth/attention/ResBlock configuration; **verify with a parameter count check** before training.

Acceptance criterion: total parameters of `IndependentTwinDDPM` must lie within `26.9M ± 0.3M` (i.e. ±1%). Log the exact count at instantiation.

### 1.3 Training

Train each U-Net independently on the corresponding channel of the existing dataset. The forward diffusion uses **identical noise realisations are not required here** — since the two networks are independent, they can each draw their own noise. Document this choice explicitly in code comments.

Use the **same** training-objective configuration as the best-performing shared-variant cell from Table 1 of the manuscript: `prediction_type=x0`, `Lγ=1.5` for the image, `Lγ=2.0` for the mask. (Different γ per modality is the empirical optimum from §3 of the existing paper; the zero-coupling baseline should use the same per-modality optima for a fair comparison.)

Other hyperparameters: identical to the shared variant — AdamW lr=1e-4, cosine annealing, EMA decay 0.999, gradient clipping at 1.0, early stopping patience 25, batch size as in `configs/examples/training_example.yaml`.

### 1.4 Cross-validation

Use the same 3-fold subject-stratified partitioning as the shared/decoupled experiments. Each fold trains both U-Nets independently, yielding 6 trained networks per fold pair (image + mask × 3 folds).

### 1.5 Generation

For each fold, generate 12 replicas × 3,000 conditioning tokens × per-replica = 108,000 synthetic samples (matching the manuscript's existing protocol, §2.9). At each sampling step, both U-Nets are queried with the same conditioning token; their outputs are concatenated along the channel axis to form a `[I, M]` synthetic pair. Store outputs in the same NPZ layout that the existing `similarity_metrics` pipeline expects.

### 1.6 Evaluation

Run the existing similarity-metrics pipeline unchanged. Produce KID, LPIPS, MMD-MF, and per-feature Wasserstein numbers per fold. The output should slot directly into a new column of the existing comparison table.

## Phase 2 — Statistical comparison

Compare the three configurations (independent / decoupled / shared) using the same non-parametric protocol as the existing manuscript (§2.9):

- **Friedman test** across folds for paired comparison of the three configurations on each metric.
- **Nemenyi post-hoc** if Friedman is significant.
- Report **Cliff's δ** for effect size, alongside p-values.

The expected effect direction is:
$$\text{MMD-MF}_{\text{independent}} \;\gg\; \text{MMD-MF}_{\text{decoupled}} \;>\; \text{MMD-MF}_{\text{shared}}$$
with the gap between independent and decoupled measuring the contribution of joint training, and the gap between decoupled and shared measuring the contribution of representational sharing.

## Deliverables

```
src/diffusion/model/twin_ddpm.py            # IndependentTwinDDPM class
src/diffusion/training/train_twin.py        # Training entrypoint (or extension of existing)
configs/examples/twin_ddpm_example.yaml     # Configuration for the zero-coupling baseline
outputs/zero_coupling/
    fold_0/
        image_unet.pt
        mask_unet.pt
        synthetic/                          # 108k sample NPZ files
    fold_1/...
    fold_2/...
    metrics_summary.json                    # KID, LPIPS, MMD-MF per fold
EXPLORATION.md                              # Phase 0 notes
RESULTS.md                                  # Final table + statistical tests
```

`RESULTS.md` must include:
1. Parameter count of `IndependentTwinDDPM` (verification of matching).
2. Per-fold KID, LPIPS, MMD-MF values for the zero-coupling baseline.
3. Friedman test results across all three configurations.
4. A coupling-continuum table (3 columns × 3 metrics) ready to drop into the manuscript.

## Compute budget

- Training: 2 single-channel U-Nets × 3 folds = 6 training runs. Each at ~half the per-step cost of the 2-channel U-Net but for the same number of steps. Estimated wall-clock: **5–6 days on Picasso** at the same resource allocation as the existing shared-variant training.
- Generation: same cost as existing variants (108k samples × 3 folds, DDIM 300 steps).
- Evaluation: negligible (re-runs of existing pipeline).

## Risk register

- **R1: Parameter count cannot be matched within ±1%.** Mitigation: adjust the channel base `C` to the closest integer; if the match is off by more than 1%, document the exact residual and report it transparently.
- **R2: Zero-coupling baseline performs surprisingly well.** This is a useful negative result, not a failure. It would mean the conditioning signal alone enforces alignment, and the architectural advantage of shared coupling is morphology-specific (consistent with the existing MMD-MF gap). The narrative redirection document handles this case.
- **R3: Each U-Net at half-channel-width fails to converge.** Unlikely given the existing model's stability, but if this happens, increase the channel base (overshooting parameter count by up to 2%) and document.

## Code-quality requirements

Per project conventions: type hints throughout; docstrings on all public methods; `dataclass` for the configuration; custom exceptions (`TwinDDPMConfigError`, `TwinDDPMTrainingError`); structured logging via `logging.getLogger(__name__)`; no `print` calls in library code.

## Out of scope

- Tuning Lγ for the zero-coupling baseline (use the per-modality optima from the existing study for fairness).
- Comparing against other prior-art methods (MedSegFactory, Siamese-Diffusion, etc.) — that is explicitly deferred per the §4.1 limitations of the current manuscript.
- Architectural variants beyond the simple twin design (no cross-attention, no shared encoder, etc.).
