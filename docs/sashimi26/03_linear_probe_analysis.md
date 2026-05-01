# Study 2.1 — Linear-Probe Analysis of Bottleneck Representations

## Objective

Quantify what the **shared** vs **decoupled** bottleneck representations of the trained diffusion U-Nets actually encode, by training **linear probes** to predict five interpretable factors from globally pooled bottleneck activations. The hypothesis is that the shared bottleneck encodes joint factors of image intensity *and* mask geometry in an entangled way, while the decoupled bottleneck separates these factors across its two paths. If confirmed, this provides a **mechanistic explanation** for the 2.6× MMD-MF gap reported in the existing manuscript and converts ICIP reviewer R2.1's "limited novelty" objection into "novel mechanistic finding".

## Scientific rationale

Linear probing is the standard tool for interrogating learned representations — it measures the degree to which a target attribute is **linearly decodable** from features, which directly indicates whether the representation has learned to encode that attribute (Alain & Bengio, *Understanding intermediate layers using linear classifier probes*, ICLR Workshop 2017; Belinkov, *Probing Classifiers: Promises, Shortcomings, and Advances*, Computational Linguistics 2022).

For a representation `h ∈ ℝᵈ` and a target `y`, the linear-probe performance is

$$\text{R}^2(h, y) \;=\; 1 - \frac{\min_{w, b}\, \mathbb{E}\!\left[(y - w^\top h - b)^2\right]}{\mathbb{V}[y]} \quad \text{for regression,}$$

or classification accuracy for discrete targets. High R² (or accuracy) implies the target is encoded in the representation; low R² implies it is not (or is encoded non-linearly, which we do not credit).

Our specific hypothesis structure:

- **Shared bottleneck** (h_shared ∈ ℝ²⁵⁶): a single representation that must encode both image and mask factors to support the 2-channel reconstruction. Linear-probe R² should be **high for both image-related and mask-related targets**.
- **Decoupled bottleneck** (h_decoupled = [h_decoupled^I, h_decoupled^M] ∈ ℝ²⁵⁶ where each path produces 128 dims): two separate representations. Linear-probe R² on the **image path** should be high for image targets and lower for mask-shape targets, and vice versa for the mask path. Critically, neither sub-path of the decoupled bottleneck should match the shared bottleneck on the **mask-shape** targets — and that is exactly the gap that explains the MMD-MF advantage.

If the hypothesis is borne out, the qualitative finding is: *forced sharing of the bottleneck causes the network to discover joint latent factors that simultaneously explain image intensity and mask geometry, whereas decoupling allows each path to specialise narrowly, losing the cross-modal alignment that lesion morphology requires.*

## Phase 0 — Codebase exploration (mandatory)

Read, do not skim. Produce `EXPLORATION.md` before any coding.

### Required exploration

1. **Locate trained checkpoints.** The shared and decoupled variants are already trained per the existing manuscript (Table 2). Find:
   - The exact `.pt` file paths (likely `outputs/<experiment>/fold_<k>/ema_best.pt` or similar).
   - The corresponding training config so the model can be reinstantiated with matching hyperparameters before loading weights.
   - How many checkpoints exist (3 folds × 2 architectures = 6 expected).

2. **Identify the bottleneck location in the model.** The U-Net has channel progression `[64, 128, 256, 256]`; the bottleneck operates at `256 × 20 × 20`. In the implementation, this is the deepest ResBlock before the decoder begins. You need to:
   - Find the exact `nn.Module` instance corresponding to the bottleneck output.
   - For the **shared variant**: this is a single block — register a forward hook on its output.
   - For the **decoupled variant**: there are two parallel paths — register hooks on both, and concatenate or analyse separately as appropriate.
   - Document the exact module names you will hook (e.g. `model.middle_block`, or whatever MONAI uses — `DiffusionModelUNet` typically exposes `down_blocks`, `mid_block`, `up_blocks`).

3. **Real-data feeding pipeline.** We probe on **real** test slices, not synthetic. Use the existing `SegmentationSliceDataset` or the diffusion-side dataset to load the real test set. Confirm the slice list matches the test partition used in the existing experiments.

4. **Conditioning and timestep choice.** The bottleneck activations depend on `t` (timestep) and `c` (conditioning). For probing, we want a representation that is informative about the **clean data**, so feed `t = 0` (or the smallest non-zero timestep available, since some implementations forbid t=0) and the correct conditioning token for each slice. Document the exact choice.

### Decisions to document in `EXPLORATION.md`

- The exact module path of the bottleneck for both architectures.
- The pooling strategy (recommended: spatial global average pooling, producing a 256-d vector for shared, two 128-d vectors for decoupled).
- The timestep used for probing.
- The number of slices that will go into each probe-train and probe-test set (rule of thumb: at least 5× the feature dimensionality, so ≥1280 slices per probe).

## Phase 1 — Implementation

### 1.1 Feature extraction

Build `BottleneckExtractor`, an inference-only utility:

```python
@dataclass(frozen=True)
class BottleneckExtractorConfig:
    checkpoint_path: Path
    architecture: Literal["shared", "decoupled"]
    bottleneck_module_path: str    # dotted path, e.g. "model.mid_block.resnets.1"
    probe_timestep: int = 1         # smallest valid t
    pooling: Literal["mean", "max"] = "mean"
    device: str = "cuda"

class BottleneckExtractor:
    def __init__(self, config: BottleneckExtractorConfig) -> None: ...
    def extract(self, dataloader: DataLoader) -> Tuple[np.ndarray, dict]:
        """Returns (features, metadata) where metadata holds per-slice targets."""
```

Implementation notes:
- Use a single forward hook to capture activations.
- Apply spatial global average pooling: `h = activations.mean(dim=(2, 3))`, yielding `(B, C)`.
- For the decoupled variant, capture both paths and return them as separate arrays; downstream probes are run on each separately and on their concatenation.

### 1.2 Probe targets

Five targets, chosen to span both image-side and mask-side factors:

| Target | Type | Rationale |
|---|---|---|
| `cp` (pathology class) | Binary classification | Conditioning input — sanity check; both architectures must encode this |
| `zbin` (axial position) | Regression (or 30-way classification) | Conditioning input — sanity check |
| `lesion_area` (continuous, normalized by slice area) | Regression | Mask-shape factor; decoupled should have lower R² in image-path |
| `lesion_eccentricity` | Regression | Mask-shape factor; differentiates round vs elongated lesions |
| `mean_intensity_lesion_region` | Regression | Image-side factor; differentiates FLAIR hyperintensity strength |

Compute targets directly from the real (image, mask) pairs. For slices without lesions, `lesion_area = 0`, `lesion_eccentricity` is undefined (exclude these slices for the eccentricity probe only — the others handle zero correctly).

### 1.3 Probe model and protocol

For each (architecture, target) pair, fit a linear probe with cross-validation:

- **Regression targets**: Ridge regression. Use 5-fold CV to choose the L2 strength `α ∈ {10⁻³, 10⁻², 10⁻¹, 10⁰, 10¹, 10²}` on a held-out probe-validation split. Report **R²** on a probe-test split.
- **Classification targets**: Logistic regression with L2. Same CV protocol. Report balanced accuracy.

**Critical**: the probe-train / probe-test split must be **subject-stratified** — slices from the same subject must not appear in both sets, or the probe leaks subject identity. Reuse `SubjectKFoldSplitter` for this.

For the decoupled variant, run **three** probe variants per target:
1. Image-path features only (128 dims).
2. Mask-path features only (128 dims).
3. Concatenated features (256 dims, equal dim to shared).

The concatenated variant is the **fair comparison** to the shared variant (same dimensionality). The path-specific variants tell us which path encodes what.

### 1.4 Statistical comparison

For each target, compare shared vs decoupled-concatenated probe R² (or accuracy) across folds:

- **Wilcoxon signed-rank test** on per-fold R² values (paired).
- **Cliff's δ** for effect size.

The hypothesis-confirming pattern is:

| Target | Expected R²(shared) vs R²(decoupled-cat) |
|---|---|
| `cp` | Approximately equal (sanity check) |
| `zbin` | Approximately equal (sanity check) |
| `lesion_area` | shared ≈ decoupled-cat or shared slightly higher |
| `lesion_eccentricity` | **shared > decoupled-cat** (key prediction) |
| `mean_intensity_lesion_region` | **shared > decoupled-cat** (key prediction) |

The two key predictions test joint coding. If both are confirmed, the mechanistic story holds.

## Phase 2 — Visualisation

For the paper, produce:

1. **Probe-R² bar chart** with shared, decoupled-image-path, decoupled-mask-path, decoupled-concatenated as four bars per target. Error bars from CV folds.
2. **t-SNE / UMAP projection** of the bottleneck features coloured by `lesion_area` (continuous) and `cp` (discrete) for both architectures. Visual differentiation of the embedding structure is itself informative.

## Deliverables

```
src/analysis/linear_probe/
    __init__.py
    extractor.py          # BottleneckExtractor with forward hooks
    probes.py             # Ridge / Logistic linear probes with CV
    targets.py            # Compute lesion_area, eccentricity, mean_intensity from masks
    statistics.py         # Wilcoxon + Cliff's δ
    visualize.py          # Bar charts and t-SNE projections
    cli.py                # entry point: linear-probe-analysis
outputs/linear_probe/
    shared/
        fold_0_features.npy
        fold_0_targets.npz
        ...
    decoupled/...
    probe_results.json
    figures/
        probe_r2_bars.pdf
        tsne_lesion_area.pdf
EXPLORATION.md
RESULTS.md
```

`RESULTS.md` must contain:
1. The full probe-R² table (5 targets × 4 conditions × per-fold values + mean ± std).
2. Wilcoxon signed-rank p-values and Cliff's δ for the key comparisons (eccentricity, mean_intensity).
3. A **3–4 sentence interpretation** of whether the hypothesis was confirmed, ready to slot into §3.3 of the SASHIMI paper.

## Compute budget

- No diffusion training. Only forward passes and tiny linear regressions.
- Forward passes: ~3–4k real slices × 6 checkpoints × single forward at one timestep ≈ **~30 minutes total on a single GPU**.
- Probes: each probe is `O(d² n)` for ridge regression, with `n ≈ 3000`, `d ≤ 256`. **Seconds per probe.**
- Total wall-clock: **~1–2 days** including exploration, debugging, and visualisation. The cheapest of the three studies.

## Risk register

- **R1: Hooks fail because the module path is wrong.** Mitigation: print the model with `torchinfo.summary(model)` or iterate `named_modules()` first to identify the exact bottleneck path. **Do this before writing the extractor.**
- **R2: Probes show no difference between shared and decoupled.** This means the architectural intervention does not reach the bottleneck representation in a linearly-decodable way. The narrative redirect would then drop §3.3 from the paper and use the saved space for a deeper qualitative analysis. Run the probes early (week 1) so this is known in time.
- **R3: Probe-test slices leak subject identity.** Catastrophic — invalidates results. Mitigation: enforce subject-stratified splits and verify by checking that subject-ID overlap is zero between probe-train and probe-test.
- **R4: Bottleneck features are too high-dimensional for the probe sample size.** Possible if probing per-fold (each fold has ~600 slices but feature dim is 256). Mitigation: aggregate slices across folds when possible, or use PCA pre-reduction to ~50 dims (and document this preprocessing). Use the rule `n_train ≥ 5 × d`.
- **R5: Timestep choice biases the result.** Mitigation: run a small ablation at three timesteps (t=1, t=100, t=500) and report whether conclusions are stable across timesteps. If they are not, this is itself an interesting finding for the paper.

## Out of scope

- Non-linear probes (MLPs, kernel methods). The whole point is to measure linear decodability — non-linear probes confound the diagnostic.
- Causal interventions (ablating individual feature dimensions and remeasuring). Out of scope for the SASHIMI submission.
- Probing intermediate encoder/decoder layers (only the bottleneck is the architectural intervention; probing other layers would be informative but is out of scope here).

## Code-quality requirements

Per project conventions: type hints, docstrings, `dataclass` configs, custom exceptions (`BottleneckExtractionError`, `ProbeFitError`), structured logging. Use `sklearn.linear_model.Ridge` and `sklearn.linear_model.LogisticRegression` rather than rolling your own.
