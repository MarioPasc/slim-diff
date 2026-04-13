# TASK 06 — Qualitative Visualization (Camera-Ready Figures)

## Project Context

SLIM-Diff is a shared-bottleneck diffusion model for joint FLAIR MRI + lesion mask synthesis. The ICIP 2026 camera-ready rebuttal commits to:

> *"We commit to an expanded qualitative figure in the camera-ready."* (R1.4)

Reviewer 1 specifically requests: *"lesion–anatomy correspondence, failure modes, or the realistic quality of generated data"* and *"representative of generated image-mask pairs across different conditions."*

Additionally, the new decoupled-bottleneck ablation (TASK-01) generates a comparison set. The camera-ready should include a figure showing shared vs. decoupled quality.

**Your job is to produce publication-quality figures** suitable for an IEEE 6-page conference paper (2-column, ~3.5 in per column or ~7.16 in full width).

---

## Scope

### Files you OWN (create)

```
src/diffusion/scripts/camera_ready/__init__.py                    # NEW subpackage
src/diffusion/scripts/camera_ready/qualitative_figure.py          # NEW — main figure
src/diffusion/scripts/camera_ready/ablation_comparison_figure.py  # NEW — shared vs decoupled
src/diffusion/scripts/camera_ready/failure_modes.py               # NEW — worst-case gallery
src/diffusion/scripts/camera_ready/figure_utils.py                # NEW — shared plotting utilities
```

### Files you READ but do NOT modify

```
src/diffusion/scripts/plot_image_grid.py                          # Existing grid plotter (reference)
src/diffusion/scripts/similarity_metrics/plotting/settings.py     # Existing plot style settings
src/diffusion/scripts/similarity_metrics/metrics/mask_morphology.py  # For LPIPS/morphology scoring
```

### Interface contracts

**From TASK-03 (generation):**
- Generated replicas at:
  ```
  {results_root}/fold_{k}/{architecture}/replicas/replica_{r}.npz
  ```
  Each `.npz` contains `images`, `masks`, `zbins`, `domains` arrays.
  - `images`: `(N, 160, 160)`, float32, `[-1, 1]`
  - `masks`: `(N, 160, 160)`, float32, `[-1, 1]` (continuous)
  - `zbins`: `(N,)`, int
  - `domains`: `(N,)`, int — 0=control, 1=epilepsy

**From TASK-02 (data pipeline):**
- Per-fold test CSVs at `{cache_dir}/fold_{k}/test.csv`.
- Real slice images and masks accessible via paths in the CSV.

---

## Detailed Requirements

### 1. Qualitative Figure — Representative Samples (`qualitative_figure.py`)

This is the primary figure addressing R1.4. It replaces or supplements the existing Figure 1(C).

#### Layout

A grid showing generated image–mask pairs across conditions:

```
              z-low (bin ~5)    z-mid (bin ~15)    z-high (bin ~25)
            ┌────────────────┬────────────────┬────────────────┐
  Control   │  FLAIR | Mask  │  FLAIR | Mask  │  FLAIR | Mask  │
  (c_p=0)   │  FLAIR | Mask  │  FLAIR | Mask  │  FLAIR | Mask  │
            ├────────────────┼────────────────┼────────────────┤
  Epilepsy  │  FLAIR | Mask  │  FLAIR | Mask  │  FLAIR | Mask  │
  (c_p=1)   │  FLAIR | Mask  │  FLAIR | Mask  │  FLAIR | Mask  │
            ├────────────────┼────────────────┼────────────────┤
  Real ref  │  FLAIR | Mask  │  FLAIR | Mask  │  FLAIR | Mask  │
            └────────────────┴────────────────┴────────────────┘
```

- **3 z-positions** (low, mid, high — select bins 5, 15, 25 or nearest with sufficient samples).
- **2 conditions** (control c_p=0, epilepsy c_p=1).
- **2 samples per condition per z-bin** (to show diversity, not cherry-picking).
- **1 real reference row** at the bottom for visual comparison.
- Each cell shows the FLAIR image and the mask side by side (or overlaid).

**Total panels:** 3 z-bins × 3 rows (control, epilepsy, real) × 2 modalities × 2 samples = 36 sub-images. At full column width (7.16 in), this is feasible with ~0.4 in per sub-image.

#### Sample selection

**Do NOT cherry-pick.** Use a systematic selection:

1. From the best fold (fold with lowest KID for the shared model), load all replicas.
2. Filter by target z-bin and condition.
3. Rank by LPIPS to the nearest real test sample (lower = more realistic).
4. Select the **median-ranked** samples — not the best, not the worst. This shows representative quality.

Expose a `--selection-mode` flag: `median` (default), `best`, `random`, `worst`.

#### Mask overlay

For epilepsy samples (c_p=1), overlay the binarised mask (τ=0.0) on the FLAIR image with a semi-transparent red/yellow colormap. Follow the existing convention in the config: `overlay.alpha=0.5`, `overlay.color=[255, 0, 0]`.

For control samples (c_p=0), the mask should be blank (no lesion). If the model incorrectly generates lesion-like structures in control masks, show them — this is informative.

#### Visual style

- **IEEE-compliant**: use `matplotlib` with `plt.rcParams` matching the existing `settings.py` in `similarity_metrics/plotting/`.
- Grayscale for FLAIR images (cmap `gray`).
- No axes, no ticks. Minimal whitespace.
- Row labels on the left: "Generated (control)", "Generated (epilepsy)", "Real".
- Column headers: z-bin labels.
- Font size: 7–8pt to fit in a 2-column figure.
- Output formats: PDF (vector) and PNG (300 DPI) for backup.

#### CLI

```bash
python -m src.diffusion.scripts.camera_ready.qualitative_figure \
    --results-root /path/to/results \
    --cache-dir /path/to/slice_cache \
    --fold 0 \
    --architecture shared \
    --output /path/to/figures/qualitative_grid.pdf \
    --zbins 5 15 25 \
    --n-samples 2 \
    --selection-mode median \
    --format pdf \
    --dpi 300
```

---

### 2. Ablation Comparison Figure (`ablation_comparison_figure.py`)

Side-by-side comparison of shared vs. decoupled quality. This visually supports the quantitative ablation in the new table.

#### Layout

```
              Shared bottleneck              Decoupled bottleneck
            ┌──────────────────────────┬──────────────────────────┐
  Epilepsy  │  FLAIR | Overlay         │  FLAIR | Overlay         │
  z=5       │  FLAIR | Overlay         │  FLAIR | Overlay         │
            ├──────────────────────────┼──────────────────────────┤
  Epilepsy  │  FLAIR | Overlay         │  FLAIR | Overlay         │
  z=15      │  FLAIR | Overlay         │  FLAIR | Overlay         │
            ├──────────────────────────┼──────────────────────────┤
  Epilepsy  │  FLAIR | Overlay         │  FLAIR | Overlay         │
  z=25      │  FLAIR | Overlay         │  FLAIR | Overlay         │
            └──────────────────────────┴──────────────────────────┘
```

- Focus on **epilepsy** samples only (c_p=1) — this is where the lesion-mask fidelity matters.
- Same fold, same z-bins, same generation seeds if possible. If TASK-03 uses deterministic seeds per (fold, architecture), then the same seed/z-bin/condition produces a paired comparison. **Check the generation seed scheme.** If seeds differ, select median-ranked samples from each architecture independently.
- 3 z-bins × 2 architectures × 2 samples = 12 cells.

#### Sample pairing strategy

If the generation seeds are matched across architectures (i.e., the same `x_T` noise is used for both shared and decoupled within the same fold/replica/condition), show paired outputs — same noise input, different architecture. This is the most informative comparison because it isolates the architectural effect.

If seeds are NOT matched, select the median-LPIPS sample from each architecture independently and note this in the figure caption.

#### CLI

```bash
python -m src.diffusion.scripts.camera_ready.ablation_comparison_figure \
    --results-root /path/to/results \
    --cache-dir /path/to/slice_cache \
    --fold 0 \
    --output /path/to/figures/ablation_comparison.pdf \
    --zbins 5 15 25 \
    --n-samples 2 \
    --format pdf
```

---

### 3. Failure Modes Gallery (`failure_modes.py`)

R1.4 explicitly asks for failure modes. Show the worst samples from both architectures.

#### Layout

A small 2×4 or 2×6 grid:

```
              Worst-1    Worst-2    Worst-3    Worst-4
  Shared     │ overlay  │ overlay  │ overlay  │ overlay │
  Decoupled  │ overlay  │ overlay  │ overlay  │ overlay │
```

#### Sample selection

1. From a given fold, load all epilepsy (c_p=1) generated samples.
2. Compute LPIPS to the nearest real test sample for each synthetic sample.
3. Select the top-K worst (highest LPIPS) samples.
4. Show the FLAIR with mask overlay.
5. Annotate each with its LPIPS score.

This is honest and addresses the reviewer's request. If the shared model's worst cases are better than the decoupled model's worst cases, the figure speaks for itself.

#### CLI

```bash
python -m src.diffusion.scripts.camera_ready.failure_modes \
    --results-root /path/to/results \
    --cache-dir /path/to/slice_cache \
    --fold 0 \
    --output /path/to/figures/failure_modes.pdf \
    --n-worst 4 \
    --format pdf
```

---

### 4. Shared Utilities (`figure_utils.py`)

Common plotting functions used by all three figure scripts:

```python
def rescale_to_display(image: NDArray) -> NDArray:
    """Map [-1, 1] to [0, 1] for matplotlib display."""

def binarise_mask(mask: NDArray, tau: float = 0.0) -> NDArray:
    """Binarise continuous mask at threshold τ."""

def overlay_mask_on_image(
    image: NDArray,
    mask: NDArray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
) -> NDArray:
    """Create RGB overlay of binary mask on grayscale image."""

def load_real_samples(
    cache_dir: Path,
    fold: int,
    zbins: list[int],
    condition: int,
    n_samples: int = 2,
) -> tuple[NDArray, NDArray, NDArray]:
    """Load real (image, mask, zbin) tuples from test set."""

def load_synthetic_samples(
    results_root: Path,
    fold: int,
    architecture: str,
    zbins: list[int],
    condition: int,
    n_samples: int = 2,
    selection_mode: str = "median",
    reference_images: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Load and select synthetic samples by LPIPS rank."""

def setup_ieee_style() -> None:
    """Configure matplotlib for IEEE 2-column format."""

def add_row_label(ax: Axes, text: str, fontsize: int = 7) -> None:
    """Add a rotated row label to the left of a subplot row."""
```

The `load_synthetic_samples` function with `selection_mode="median"` needs LPIPS computation. To avoid a GPU dependency for figure generation:
- **Option A**: Pre-compute LPIPS for all samples during TASK-04 and store a ranking index.
- **Option B**: Compute LPIPS on-the-fly using CPU (slow but works).
- **Option C**: Use a simpler proxy metric (MSE to nearest real) for ranking.

Implement Option C as default with a `--use-lpips` flag that falls back to GPU-based LPIPS if available. The MSE proxy is sufficient for selecting "median representative" samples.

---

## Design Principles

1. **No cherry-picking.** Median selection ensures representative samples. Document the selection method in the figure caption.
2. **Show both architectures.** The ablation comparison directly supports the shared-bottleneck hypothesis.
3. **Show failures honestly.** If the shared model has bad failure modes, show them. Reviewers respect honesty.
4. **IEEE formatting.** All figures must fit within a 2-column (3.5 in wide) or full-width (7.16 in wide) IEEE layout. Use vector PDF output.
5. **Reproducibility.** Fix random seeds for any stochastic sample selection. Log the selected sample indices to a JSON sidecar file.

---

## Acceptance Criteria (Testable on 8 GB VRAM)

### AC-1: Qualitative figure on dummy data
Generate 100 random images and masks in `[-1, 1]` at 160×160, split across 3 z-bins and 2 conditions. Create dummy real test data.
Run `qualitative_figure.py` with `--selection-mode random`.
Assert: output PDF/PNG exists and is non-empty.
Assert: figure has the expected grid layout (check via `matplotlib.figure.get_axes()` count).

### AC-2: Ablation comparison on dummy data
Generate dummy data for both `shared` and `decoupled` directories.
Run `ablation_comparison_figure.py`.
Assert: output exists.
Assert: figure has 2 columns (shared, decoupled) and correct number of rows.

### AC-3: Failure modes on dummy data
Generate 100 random epilepsy samples.
Create 20 real test samples.
Run `failure_modes.py --n-worst 4`.
Assert: output exists.
Assert: 4 worst samples are shown per architecture row.
Assert: LPIPS/MSE scores are annotated on each panel (check via `ax.texts`).

### AC-4: Overlay correctness
Create a synthetic image (all zeros) and a known binary mask (a circle in the center).
Call `overlay_mask_on_image`.
Assert: the output is RGB (3 channels).
Assert: pixels inside the circle have the overlay color blended.
Assert: pixels outside the circle remain grayscale.

### AC-5: IEEE figure dimensions
For each figure script, check that `fig.get_size_inches()` returns dimensions compatible with IEEE 2-column format:
- Full width: width ≤ 7.16 in.
- Single column: width ≤ 3.5 in.
- Height: reasonable (no more than 9 in for a single figure).

### AC-6: Selection reproducibility
Run `qualitative_figure.py` twice with the same seed.
Assert: the sidecar JSON files listing selected sample indices are identical.
Run again with a different seed.
Assert: the indices differ.

### AC-7: No modification of existing files
`git diff --stat` shows only new files under `src/diffusion/scripts/camera_ready/`. No existing files modified.

---

## Anti-Patterns

- **Do NOT use `plt.show()`.** All figures are saved to files. Scripts must run headless (no display server) on Picasso.
- **Do NOT use `plt.tight_layout()` alone.** Use `fig.savefig(..., bbox_inches='tight')` combined with explicit `subplots_adjust` for fine control.
- **Do NOT select the best-looking samples.** Median selection is the default. If you implement a `best` mode, document it clearly and do not use it for the camera-ready.
- **Do NOT overlay masks on control (c_p=0) samples** unless the model erroneously generates lesion pixels — in that case, show the overlay to expose the error.
- **Do NOT use color for FLAIR images.** Grayscale only. Color is reserved for the mask overlay.
- **Do NOT produce figures wider than 7.16 inches or with font size below 6pt.** IEEE will reject them.
- **Do NOT import from TASK-04 or TASK-05 modules.** These tasks are independent. Load data directly from `.npz` and CSV files.
