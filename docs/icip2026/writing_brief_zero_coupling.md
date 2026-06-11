# Writing brief: zero-coupling architecture and MMD-PL diagnostic

This brief packages the three new contributions to be folded into the ICIP 2026
camera-ready manuscript. It is fact-only: numbers, definitions, and references
to source CSVs. No architecture is favoured in the prose; the writing agent is
free to draw the comparative story.

In every table below, **bold** marks the lowest (best) value per column and
<u>underline</u> marks the highest (worst). For metrics where the target is to
match a real baseline (e.g. `frac_bright`), best is the value closest to the
baseline and worst is the value furthest from it.

All raw artefacts referenced live under

```
/media/mpascual/Sandisk2TB/completed/jsddpm/results/epilepsy/icip2026/camera_ready/evaluations/
├── eval_output/      # canonical CSVs (fold_metrics, summary_metrics, kid_per_zbin, ...)
├── posthoc_output/   # tau, NN, MMD-PL, tables/, json sidecars
└── figures/          # ablation_comparison.pdf + sidecar
```

---

## 1. Zero-coupling architecture

### Definition

The zero-coupling baseline (`IndependentTwinDDPM`) removes every cross-modal
parameter from the joint diffusion model. It instantiates two independent
DDPMs — one over FLAIR slices, one over lesion masks — that share no encoder,
no bottleneck, no decoder, and no conditioning pathway. Each network keeps the
same U-Net depth and channel multipliers as the shared variant, so per-network
capacity is unchanged, but the *total* parameter count is approximately twice
that of the param-matched shared/decoupled pair.

Training and sampling settings shared across all three architectures:

| Setting | Value |
|---|---|
| Prediction target | x₀ |
| Loss | Lₚ with γ = 1.5 |
| Self-conditioning probability | 0.5 |
| Noise schedule | cosine, T = 1000 |
| Sampler | DDIM, 300 steps, η = 0.2 |
| Conditioning | (z-bin, pathology-class) token via AdaGN |
| Image spatial size | 160 × 160 |

The three architectures span the cross-modal coupling axis:

| Variant | Cross-modal coupling | Parameter budget |
|---|---|---|
| Shared | Full bottleneck shared between branches | ≈ 26.9 M |
| Decoupled | Shared early/late stages, branch-split bottleneck | ≈ 26.9 M (param-matched, +0.5 %) |
| Zero-coupling | None — two independent U-Nets | ≈ 2 × 26.9 M |

### Per-fold replica counts

Zero-coupling jobs timed out on Picasso before they could complete the same
number of generation replicas as the shared/decoupled cells. The realised
counts are:

| Fold | Shared replicas | Decoupled replicas | Zero-coupling replicas | Zero-coupling synth slices |
|---|---|---|---|---|
| 0 | 12 | 12 | 2 | 18 000 |
| 1 | 12 | 12 | 3 | 27 000 |
| 2 | 12 | 12 | 3 | 27 000 |

(Shared and decoupled cells contribute 108 000 synth slices each per fold.)

Source: `eval_output/eval_sample_counts.json`.

### Code reference

`configs/camera_ready/base_zero_coupling.yaml`,
`slurm/camera_ready/zero_coupling_fold_{0,1,2}/`,
`src/diffusion/model/factory.py` (IndependentTwinDDPM branch).

---

## 2. Three-axis evaluation protocol

The evaluation now covers three orthogonal axes:

| Axis | Metric(s) | Features used | What it answers |
|---|---|---|---|
| Image marginal | KID, LPIPS | image only | Does the FLAIR look real? |
| Mask marginal | MMD-MF | mask only (10 morphological scalars) | Do the masks have realistic shape? |
| Image–mask joint | **MMD-PL** (new) | image + mask jointly (5 scalars) | Do the masks sit on tissue that looks like a lesion? |

The three feature sets are pairwise disjoint, so the three axes can independently
register success or failure.

---

## 3. MMD-PL — paired-lesion image–mask coupling

### Why a new metric

KID and LPIPS evaluate the FLAIR slice in isolation; MMD-MF evaluates the mask
polygon in isolation. Neither metric is sensitive to the failure mode in which
a model generates a realistic FLAIR slice and, independently, a morphologically
plausible mask, but places the mask on tissue that does not actually look like
a lesion. MMD-PL measures that correspondence directly.

### Per-lesion feature vector

For every connected mask component with ≥ 5 pixels, MMD-PL computes a 5-D
feature vector. Each feature requires both the FLAIR slice and the mask;
none duplicates an MMD-MF feature.

Let `M ⊂ Ω` be the mask, `R` a 5-pixel-dilated perilesion ring (excluding
`M`), and `B` the brain mask (FLAIR > −0.95, eroded by 1 px). Let `μ_X` and
`σ_X` be the mean and standard deviation of FLAIR over set `X`.

| dim | feature | formula | failure mode it surfaces |
|---|---|---|---|
| 1 | `Δ` | `μ_M − μ_R` | mask interior is not brighter than its neighbourhood |
| 2 | `z_in` | `(μ_M − μ_B) / σ_B` | mask interior is not hyperintense relative to brain tissue |
| 3 | `σ_norm` | `σ_M / σ_B` | mask interior lacks lesion-like intensity variance |
| 4 | `edge_align` | `mean(|∇FLAIR|, ∂M) / mean(|∇FLAIR|, B)` | image has no gradient where the mask draws an edge |
| 5 | `frac_bright` | `|{x ∈ M : FLAIR(x) > μ_B + σ_B}| / |M|` | most "lesion" pixels are not actually bright |

Components with non-finite features (e.g. mask at brain boundary with empty
perilesion ring) are dropped. Features are z-scored using the *real* population
mean and standard deviation per fold so that all five dimensions contribute on
comparable scales to the kernel evaluation.

### Estimator

MMD-PL uses the same polynomial kernel `k(x, y) = (x · y / D + 1)³` as KID and
MMD-MF, applied to the standardised 5-D vectors. The subset-MMD estimator
draws 100 random subsets of size 500 (capped to `min(N_real, N_synth)`) and
reports the mean and standard deviation of the unbiased squared-MMD across
subsets. Seed = 42. **Lower is better.**

Per-feature Wasserstein-1 distances `W₁(D_real^j, D_synth^j)` are reported for
each feature `j`. These are lower-variance summaries that show which axis a
model breaks on, independent of the kernel choice.

### Orthogonality to MMD-MF

The MMD-MF feature set is

```
{area, perimeter, circularity, solidity, extent, eccentricity,
 major_axis_length, minor_axis_length, equivalent_diameter, geometric_mean}
```

— every entry computable from the mask alone. The MMD-PL feature set is the
five image-given-mask scalars above — every entry collapses to a constant
without access to the FLAIR slice. The two feature sets have empty intersection,
and each metric is invariant to perturbations the other is sensitive to:

- Perturbing only the mask polygon (e.g. eroding/dilating it while leaving its
  centre fixed on the same image patch) changes MMD-MF but leaves the
  image-given-mask statistics essentially unchanged.
- Perturbing only the underlying image (e.g. swapping the FLAIR slice while
  keeping the same mask) leaves MMD-MF unchanged but moves MMD-PL.

Source: `src/diffusion/scripts/similarity_metrics/posthoc/paired_lesion_mmd.py`.

---

## 4. Memorisation control (NN diagnostic)

To rule out training-set memorisation as an alternative explanation for the
marginal-metric values, we report nearest-neighbour distances in InceptionV3
pool3 feature space (L2):

- `D_test` = NN distance from each held-out test slice to its nearest train
  slice (baseline);
- `D_synth` = NN distance from each synthetic slice to its nearest train slice.

Indicators per (fold, architecture):

- `R = median(D_synth) / median(D_test)` — values close to 1 are consistent
  with generalisation; values ≪ 1 indicate that synthetic samples sit
  anomalously close to the training slices.
- `Suspect %` = fraction of synthetic samples with
  `D_synth < min(D_test)` — a Carlini-style "closer to *some* train slice than
  the closest held-out test slice is" count.
- `W₁(D_synth, D_test)` — whole-distribution Wasserstein-1 gap.

Source: `src/diffusion/scripts/similarity_metrics/posthoc/memorization_nn.py`.

---

## 5. Results

All numbers are mean ± standard deviation across the three stratified folds.

### 5.1 Three-axis comparison (replaces / extends Table 1)

| Architecture | KID ↓ | LPIPS ↓ | MMD-MF ↓ | MMD-PL ↓ |
|---|---|---|---|---|
| Decoupled | 0.018 ± 0.003 | **0.304 ± 0.001** | <u>13.08 ± 10.13</u> | **13.77 ± 5.38** |
| Zero-coupling | **0.017 ± 0.001** | 0.305 ± 0.000 | **0.42 ± 0.11** | <u>112.05 ± 47.37</u> |
| Shared | <u>0.021 ± 0.004</u> | <u>0.305 ± 0.001</u> | 5.04 ± 4.70 | 62.19 ± 72.39 |

LPIPS raw values: decoupled 0.30380, zero-coupling 0.30453, shared 0.30470. All
three round to ≤ 0.001 of each other; the ranking is given for completeness.

LaTeX:

```tex
\begin{table}[t]
\centering
\caption{Three-axis comparison across the camera-ready cells.
Mean $\pm$ std across 3 stratified folds. Lower is better in every column;
\textbf{best} in bold, \underline{worst} underlined.}
\label{tab:three_axis}
\begin{tabular}{lcccc}
\toprule
Architecture & KID $\downarrow$ & LPIPS $\downarrow$ & MMD-MF $\downarrow$ & MMD-PL $\downarrow$ \\
\midrule
Decoupled     & $0.018 \pm 0.003$ & $\mathbf{0.304 \pm 0.001}$ & $\underline{13.08 \pm 10.13}$ & $\mathbf{13.77 \pm 5.38}$ \\
Zero-coupling & $\mathbf{0.017 \pm 0.001}$ & $0.305 \pm 0.000$ & $\mathbf{0.42 \pm 0.11}$ & $\underline{112.05 \pm 47.37}$ \\
Shared        & $\underline{0.021 \pm 0.004}$ & $\underline{0.305 \pm 0.001}$ & $5.04 \pm 4.70$ & $62.19 \pm 72.39$ \\
\bottomrule
\end{tabular}
\end{table}
```

Sources: `eval_output/summary_metrics.csv`,
`posthoc_output/paired_lesion_mmd_summary.csv`.

### 5.2 Per-fold breakdown

| Fold | Arch | KID | LPIPS | MMD-MF | MMD-PL |
|---|---|---|---|---|---|
| 0 | Shared | 0.0188 ± 0.0007 | 0.3038 ± 0.0648 | 11.68 ± 4.51 | 2.20 ± 1.53 |
| 0 | Decoupled | 0.0150 ± 0.0006 | 0.3029 ± 0.0644 | 25.99 ± 10.55 | 8.52 ± 14.52 |
| 0 | Zero-coupling | 0.0163 ± 0.0007 | 0.3040 ± 0.0669 | 0.48 ± 0.30 | 83.41 ± 29.74 |
| 1 | Shared | 0.0184 ± 0.0008 | 0.3048 ± 0.0643 | 1.96 ± 0.58 | 164.02 ± 92.77 |
| 1 | Decoupled | 0.0171 ± 0.0007 | 0.3040 ± 0.0654 | 1.24 ± 0.17 | 21.17 ± 16.09 |
| 1 | Zero-coupling | 0.0174 ± 0.0007 | 0.3048 ± 0.0673 | 0.27 ± 0.09 | 178.81 ± 60.60 |
| 2 | Shared | 0.0261 ± 0.0011 | 0.3056 ± 0.0631 | 1.48 ± 0.47 | 20.34 ± 14.42 |
| 2 | Decoupled | 0.0224 ± 0.0008 | 0.3044 ± 0.0634 | 12.00 ± 3.82 | 11.63 ± 13.99 |
| 2 | Zero-coupling | 0.0176 ± 0.0007 | 0.3048 ± 0.0678 | 0.51 ± 0.33 | 73.92 ± 30.64 |

Sources: `eval_output/fold_metrics.csv`, `posthoc_output/paired_lesion_mmd.csv`.

### 5.3 MMD-PL per-feature Wasserstein-1 to real

Lower is better; **best** bold, <u>worst</u> underlined.

| Architecture | W₁[Δ] | W₁[z_in] | W₁[σ_norm] | W₁[edge_align] | W₁[frac_bright] |
|---|---|---|---|---|---|
| Decoupled | 0.046 ± 0.021 | 0.122 ± 0.021 | **0.041 ± 0.020** | <u>0.272 ± 0.042</u> | **0.073 ± 0.043** |
| Zero-coupling | <u>0.169 ± 0.009</u> | <u>0.506 ± 0.040</u> | <u>0.201 ± 0.027</u> | **0.069 ± 0.023** | <u>0.118 ± 0.009</u> |
| Shared | **0.040 ± 0.021** | **0.140 ± 0.073** | 0.051 ± 0.025 | 0.227 ± 0.038 | 0.079 ± 0.040 |

Source: `posthoc_output/paired_lesion_mmd_summary.csv`.

### 5.4 Hyperintensity falsification (`frac_bright`)

Fraction of synthetic "lesion" pixels above the brain hyperintensity threshold
`μ_B + σ_B`. Real baseline: **0.235**. Best = closest to baseline.

| Architecture | Mean `frac_bright` | Gap vs real |
|---|---|---|
| Decoupled | 0.178 ± 0.062 | 0.057 |
| Zero-coupling | <u>0.118 ± 0.009</u> | <u>0.117</u> |
| Shared | **0.156 ± 0.040** | **0.079** |

Note: no architecture matches the real baseline; decoupled is closest by mean
but the inter-fold spread is large. Bold/underline applied on the gap column
(closest / furthest from real).

### 5.5 Per-feature distribution at fold 0

Mean of each MMD-PL feature in the real and synthetic populations, fold 0.
This is the most direct evidence of where zero-coupling diverges from real.

| Feature | Real | Decoupled | Zero-coupling | Shared |
|---|---|---|---|---|
| `Δ` | 0.216 | 0.245 | 0.055 | 0.285 |
| `z_in` | 0.652 | 0.532 | 0.162 | 0.611 |
| `σ_norm` | 0.434 | 0.454 | 0.640 | 0.520 |
| `edge_align` | 1.024 | 1.248 | 0.984 | 1.305 |
| `frac_bright` | 0.235 | 0.106 | 0.116 | 0.196 |

Source: `posthoc_output/paired_lesion_features.csv`.

### 5.6 Tau-sensitivity (R1.5)

MMD-MF as a function of the mask binarisation threshold τ ∈ {−0.30, …, +0.30}.
The complete sweep is in `posthoc_output/tau_sensitivity_summary.csv` and the
three rendered tables are at:

- `posthoc_output/tables/table_tau_sensitivity_shared.tex`
- `posthoc_output/tables/table_tau_sensitivity_decoupled.tex`
- `posthoc_output/tables/table_tau_sensitivity_zerocoupled.tex`

Range of MMD-MF across the 9-point τ sweep, per architecture:

| Architecture | MMD-MF range across τ | Range / median(τ=0) |
|---|---|---|
| Decoupled | 12.25 – 13.50 | 0.10 |
| Zero-coupling | 0.44 – 0.46 | 0.05 |
| Shared | 5.14 – 5.37 | 0.05 |

All three architectures are roughly insensitive to τ in absolute terms, with
zero-coupling and shared showing the flattest profiles.

### 5.7 NN memorisation diagnostic

`R` near 1 = generalisation; `R ≪ 1` = memorisation. `Suspect %` ≪ 1 % on all
cells. **Best** bold (closest to 1 for `R`, smallest for the others);
<u>worst</u> underlined.

| Architecture | R = med(D_synth) / med(D_test) | Suspect % | W₁(D_synth, D_test) |
|---|---|---|---|
| Decoupled | **1.045 ± 0.014** | **0.019 %** | 0.612 ± 0.166 |
| Zero-coupling | <u>0.961 ± 0.006</u> | <u>0.036 %</u> | **0.512 ± 0.105** |
| Shared | 1.065 ± 0.025 | 0.023 % | <u>0.809 ± 0.261</u> |

No architecture's `R` or suspect fraction crosses the memorisation regime
(`R ≪ 1` together with a non-trivial suspect fraction).

LaTeX table:
`posthoc_output/tables/table_memorization_nn.tex`.

Sources: `posthoc_output/memorization_nn.csv`,
`posthoc_output/memorization_nn_summary.csv`.

---

## 6. File index for the writing agent

Updated / new artefacts (paths relative to
`/media/mpascual/Sandisk2TB/completed/jsddpm/results/epilepsy/icip2026/camera_ready/evaluations/`):

```
eval_output/
├── fold_metrics.csv                       # 3 arches × 3 folds = 9 rows
├── summary_metrics.csv                    # 3 arches
├── kid_per_zbin.csv                       # 7 zbins × 3 arches × 3 folds = 63 rows
├── kid_per_zbin_summary.csv               # 21 rows
├── wasserstein_per_feature.csv            # 9 rows (per-cell mask morphology W1)
└── eval_sample_counts.json                # 9 cells with n_replicas, n_synth

posthoc_output/
├── tau_sensitivity.csv                    # 81 rows (9 τ × 3 arches × 3 folds)
├── tau_sensitivity_summary.csv            # 27 rows
├── cross_fold_comparison.json             # cross-fold statistical tests
├── memorization_nn.csv                    # 9 rows
├── memorization_nn_summary.csv            # 3 rows
├── memorization_nn.json
├── paired_lesion_mmd.csv                  # 9 rows
├── paired_lesion_mmd_summary.csv          # 3 rows
├── paired_lesion_features.csv             # 60 rows (real + 3 arches × 5 features × 3 folds)
├── paired_lesion_mmd.json
└── tables/
    ├── table_ablation.tex                 # three-architecture KID / LPIPS / MMD-MF
    ├── table_main_updated.tex             # updated paper Table 1 cell
    ├── table_tau_sensitivity_shared.tex
    ├── table_tau_sensitivity_decoupled.tex
    ├── table_tau_sensitivity_zerocoupled.tex
    ├── table_memorization_nn.tex
    └── table_paired_lesion_mmd.tex

figures/
└── ablation_comparison.pdf                # 4-row grid (shared / decoupled / zero-coupling / real)
                                            # × 7 z-bins, with per-cell KID labels
```

---

## 7. Caveats to surface in the manuscript

- **Replica imbalance.** Zero-coupling cells contribute 18 000 / 27 000 / 27 000
  synthetic slices per fold versus 108 000 for shared and decoupled. Subset-MMD
  estimators are designed to be robust under this imbalance (they cap the
  subset to `min(N_real, N_synth)`), but per-fold standard deviations on
  zero-coupling are tighter than they would be at parity.
- **MMD-PL variance.** The polynomial-kernel MMD on a 5-D vector has high
  shot noise; the per-fold std for shared (72.4) and zero-coupling (47.4) are
  comparable to or larger than the means. The per-feature Wasserstein-1
  columns (§5.3) are lower-variance and show the same ordering on every fold
  — they are the more reliable summary for the manuscript.
- **Parameter budget.** Zero-coupling has ≈ 2× the total parameter count of
  the shared / decoupled pair. The shared-vs-decoupled comparison is
  param-matched; the shared-vs-zero-coupling comparison is not.
- **No architecture matches `frac_bright` of real lesions.** All three sit
  below the real baseline (0.235). The mask-on-hyperintense-tissue alignment
  is imperfect for every cell in the grid; the gap is largest for
  zero-coupling.
- **Figure regeneration replica cap.** `figures/ablation_comparison.pdf` was
  re-rendered with `--max-replicas 3` (uniform across architectures) because
  loading all 12 shared and decoupled replicas exhausted system RAM. The
  per-cell KID labels on the figure come from `kid_per_zbin_summary.csv`
  computed on the full replica set, so the labels remain comparable across
  rows.
