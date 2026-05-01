# Narrative Redirection Guide for the SASHIMI 2026 Submission

## Purpose of this document

Once the three studies (1.1 zero-coupling baseline, 1.2 distributional fidelity via TRTS, 2.1 linear-probe analysis) finish executing, the paper must be restructured from its current 6-page ICIP form to a 10-page LNCS SASHIMI form, with the new evidence integrated coherently. This guide tells the local agent **what to change in the manuscript** and how to handle each plausible result outcome.

This is a **content guide**, not a coding task. The agent's deliverable is a revised LaTeX manuscript and the supporting figures. No new experiments should be run as part of this guide — the experiments are fixed by docs `01_zero_coupling_baseline.md`, `02_distributional_fidelity_trts.md`, and `03_linear_probe_analysis.md`.

## Inputs the agent must have before starting

1. The current ICIP manuscript LaTeX source (likely `paper/main.tex` or `manuscript/`).
2. The SASHIMI 2026 LNCS template (download from `https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines`). The page limit is **10 pages including references plus 2 pages of appendix**, single-column LNCS layout.
3. `RESULTS.md` from each of the three study directories.
4. The newly produced figures and tables in `outputs/zero_coupling/`, `outputs/trts/`, `outputs/linear_probe/`.

## Step 1 — Confirm the result outcomes before editing

Before touching the manuscript, classify each study's outcome into one of three buckets — this dictates the narrative structure.

### Study 1.1 — Zero-coupling baseline

| Bucket | Criterion | Narrative implication |
|---|---|---|
| **A. Strong joint-training advantage** | Both shared and decoupled significantly outperform independent on MMD-MF (Friedman p<0.05, Cliff's δ ≥ 0.5) | Lead with the coupling-continuum framing. Independent baseline becomes a clean "lower-bound on what conditioning alone can do." |
| **B. Joint training matters only for masks** | Shared/decoupled beat independent on MMD-MF but not on KID/LPIPS | Frame as "joint training is necessary specifically for cross-modal mask-anatomy alignment, not for marginal image realism." This is actually a **stronger** scientific claim. |
| **C. Independent baseline is competitive** | No significant gap between independent and the joint variants | Demote the architectural-coupling story, lead with the loss-geometry / prediction-target factorial as the primary contribution. The shared-vs-decoupled gap (still 2.6× MMD-MF) becomes a secondary architectural finding. |

### Study 1.2 — Distributional fidelity (TRTS)

| Bucket | Criterion | Narrative implication |
|---|---|---|
| **D. Small TRTS gap, ordering as predicted** | Δ_TRTS small for shared, larger for decoupled, largest for independent | Triangulates with KID/LPIPS/MMD-MF; the synthetic distribution is faithful at multiple feature levels. Strong story. |
| **E. Small TRTS gap, no ordering** | All synthetic configurations achieve Dice within noise of real | Frame as "all configurations are sufficient for segmenter-level fidelity, but they differ at the morphology-distribution level (MMD-MF)" — uses the TRTS to distinguish coarse vs. fine fidelity. |
| **F. Large TRTS gap** | Shared Δ_TRTS > 0.15 Dice | This is a problem for the synthesis-quality story. **Do not hide it.** Frame as a calibrated limitation: "FID-style metrics may overstate fidelity relative to task-relevant feature spaces." This is publishable and honest. |

### Study 2.1 — Linear probe

| Bucket | Criterion | Narrative implication |
|---|---|---|
| **G. Hypothesis confirmed** | Shared probe R² > decoupled-concatenated R² on eccentricity and mean_intensity (Wilcoxon p<0.05) | Include §3.3 as written in the experimental brief. Provides mechanistic explanation. |
| **H. Hypothesis partially confirmed** | One of the two key targets shows the predicted gap, the other does not | Include §3.3 but narrow the claim to the confirmed target. Be specific about what is and is not encoded. |
| **I. No probe gap** | No significant difference on any key target | **Drop §3.3 entirely**. Use the page space for an expanded qualitative analysis of failure modes, or for a deeper limitations discussion. Do not torture the data. |

## Step 2 — Apply the narrative-redirect template by outcome

The combinations that matter are joint outcomes. The most likely scenarios are:

### Scenario α: A + D + G (best case, ~30% probability)

This is the clean three-way confirmation. Restructure aggressively:

- **Title**: *"SLIM-Diff: A Controlled Study of Representational Coupling for Joint Image–Mask Diffusion under Data Scarcity"*
- **Lead contribution**: the coupling continuum (independent / decoupled / shared) at matched 26.9M parameters.
- **Headline result**: 2.6× MMD-MF gap (shared vs decoupled) explained mechanistically by the linear probe (§3.3) and validated externally by the TRTS gap (§3.4).
- **Loss geometry / prediction target study**: demoted to a secondary contribution that completes the design-axis study.

### Scenario β: A + E + G (second-best, ~25%)

The architectural story holds but TRTS does not separate the configurations. Reframe TRTS as confirming **coarse fidelity** while MMD-MF and the linear probe characterise **fine-grained morphological fidelity**. This is actually a clean two-level fidelity story.

- Add one paragraph in the discussion about the **resolution of fidelity metrics**: TRTS measures whether the synthetic image-mask coupling is preserved at the feature level a real-trained segmenter uses; MMD-MF measures whether the morphology-distribution fine structure is preserved. They answer different questions.

### Scenario γ: B + D + G (~15%)

Joint training matters specifically for masks, not for image marginals. Frame the paper around the **modality-asymmetric role of coupling**:

- §3.1: image-side metrics (KID, LPIPS) show all three configurations are similar; image marginal is easy to model independently.
- §3.2: mask-side metrics (MMD-MF, per-feature Wasserstein) show a strong coupling effect.
- §3.3: the linear probe localises the asymmetry to the bottleneck — shared encodes mask geometry jointly with image features, decoupled does not.

### Scenario δ: C or F or I (~30%)

One or more results disagree with the hypothesis. **Do not panic, do not hide.** The honest framing options:

- **C** (independent competitive): demote architecture, promote the loss-geometry × prediction-target factorial. Title becomes *"An Empirical Study of Training-Objective Design for Joint Image–Mask Diffusion under Data Scarcity"*. The factorial study is genuinely novel and well-controlled.
- **F** (large TRTS gap): include as a calibrated finding. Title can absorb this: *"...with Calibration Against a Domain-Relevant Reference Model"*. Emphasise that distributional metrics and task-relevant fidelity can diverge, which is itself a contribution to the synthesis-evaluation literature.
- **I** (no probe gap): drop §3.3 entirely; expand qualitative figures and limitations to fill the page.

## Step 3 — Section-by-section rewrite checklist

Apply these edits regardless of outcome bucket; outcome-specific edits are noted inline.

### Title

Replace `SLIM-Diff: Shared Latent Image-Mask Diffusion with Lp Loss for Data-Scarce Epilepsy FLAIR MRI` with one of:

- (Scenarios α, β, γ) *"SLIM-Diff: A Controlled Study of Representational Coupling for Joint Image–Mask Diffusion under Data Scarcity"*
- (Scenario C) *"An Empirical Study of Training-Objective Design for Joint Image–Mask Diffusion under Data Scarcity"*
- (Scenario F) *"Joint Image–Mask Diffusion under Data Scarcity: Coupling, Loss Geometry, and Calibration Against a Domain-Relevant Reference Model"*

### Abstract

Restructure to four sentences:
1. Problem (data-scarce joint image–mask synthesis for FCD FLAIR MRI).
2. Method (controlled factorial study over coupling × prediction target × loss geometry, at matched 26.9M parameters).
3. Key finding (the strongest single result from Step 1's outcome bucket).
4. Validation (TRTS gap result if D/E, otherwise distributional metrics + linear probe if G/H).

Drop the original "(i)... (ii)..." contributions sentence — that framing primed the ICIP "limited novelty" reading.

### §1 Introduction

Restructure the contributions paragraph. The new five-bullet contribution list is:

1. A coupling-degree study (independent / decoupled / shared) at matched capacity, isolating the role of representational sharing.
2. A factorial study of training-objective design (γ × prediction target).
3. (If G/H) A mechanistic analysis of the shared bottleneck via linear probing.
4. (If D/E) A distributional-fidelity evaluation via a real-trained reference segmenter (TRTS).
5. The evaluation protocol (KID/LPIPS + MMD-MF + per-feature Wasserstein + binarisation sensitivity).

Drop or move into §2 the current sentence on "compactness as implicit regularisation" — it was a defensive move at ICIP; at SASHIMI the coupling-continuum framing makes the case for capacity matching directly.

### §2 Methodology

- **§2.5 (renamed)**: *"Coupling Continuum: Independent, Decoupled, and Shared Bottlenecks"*. Describe all three configurations in one place. Emphasise parameter matching at 26.9M total.
- **§2.x (new)**: *"Distributional Fidelity via a Domain-Relevant Reference Model"*. Describe the TRTS protocol. **Critical**: do not use the words "data augmentation", "downstream task improvement", or any phrasing that implies the synthetic data is being used as training material. The reference segmenter is a **feature extractor**, exactly analogous to the Inception network in KID. Use this analogy explicitly: *"Just as KID employs Inception features as a general-purpose image-quality proxy, the TRTS protocol employs a real-trained nnU-Net as a domain-relevant feature proxy specific to FCD lesion morphology."*
- **§2.y (new, only if outcome G or H)**: *"Bottleneck Linear-Probe Protocol"*. One paragraph defining the probe targets and the subject-stratified probe-train/test split.

### §3 Results and Discussion

Reorganise into:

- **§3.1 Coupling-continuum results** (Tier 1.1 result; replaces and extends the current Table 2). Includes the new independent baseline as a third column.
- **§3.2 Loss-geometry × prediction-target factorial** (current Table 1). Move from §3 lead to §3.2.
- **§3.3 Mechanistic analysis: bottleneck linear probes** (only if G or H).
- **§3.4 Distributional fidelity via the reference segmenter** (only if D, E, or F).
- **§3.5 Binarisation sensitivity** (current Figure 4, unchanged).
- **§3.6 Qualitative analysis** (current Figure 3, unchanged).

Each subsection is ~½ page. The total budget for §3 is approximately 4 pages of the 10-page LNCS allowance.

### §4 Conclusion

Three sentences:
1. The primary finding of the paper, dictated by the outcome bucket from Step 1.
2. The mechanistic / external-validation evidence, if available (G or H, D or E).
3. The honest scope: this is a study of a specific data-scarce regime; generalisation to larger cohorts and other rare-pathology domains is future work.

### §4.1 Limitations

Update to reflect the new evidence:

- The current "no external comparison" limitation can be relaxed since the independent baseline (Study 1.1) is now an external-style comparison.
- The current "no downstream evaluation" limitation can be **reframed**, not removed: even with the TRTS protocol in place, a full TSTR data-augmentation evaluation is not performed because (i) it would require training a segmenter from scratch under multiple training-data mixes, and (ii) it asks a different question than this paper sets out to answer (fidelity, not utility).
- Add a forward-looking limitation about other rare-pathology FLAIR cohorts (MS, stroke), as in the existing manuscript.

## Step 4 — Figures

Plan for the 10-page LNCS submission:

1. **Figure 1 (existing)**: SLIM-Diff overview (panels A, B, C). Keep, but update panel B to show the coupling continuum (three configurations side by side) rather than only the shared variant.
2. **Figure 2 (existing)**: similarity / mask-quality metrics across prediction targets and Lₚ. Keep mostly unchanged; consider adding a sub-panel showing the new independent baseline.
3. **New Figure 3**: coupling continuum bar chart (KID, LPIPS, MMD-MF for independent / decoupled / shared, with statistical-significance brackets).
4. **Figure 4 (existing)**: binarisation sensitivity curves. Keep.
5. **Figure 5 (existing)**: qualitative slice gallery. **Expand** to include the independent baseline as an additional row.
6. **New Figure 6 (only if G or H)**: linear-probe R² bars across targets (5 targets × 4 conditions).
7. **New Figure 7 (only if D, E, or F)**: TRTS Dice per configuration vs. real reference, with per-z-bin curve breakdown.

The 10-page LNCS allowance accommodates all of these comfortably; ICIP's 6-page limit was the binding constraint that forced earlier compression.

## Step 5 — Anonymisation and submission hygiene

SASHIMI is **double-blind**. The agent must:

- Replace the GitHub URL `https://github.com/MarioPasc/slim-diff` with `https://anonymous.4open.science/...` for the submission.
- Remove the "Acknowledgements" section in the submitted version (reinsert in camera-ready).
- Remove author affiliations and emails.
- Verify the bibliography does not include any SLIM-Diff self-references in a way that deanonymises (e.g., do not cite "Pascual-González et al., 2026" — cite as "[author-citation]").
- Check that any preprint URL (the arXiv ID 2602.03372 in the README's BibTeX) is not exposed in the submitted PDF.

## Step 6 — Final checklist

Before submission on 1 July 2026, verify:

- [ ] Page count ≤ 10 main + ≤ 2 appendix in LNCS template.
- [ ] All three new studies have results integrated (or, in the case of dropped §3.3, the page space is reallocated cleanly).
- [ ] The TRTS section is framed as fidelity, not utility, in every sentence.
- [ ] The narrative outcome bucket from Step 1 is consistently reflected in title, abstract, contributions list, results subsections, and conclusion.
- [ ] Figures and tables match the outcome bucket (no stale figures from a different scenario).
- [ ] All co-authors have signed off on the new framing.
- [ ] Anonymisation is complete and verified by an independent reader.
- [ ] PDF generated from a clean LaTeX build with no warnings.

## Step 7 — What this document does not handle

The agent should explicitly stop and consult the user if:

- Any outcome bucket combination not listed in Step 2 occurs (e.g., a contradictory result the matrix did not anticipate).
- The TRTS experiment cannot be completed (peer's nnU-Net never arrives, Med-SAM2 fallback also fails). In that case the paper drops §3.4 and rebalances pages but the user should sign off on this decision before the agent proceeds.
- Any of the studies produce a result that contradicts the existing Table 1 / Table 2 numbers in the manuscript (this would indicate a regression or bug, not a finding, and must be debugged before paper edits).
- The 10-page budget cannot accommodate the full restructure even with aggressive trimming. (Unlikely — 10 LNCS pages is roughly 50% more text than the current 6-page IEEE format, even allowing for expanded results.)

In all these cases the agent should write the diagnostic into a `BLOCKER.md` file and request user input before continuing.
