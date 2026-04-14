"""Post-hoc analyses for the ICIP 2026 camera-ready rebuttal (TASK-05).

Consumes the outputs of TASK-04 (per-cell / summary metric CSVs) and the raw
continuous synthetic masks produced by TASK-03; produces:

* ``tau_sensitivity.csv`` + ``tau_sensitivity_summary.csv``
  — MMD-MF sensitivity to the mask binarisation threshold tau (R1.5).
* ``cross_fold_comparison.json``
  — shared-vs-decoupled paired differences, Cliff's delta, Cohen's d,
    directional consistency, Wilcoxon (R1.1/R1.3/R2.2).
* ``tables/table_*.tex``
  — publication-ready LaTeX tables for the camera-ready manuscript.

The public entry point is :func:`.cli.main`
(``python -m src.diffusion.scripts.similarity_metrics.posthoc.cli``).
"""

from __future__ import annotations
