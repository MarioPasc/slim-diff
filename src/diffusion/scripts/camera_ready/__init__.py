"""Camera-ready qualitative figure scripts (TASK-06).

IEEE-compliant figure generation for the ICIP 2026 camera-ready rebuttal.
The three figure scripts (``qualitative_figure``, ``ablation_comparison_figure``,
``failure_modes``) share utilities from :mod:`figure_utils`. None of these
modules import from TASK-04 / TASK-05; all data is loaded directly from
the ``.npz`` replicas and the per-fold CSV triplets.
"""
