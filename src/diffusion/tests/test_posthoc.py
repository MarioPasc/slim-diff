"""Acceptance tests for TASK-05 (post-hoc analyses).

All tests CPU-only; each builds its own isolated fixture under ``tmp_path``.
The τ sensitivity tests rely on :class:`MaskMorphologyDistanceComputer`,
which is pure CPU (scikit-image features + polynomial-kernel MMD).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.diffusion.scripts.similarity_metrics.data.fold_loaders import FoldEvalData
from src.diffusion.scripts.similarity_metrics import fold_evaluation as fe
from src.diffusion.scripts.similarity_metrics.posthoc.cross_fold_comparison import (
    cross_fold_comparison,
)
from src.diffusion.scripts.similarity_metrics.posthoc.latex_tables import (
    generate_ablation_table,
    generate_all_tables,
    generate_tau_sensitivity_table,
)
from src.diffusion.scripts.similarity_metrics.posthoc.tau_sensitivity import (
    compute_tau_sensitivity,
    save_tau_sensitivity_outputs,
)


# ---------------------------------------------------------------------------
# Fixture builders (self-contained, no dependency on other test files)
# ---------------------------------------------------------------------------


def _stamped_binary_mask(
    img_size: int, block_half: int, fill_value: float,
) -> np.ndarray:
    """Single mask with a centred square at ``fill_value`` (background = -1)."""
    m = np.full((img_size, img_size), -1.0, dtype=np.float32)
    c = img_size // 2
    m[c - block_half : c + block_half, c - block_half : c + block_half] = fill_value
    return m


def _make_fold_eval_data(
    n: int = 50,
    img_size: int = 32,
    block_half: int = 5,
    synth_fill: float = 0.3,
    dtype_synth: np.dtype = np.float16,
    fold: int = 0,
    architecture: str = "shared",
) -> FoldEvalData:
    """In-memory ``FoldEvalData`` with matched-geometry real / synth lesions.

    * ``real_masks``: ``{-1, +1}`` binary, centred block of side ``2*block_half``.
    * ``synth_masks``: continuous; background at ``-1``, centred block at
      ``synth_fill`` (so the block is detected for every τ ∈ (-1, synth_fill)).
    """
    real = np.stack(
        [_stamped_binary_mask(img_size, block_half, 1.0) for _ in range(n)]
    ).astype(np.float32)
    synth = np.stack(
        [_stamped_binary_mask(img_size, block_half, synth_fill) for _ in range(n)]
    ).astype(dtype_synth)
    real_imgs = np.zeros((n, img_size, img_size), dtype=np.float32)
    synth_imgs = np.zeros((n, img_size, img_size), dtype=dtype_synth)
    return FoldEvalData(
        fold=fold,
        architecture=architecture,
        real_images=real_imgs,
        real_masks=real,
        real_zbins=np.zeros(n, dtype=np.int32),
        synth_images=synth_imgs,
        synth_masks=synth,
        synth_zbins=np.zeros(n, dtype=np.int32),
        synth_domains=np.zeros(n, dtype=np.int32),
        n_replicas=1,
    )


def _synthetic_fold_metrics_csv(
    tmp_path: Path,
    shared_values: dict[str, list[float]],
    decoupled_values: dict[str, list[float]],
    n_folds: int = 3,
) -> Path:
    """Write a minimal ``fold_metrics.csv`` with user-supplied values.

    ``shared_values`` / ``decoupled_values`` map each metric key
    (``kid_mean, lpips_mean, mmd_mf_mean``) to a list of length ``n_folds``.
    ``kid_std`` etc. are filled with small constants for completeness.
    """
    rows: list[dict] = []
    for fold in range(n_folds):
        for arch, vals in (("shared", shared_values), ("decoupled", decoupled_values)):
            rows.append(
                {
                    "fold": fold,
                    "architecture": arch,
                    "kid_mean": vals["kid_mean"][fold],
                    "kid_std": 0.001,
                    "lpips_mean": vals["lpips_mean"][fold],
                    "lpips_std": 0.002,
                    "mmd_mf_mean": vals["mmd_mf_mean"][fold],
                    "mmd_mf_std": 0.01,
                    "n_real": 100,
                    "n_synth": 1000,
                }
            )
    path = tmp_path / "fold_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# AC-1: τ sensitivity on dummy data
# ---------------------------------------------------------------------------


def test_ac1_tau_sensitivity_on_dummy_data(tmp_path: Path) -> None:
    data = _make_fold_eval_data(n=50, img_size=32, block_half=5, synth_fill=0.3)
    tau_values = [-0.2, -0.1, 0.0, 0.1, 0.2]

    result = compute_tau_sensitivity(
        data,
        tau_values=tau_values,
        min_lesion_size_px=3,
        subset_size=20,
        num_subsets=10,
        seed=42,
    )

    assert result.tau_values == [float(t) for t in tau_values]
    assert len(result.mmd_mf_values) == 5
    assert len(result.n_lesions_detected) == 5

    # MMD-MF at τ=0.0 is finite and ≥ 0.
    tau0_idx = tau_values.index(0.0)
    assert np.isfinite(result.mmd_mf_values[tau0_idx])
    assert result.mmd_mf_values[tau0_idx] >= 0

    # Lesions are detected for τ ∈ {-0.1, 0.0, 0.1} (all within the lesion
    # value 0.3, so every synth mask contributes one connected component).
    for tau in (-0.1, 0.0, 0.1):
        i = tau_values.index(tau)
        assert result.n_lesions_detected[i] > 0, (
            f"expected lesions detected at τ={tau}, got {result.n_lesions_detected[i]}"
        )

    # Serialisation round-trip: tau_sensitivity.csv has the expected shape.
    save_tau_sensitivity_outputs([result], tmp_path)
    long_df = pd.read_csv(tmp_path / "tau_sensitivity.csv")
    assert len(long_df) == 5  # 1 fold × 1 arch × 5 τ
    assert set(long_df.columns) == {
        "fold", "architecture", "tau", "mmd_mf_mean", "mmd_mf_std", "n_lesions",
    }
    summary_df = pd.read_csv(tmp_path / "tau_sensitivity_summary.csv")
    assert len(summary_df) == 5  # per architecture (1) × per τ (5)


# ---------------------------------------------------------------------------
# AC-2: cross-fold comparison, shared consistently wins
# ---------------------------------------------------------------------------


def test_ac2_cross_fold_comparison_clear_separation(tmp_path: Path) -> None:
    shared = {
        "kid_mean": [0.010, 0.012, 0.014],
        "lpips_mean": [0.220, 0.225, 0.228],
        "mmd_mf_mean": [1.10, 1.20, 1.15],
    }
    decoupled = {
        "kid_mean": [0.020, 0.024, 0.025],
        "lpips_mean": [0.260, 0.270, 0.265],
        "mmd_mf_mean": [1.80, 1.90, 1.85],
    }
    csv_path = _synthetic_fold_metrics_csv(tmp_path, shared, decoupled)

    out_json = tmp_path / "cross_fold_comparison.json"
    report = cross_fold_comparison(csv_path, output_json=out_json)

    assert out_json.exists()
    disk_report = json.loads(out_json.read_text())
    assert disk_report == report

    for key in ("kid", "lpips", "mmd_mf"):
        m = report["metrics"][key]
        # Every field the spec mandates exists.
        for field in (
            "shared", "decoupled", "delta_mean", "delta_std", "cliffs_delta",
            "cohens_d", "all_folds_consistent", "wilcoxon_p",
        ):
            assert field in m, f"metric {key!r} missing field {field!r}"
        assert m["all_folds_consistent"] is True
        assert m["cliffs_delta"] == pytest.approx(1.0)
        assert np.isfinite(m["cohens_d"])
        assert m["cohens_d"] > 0
        assert m["delta_mean"] > 0  # shared wins (delta = decoupled - shared)


# ---------------------------------------------------------------------------
# AC-3: cross-fold comparison with mixed directions
# ---------------------------------------------------------------------------


def test_ac3_cross_fold_comparison_mixed_directions(tmp_path: Path) -> None:
    # Shared wins on folds 0 and 1; decoupled wins on fold 2.
    shared = {
        "kid_mean": [0.010, 0.012, 0.020],
        "lpips_mean": [0.220, 0.225, 0.300],
        "mmd_mf_mean": [1.10, 1.20, 2.50],
    }
    decoupled = {
        "kid_mean": [0.020, 0.024, 0.015],
        "lpips_mean": [0.260, 0.270, 0.280],
        "mmd_mf_mean": [1.80, 1.90, 2.30],
    }
    csv_path = _synthetic_fold_metrics_csv(tmp_path, shared, decoupled)

    out_json = tmp_path / "cross_fold_comparison.json"
    report = cross_fold_comparison(csv_path, output_json=out_json)

    for key in ("kid", "lpips", "mmd_mf"):
        m = report["metrics"][key]
        assert m["all_folds_consistent"] is False
        # Every field still populated.
        assert np.isfinite(m["delta_mean"])
        assert np.isfinite(m["delta_std"])
        assert np.isfinite(m["cliffs_delta"])
        # Cohen's d may still be finite; just require no crash.
        assert "cohens_d" in m


# ---------------------------------------------------------------------------
# AC-4: LaTeX table emission
# ---------------------------------------------------------------------------


def test_ac4_latex_ablation_table(tmp_path: Path) -> None:
    shared = {
        "kid_mean": [0.010, 0.012, 0.014],
        "lpips_mean": [0.220, 0.225, 0.228],
        "mmd_mf_mean": [1.10, 1.20, 1.15],
    }
    decoupled = {
        "kid_mean": [0.020, 0.024, 0.025],
        "lpips_mean": [0.260, 0.270, 0.265],
        "mmd_mf_mean": [1.80, 1.90, 1.85],
    }
    csv_path = _synthetic_fold_metrics_csv(tmp_path, shared, decoupled)

    out_tex = tmp_path / "tables" / "table_ablation.tex"
    body = generate_ablation_table(csv_path, out_tex)
    assert out_tex.exists()

    # Structural checks.
    assert r"\begin{table}" in body
    assert r"\end{table}" in body
    assert r"\begin{tabular}" in body
    assert r"\end{tabular}" in body

    # Winner (shared) has \mathbf; loser (decoupled) does not. Match the
    # data rows by their row labels from ARCH_LABELS, not the caption text.
    lines = body.splitlines()
    shared_line = next(ln for ln in lines if "Shared (ours)" in ln)
    decoupled_line = next(
        ln for ln in lines if ln.lstrip().startswith("Decoupled")
    )
    assert r"\mathbf" in shared_line
    assert r"\mathbf" not in decoupled_line

    # Matched braces sanity (every \mathbf{...} and \caption{...} closes).
    assert body.count("{") == body.count("}")

    # generate_all_tables should also work end-to-end.
    written = generate_all_tables(csv_path, tmp_path)
    assert any(p.name == "table_ablation.tex" for p in written)
    assert any(p.name == "table_main_updated.tex" for p in written)


# ---------------------------------------------------------------------------
# AC-5: τ=0.0 MMD matches the fold_metrics.csv value (atol=0.01)
# ---------------------------------------------------------------------------


def test_ac5_tau0_matches_fold_metrics(tmp_path: Path) -> None:
    data = _make_fold_eval_data(n=60, img_size=32, block_half=5, synth_fill=0.3)

    # 1) τ sensitivity at τ=0 only (matches TASK-04's ``mask > 0`` threshold).
    tau_result = compute_tau_sensitivity(
        data,
        tau_values=[0.0],
        min_lesion_size_px=3,
        subset_size=20,
        num_subsets=30,
        seed=42,
    )
    tau0_mmd = tau_result.mmd_mf_values[0]

    # 2) TASK-04 reference via fold_evaluation.compute_cell_metrics with KID
    #    and LPIPS disabled, KID/LPIPS sections never touch np.random here so
    #    seeding just before MMD is sufficient.
    cfg = {
        "kid": {"enabled": False},
        "lpips": {"enabled": False},
        "mmd_mf": {
            "enabled": True,
            "min_lesion_size_px": 3,
            "subset_size": 20,
            "num_subsets": 30,
            "degree": 3,
            "normalize_features": True,
        },
        "batch_size": 4,
        "seed": 42,
    }
    np.random.seed(42)
    cell_metrics, _wass = fe.compute_cell_metrics(data, cfg, device="cpu")

    assert np.isfinite(cell_metrics.mmd_mf_mean)
    assert np.isfinite(tau0_mmd)
    assert abs(tau0_mmd - cell_metrics.mmd_mf_mean) <= 0.01, (
        f"τ=0 MMD {tau0_mmd:.4f} vs fold-eval MMD "
        f"{cell_metrics.mmd_mf_mean:.4f} diverge by more than atol=0.01"
    )


# ---------------------------------------------------------------------------
# AC-6: no existing modules modified
# ---------------------------------------------------------------------------


def test_ac6_only_posthoc_files_are_new() -> None:
    """``git status`` must not show any modified pre-existing module files.

    New files are expected only under ``posthoc/`` (the package) and under
    ``src/diffusion/tests/test_posthoc.py`` (tests follow project layout
    convention — acknowledged deviation from the strict letter of AC-6 but
    preserves its spirit: no functional module is mutated).
    """
    repo_root = Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        pytest.skip("git not available")
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"git status failed: {exc}")

    allowed_new_prefixes = (
        "src/diffusion/scripts/similarity_metrics/posthoc/",
        "src/diffusion/tests/test_posthoc.py",
    )
    # Pre-existing open items we do not own.
    preexisting_known = {
        "docs/icip2026/rebuttal_plans/progress.md",
        "configs/camera_ready/",
        "slurm/camera_ready/",
        # TASK-04 deliverables already merged on this branch:
        "src/diffusion/scripts/similarity_metrics/fold_evaluation.py",
        "src/diffusion/scripts/similarity_metrics/cli.py",
        "src/diffusion/scripts/similarity_metrics/data/fold_loaders.py",
        "src/diffusion/scripts/similarity_metrics/config/icip2026_camera_ready.yaml",
        "src/diffusion/tests/test_fold_evaluation.py",
        "src/diffusion/tests/test_camera_ready_configs.py",
    }

    offenders: list[str] = []
    for raw in out.stdout.splitlines():
        status, _, path = raw.strip().partition(" ")
        path = path.strip()
        if not path:
            continue
        if path in preexisting_known or any(
            path.startswith(p) for p in preexisting_known
        ):
            continue
        if any(path.startswith(p) for p in allowed_new_prefixes):
            continue
        # If the file is MODIFIED (M) rather than added (??/A), it's a
        # violation of "no existing files modified".
        if status.startswith(("M", "AM")):
            offenders.append(raw)
        elif status.startswith("??"):
            offenders.append(raw)
    assert not offenders, (
        "TASK-05 should only add files under posthoc/ and "
        "tests/test_posthoc.py; offending git status entries:\n"
        + "\n".join(offenders)
    )
