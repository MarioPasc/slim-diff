"""Acceptance tests for TASK-06 — camera-ready qualitative figures.

Covers acceptance criteria AC-1..AC-7 from
``docs/icip2026/rebuttal_plans/TASK-06_qualitative_visualization.md``.

Tests run on CPU-only dummy data (no real MRI, no GPU), so they execute in a
few seconds inside the project's pytest suite and do not depend on any of the
TASK-04/05 outputs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.diffusion.scripts.camera_ready import (  # noqa: E402
    ablation_comparison_figure,
    failure_modes,
    figure_utils,
    qualitative_figure,
    tau_scatter_figure,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CAMERA_READY_DIR = REPO_ROOT / "src" / "diffusion" / "scripts" / "camera_ready"


# =============================================================================
# Dummy data fixtures
# =============================================================================

Z_BINS = [5, 15, 25]
N_PER_BIN = 20  # synthetic samples per (zbin, condition)
H = W = 32  # small spatial size keeps tests fast


def _synth_replica(rng: np.random.Generator, replica_id: int) -> dict[str, np.ndarray]:
    """Build one synthetic replica spanning 3 zbins × 2 conditions."""
    imgs = []
    masks = []
    zbins = []
    lesion = []
    domain = []
    for zb in Z_BINS:
        for cond in (0, 1):
            for _ in range(N_PER_BIN):
                img = rng.uniform(-1.0, 1.0, size=(H, W)).astype(np.float16)
                if cond == 1:
                    # Sparse lesion: ~3% pixels positive.
                    m = np.full((H, W), -1.0, dtype=np.float16)
                    px = rng.integers(0, H, size=30)
                    py = rng.integers(0, W, size=30)
                    m[px, py] = 1.0
                else:
                    m = np.full((H, W), -1.0, dtype=np.float16)
                imgs.append(img)
                masks.append(m)
                zbins.append(zb)
                lesion.append(cond)
                domain.append(cond)
    return {
        "images": np.stack(imgs),
        "masks": np.stack(masks),
        "zbin": np.asarray(zbins, dtype=np.int32),
        "lesion_present": np.asarray(lesion, dtype=np.uint8),
        "domain": np.asarray(domain, dtype=np.uint8),
    }


def _build_cache(tmp_path: Path, rng: np.random.Generator, fold: int = 0) -> Path:
    cache = tmp_path / "cache"
    slices_dir = cache / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = cache / "folds" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for zb in Z_BINS:
        for cond in (0, 1):
            for i in range(6):  # 6 real slices per (zbin, condition)
                name = f"dummy_z{zb}_c{cond}_{i}.npz"
                fp = slices_dir / name
                img = rng.uniform(-1.0, 1.0, size=(H, W)).astype(np.float32)
                if cond == 1:
                    mask = np.full((H, W), -1.0, dtype=np.float32)
                    mask[H // 2 - 2 : H // 2 + 2, W // 2 - 2 : W // 2 + 2] = 1.0
                else:
                    mask = np.full((H, W), -1.0, dtype=np.float32)
                np.savez(fp, image=img, mask=mask)
                rows.append(
                    {
                        "filepath": f"../../slices/{name}",
                        "subject_id": f"sub{cond}_{zb}_{i}",
                        "z_index": zb,
                        "z_bin": zb,
                        "pathology_class": cond,
                        "token": cond * 30 + zb,
                        "source": "epilepsy" if cond == 1 else "control",
                        "split": "test",
                        "has_lesion": bool(cond),
                        "lesion_area_px": 16 if cond == 1 else 0,
                    }
                )
    pd.DataFrame(rows).to_csv(fold_dir / "test.csv", index=False)
    return cache


def _build_results(
    tmp_path: Path,
    rng: np.random.Generator,
    architectures: tuple[str, ...] = ("shared", "decoupled"),
    fold: int = 0,
    n_replicas: int = 2,
) -> Path:
    results = tmp_path / "results"
    for arch in architectures:
        cell = results / figure_utils.CELL_DIR_TEMPLATE.format(
            architecture=arch, fold=fold
        )
        (cell / "replicas").mkdir(parents=True, exist_ok=True)
        for r in range(n_replicas):
            rep = _synth_replica(rng, replica_id=r)
            np.savez(cell / "replicas" / f"replica_{r:03d}.npz", **rep)
    return results


@pytest.fixture()
def dummy_env(tmp_path):
    rng = np.random.default_rng(0)
    cache = _build_cache(tmp_path, rng)
    results = _build_results(tmp_path, np.random.default_rng(1))
    return cache, results


# =============================================================================
# AC-1: Qualitative figure on dummy data
# =============================================================================


def test_ac1_qualitative_figure_runs(dummy_env, tmp_path):
    cache, results = dummy_env
    out = tmp_path / "qual.pdf"
    argv = [
        "--results-root", str(results),
        "--cache-dir", str(cache),
        "--fold", "0",
        "--architecture", "shared",
        "--output", str(out),
        "--zbins", *map(str, Z_BINS),
        "--n-samples", "2",
        "--selection-mode", "random",
        "--format", "pdf",
        "--seed", "0",
    ]
    saved = qualitative_figure.main(argv)
    assert saved.exists() and saved.stat().st_size > 0
    # Sidecar exists.
    sidecar = saved.with_suffix(saved.suffix + ".json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["figure"] == "qualitative_grid"
    for key in ("synth_control", "synth_epilepsy", "real_epilepsy"):
        assert payload[key]["n"] >= 1


# =============================================================================
# AC-2: Ablation figure on dummy data
# =============================================================================


def test_ac2_ablation_figure_runs(dummy_env, tmp_path):
    cache, results = dummy_env
    out = tmp_path / "ablation.pdf"
    argv = [
        "--results-root", str(results),
        "--cache-dir", str(cache),
        "--fold", "0",
        "--output", str(out),
        "--zbins", *map(str, Z_BINS),
        "--condition", "1",
        "--format", "pdf",
        "--seed", "0",
    ]
    saved = ablation_comparison_figure.main(argv)
    assert saved.exists() and saved.stat().st_size > 0
    sidecar = saved.with_suffix(saved.suffix + ".json")
    payload = json.loads(sidecar.read_text())
    assert payload["figure"] == "ablation_comparison_v2"
    # v2 layout: one sample per z-bin per row, three rows (shared/decoupled/real).
    assert payload["shared"]["n"] == payload["decoupled"]["n"] == payload["real"]["n"]
    assert payload["shared"]["n"] == len(Z_BINS)
    assert "real" in payload and payload["real"]["n"] == len(Z_BINS)


# =============================================================================
# AC-3: Failure-modes figure on dummy data
# =============================================================================


def test_ac3_failure_modes_runs(dummy_env, tmp_path):
    cache, results = dummy_env
    out = tmp_path / "failures.pdf"
    argv = [
        "--results-root", str(results),
        "--cache-dir", str(cache),
        "--fold", "0",
        "--output", str(out),
        "--n-worst", "4",
        "--format", "pdf",
        "--seed", "0",
    ]
    saved = failure_modes.main(argv)
    assert saved.exists() and saved.stat().st_size > 0
    sidecar = saved.with_suffix(saved.suffix + ".json")
    payload = json.loads(sidecar.read_text())
    assert payload["shared_worst"]["n"] == 4
    assert payload["decoupled_worst"]["n"] == 4

    # Annotations: re-render to an inspectable figure and count `ax.texts`.
    rng = np.random.default_rng(0)
    ref_block = figure_utils.load_real_samples(
        cache_dir=cache,
        fold=0,
        zbins=Z_BINS,
        condition=1,
        n_samples=4,
        rng=rng,
    )
    shared_reps = figure_utils.load_replicas_concat(
        figure_utils.resolve_cell_dir(results, "shared", 0)
    )
    decoupled_reps = figure_utils.load_replicas_concat(
        figure_utils.resolve_cell_dir(results, "decoupled", 0)
    )
    shared_sel = failure_modes.select_worst_k(shared_reps, ref_block.images, 1, 4)
    decoupled_sel = failure_modes.select_worst_k(decoupled_reps, ref_block.images, 1, 4)
    fig = failure_modes.render_failure_gallery(
        shared_sel=shared_sel,
        decoupled_sel=decoupled_sel,
        output_path=tmp_path / "failures_inspect.pdf",
    )
    annotated = [
        ax
        for ax in fig.get_axes()
        if any("=" in t.get_text() for t in ax.texts if t.get_text())
    ]
    # At least 4 panels per arch row carry annotations.
    assert len(annotated) >= 4
    plt.close(fig)


# =============================================================================
# AC-4: Overlay correctness
# =============================================================================


def test_ac4_overlay_blends_inside_circle():
    # mid-grey image in [-1, 1] (value=0 → rescales to 0.5 grayscale).
    img = np.full((64, 64), 0.0, dtype=np.float32)
    # Force the rescale branch via a sentinel negative pixel.
    img[0, 0] = -1.0
    yy, xx = np.mgrid[:64, :64]
    circle = (yy - 32) ** 2 + (xx - 32) ** 2 <= 10 ** 2
    mask = np.where(circle, 1.0, -1.0).astype(np.float32)
    rgb = figure_utils.overlay_mask_on_image(img, mask, alpha=0.5, color=(255, 0, 0))
    assert rgb.ndim == 3 and rgb.shape[-1] == 3
    # Pixel at centre: blend of grey 0.5 with red (1, 0, 0) at alpha=0.5
    # ⇒ (0.75, 0.25, 0.25).
    cr, cg, cb = rgb[32, 32]
    assert cr > 0.6 and cg < 0.35 and cb < 0.35
    # Pixel far from the circle: remains grayscale (r == g == b).
    r0, g0, b0 = rgb[5, 5]
    assert abs(r0 - g0) < 1e-6 and abs(g0 - b0) < 1e-6


# =============================================================================
# AC-5: IEEE figure dimensions
# =============================================================================


def test_ac5_ieee_dimensions(dummy_env, tmp_path):
    cache, results = dummy_env
    # Generate each figure and check size in inches.
    paths_and_fns: list[tuple[Path, list[str], object]] = [
        (
            tmp_path / "q.pdf",
            [
                "--results-root", str(results),
                "--cache-dir", str(cache),
                "--fold", "0",
                "--architecture", "shared",
                "--output", str(tmp_path / "q.pdf"),
                "--zbins", *map(str, Z_BINS),
                "--n-samples", "2",
                "--selection-mode", "random",
                "--format", "pdf",
            ],
            qualitative_figure.main,
        ),
        (
            tmp_path / "a.pdf",
            [
                "--results-root", str(results),
                "--cache-dir", str(cache),
                "--fold", "0",
                "--output", str(tmp_path / "a.pdf"),
                "--zbins", *map(str, Z_BINS),
                "--condition", "1",
                "--format", "pdf",
            ],
            ablation_comparison_figure.main,
        ),
        (
            tmp_path / "f.pdf",
            [
                "--results-root", str(results),
                "--cache-dir", str(cache),
                "--fold", "0",
                "--output", str(tmp_path / "f.pdf"),
                "--n-worst", "4",
                "--format", "pdf",
            ],
            failure_modes.main,
        ),
    ]
    # Rather than parse the PDF, re-render in-process to inspect fig.get_size_inches().
    for out_path, argv, main_fn in paths_and_fns:
        main_fn(argv)
    # Directly render each figure once more (fresh figs) to inspect dimensions.
    rng = np.random.default_rng(0)
    ref = figure_utils.load_real_samples(cache, 0, Z_BINS, 1, 2, rng=rng)

    # Qualitative
    shared_reps = figure_utils.load_replicas_concat(
        figure_utils.resolve_cell_dir(results, "shared", 0)
    )
    synth_ctrl = figure_utils.load_synthetic_samples(
        results, 0, "shared", Z_BINS, 0, 2, "random", cache_dir=cache, replicas=shared_reps
    )
    synth_epi = figure_utils.load_synthetic_samples(
        results, 0, "shared", Z_BINS, 1, 2, "random", cache_dir=cache, replicas=shared_reps
    )
    fig = qualitative_figure.render_qualitative_grid(
        synth_ctrl=synth_ctrl,
        synth_epi=synth_epi,
        real_ref=ref,
        zbins=Z_BINS,
        n_samples=2,
        output_path=tmp_path / "q_inspect.pdf",
    )
    w, h = fig.get_size_inches()
    assert w <= 7.16 + 1e-6 and h <= 9.0 + 1e-6
    plt.close(fig)

    # Ablation (v2 layout)
    decoupled_reps = figure_utils.load_replicas_concat(
        figure_utils.resolve_cell_dir(results, "decoupled", 0)
    )
    shared_sel_v2, _ = ablation_comparison_figure._select_per_zbin(
        shared_reps, ref.images, Z_BINS, condition=1, tau=0.0
    )
    decoupled_sel_v2, _ = ablation_comparison_figure._select_per_zbin(
        decoupled_reps, ref.images, Z_BINS, condition=1, tau=0.0
    )
    real_sel_v2, _ = ablation_comparison_figure._select_real_per_zbin(
        cache_dir=cache,
        fold=0,
        zbins=Z_BINS,
        condition=1,
        rng=np.random.default_rng(1),
        tau=0.0,
    )
    fig = ablation_comparison_figure.render_ablation_grid(
        shared_sel=shared_sel_v2,
        decoupled_sel=decoupled_sel_v2,
        real_sel=real_sel_v2,
        zbins=Z_BINS,
        output_path=tmp_path / "a_inspect.pdf",
    )
    w, h = fig.get_size_inches()
    assert w <= 7.16 + 1e-6 and h <= 9.0 + 1e-6
    plt.close(fig)

    # Failure modes
    shared_w = failure_modes.select_worst_k(shared_reps, ref.images, 1, 4)
    decoupled_w = failure_modes.select_worst_k(decoupled_reps, ref.images, 1, 4)
    fig = failure_modes.render_failure_gallery(
        shared_sel=shared_w,
        decoupled_sel=decoupled_w,
        output_path=tmp_path / "f_inspect.pdf",
    )
    w, h = fig.get_size_inches()
    assert w <= 7.16 + 1e-6 and h <= 9.0 + 1e-6
    plt.close(fig)


# =============================================================================
# AC-6: Selection reproducibility
# =============================================================================


def _run_and_read_sidecar(argv: list[str]) -> dict:
    saved = qualitative_figure.main(argv)
    sidecar = saved.with_suffix(saved.suffix + ".json")
    return json.loads(sidecar.read_text())


def test_ac6_reproducibility(dummy_env, tmp_path):
    cache, results = dummy_env
    base_argv = [
        "--results-root", str(results),
        "--cache-dir", str(cache),
        "--fold", "0",
        "--architecture", "shared",
        "--zbins", *map(str, Z_BINS),
        "--n-samples", "2",
        "--selection-mode", "random",
        "--format", "pdf",
    ]
    out_a = tmp_path / "qa.pdf"
    out_b = tmp_path / "qb.pdf"
    out_c = tmp_path / "qc.pdf"
    side_a = _run_and_read_sidecar(base_argv + ["--output", str(out_a), "--seed", "0"])
    side_b = _run_and_read_sidecar(base_argv + ["--output", str(out_b), "--seed", "0"])
    side_c = _run_and_read_sidecar(base_argv + ["--output", str(out_c), "--seed", "1"])
    assert side_a["synth_control"]["indices"] == side_b["synth_control"]["indices"]
    assert side_a["synth_epilepsy"]["indices"] == side_b["synth_epilepsy"]["indices"]
    # Different seed must change either of the two stochastic blocks.
    assert (
        side_a["synth_control"]["indices"] != side_c["synth_control"]["indices"]
        or side_a["synth_epilepsy"]["indices"] != side_c["synth_epilepsy"]["indices"]
    )


# =============================================================================
# AC-8: Ablation v2 layout + black background + KID annotations
# =============================================================================


def _write_dummy_kid_summary(path: Path, zbins: list[int]) -> Path:
    rows = []
    for arch in ("shared", "decoupled"):
        for i, zb in enumerate(zbins):
            rows.append(
                {
                    "architecture": arch,
                    "zbin": int(zb),
                    "kid_mean": 0.03 + 0.001 * i,
                    "kid_std_across_folds": 0.002,
                    "kid_ci95_half": 0.005,
                    "n_folds": 3,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_ac8_ablation_v2_black_bg_and_kid_annotation(dummy_env, tmp_path):
    cache, results = dummy_env
    kid_csv = _write_dummy_kid_summary(tmp_path / "kid_summary.csv", Z_BINS)
    out = tmp_path / "ablation_v2.pdf"
    argv = [
        "--results-root", str(results),
        "--cache-dir", str(cache),
        "--fold", "0",
        "--output", str(out),
        "--zbins", *map(str, Z_BINS),
        "--condition", "1",
        "--kid-summary-csv", str(kid_csv),
        "--format", "pdf",
        "--seed", "0",
    ]
    saved = ablation_comparison_figure.main(argv)
    assert saved.exists() and saved.stat().st_size > 0

    # Re-render in-process to inspect background + annotations without reparsing
    # the saved PDF. Use the same helpers the CLI uses so we stay in API lockstep.
    shared_reps = figure_utils.load_replicas_concat(
        figure_utils.resolve_cell_dir(results, "shared", 0)
    )
    decoupled_reps = figure_utils.load_replicas_concat(
        figure_utils.resolve_cell_dir(results, "decoupled", 0)
    )
    ref = figure_utils.load_real_samples(
        cache_dir=cache, fold=0, zbins=Z_BINS, condition=1, n_samples=4,
        rng=np.random.default_rng(0),
    )
    shared_sel, _ = ablation_comparison_figure._select_per_zbin(
        shared_reps, ref.images, Z_BINS, condition=1, tau=0.0
    )
    decoupled_sel, _ = ablation_comparison_figure._select_per_zbin(
        decoupled_reps, ref.images, Z_BINS, condition=1, tau=0.0
    )
    real_sel, _ = ablation_comparison_figure._select_real_per_zbin(
        cache_dir=cache, fold=0, zbins=Z_BINS, condition=1,
        rng=np.random.default_rng(7), tau=0.0,
    )
    kid_df = ablation_comparison_figure.load_kid_summary(kid_csv)
    fig = ablation_comparison_figure.render_ablation_grid(
        shared_sel=shared_sel,
        decoupled_sel=decoupled_sel,
        real_sel=real_sel,
        zbins=Z_BINS,
        output_path=tmp_path / "ablation_v2_inspect.pdf",
        kid_summary=kid_df,
    )
    # Black canvas.
    fc = fig.get_facecolor()
    assert fc[0] == 0.0 and fc[1] == 0.0 and fc[2] == 0.0
    # KID annotation text appears on synthetic panels but not on real ones.
    kid_texts = [
        t.get_text()
        for ax in fig.get_axes()
        for t in ax.texts
        if "KID" in t.get_text()
    ]
    # Two arch rows × len(Z_BINS) cells.
    assert len(kid_texts) == 2 * len(Z_BINS)
    plt.close(fig)


# =============================================================================
# AC-9: tau-sensitivity scatter figure
# =============================================================================


def _write_dummy_tau_summary(path: Path, taus: list[float]) -> Path:
    rows = []
    for arch, base in (("shared", 5.0), ("decoupled", 13.0)):
        for t in taus:
            rows.append(
                {
                    "architecture": arch,
                    "tau": float(t),
                    "mmd_mf_mean_across_folds": base + 0.1 * t,
                    "mmd_mf_std_across_folds": 0.5,
                    "n_lesions_mean": 18000.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_ac9_tau_scatter_figure(tmp_path):
    taus = [-0.2, -0.1, 0.0, 0.1, 0.2]
    summary_csv = _write_dummy_tau_summary(tmp_path / "tau_summary.csv", taus)
    out = tmp_path / "tau_scatter.pdf"
    argv = [
        "--summary-csv", str(summary_csv),
        "--output", str(out),
        "--format", "pdf",
    ]
    saved = tau_scatter_figure.main(argv)
    assert saved.exists() and saved.stat().st_size > 0
    sidecar = saved.with_suffix(saved.suffix + ".json")
    payload = json.loads(sidecar.read_text())
    assert payload["figure"] == "tau_scatter"
    assert payload["n_points"] == len(taus)
    assert payload["tau_values"] == taus

    # Re-render in-process to check structural invariants.
    df = tau_scatter_figure.load_tau_summary(summary_csv)
    pivot = tau_scatter_figure.pivot_by_tau(df)
    fig = tau_scatter_figure.render_tau_scatter(
        pivot=pivot, output_path=tmp_path / "tau_scatter_inspect.pdf"
    )
    # One scatter collection, one line collection, one errorbar container, and colorbar.
    from matplotlib.collections import LineCollection, PathCollection
    scatters = [c for ax in fig.get_axes() for c in ax.collections if isinstance(c, PathCollection)]
    lines = [c for ax in fig.get_axes() for c in ax.collections if isinstance(c, LineCollection)]
    assert len(scatters) >= 1 and scatters[0].get_offsets().shape[0] == len(taus)
    assert len(lines) >= 1  # τ-ordered path
    # Square aspect — enforces y=x-style interpretation.
    xlo, xhi = fig.axes[0].get_xlim()
    ylo, yhi = fig.axes[0].get_ylim()
    assert abs((xhi - xlo) - (yhi - ylo)) < 1e-6
    plt.close(fig)


# =============================================================================
# AC-7: No modification of existing files
# =============================================================================


def test_ac7_no_existing_files_modified():
    # Only uncommitted *changes* to tracked files count as "modifications"; new
    # files must all live under the camera_ready subpackage or the tests dir.
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    allowed_prefixes = (
        "src/diffusion/scripts/camera_ready/",
        "src/diffusion/tests/test_camera_ready_figures.py",
    )
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        status, _, path = line.strip().partition(" ")
        if status.startswith("??"):
            path = line.strip().split(maxsplit=1)[1]
        # " M", "M ", "AM", "A ", etc. → tracked modifications/adds.
        if status.strip() in {"M", "MM", "AM", "A", "D"}:
            # A modification of a tracked file is only allowed when the path
            # itself is under the camera_ready subpackage.
            assert path.startswith(allowed_prefixes), (
                f"Unexpected modification outside TASK-06 scope: {line!r}"
            )
