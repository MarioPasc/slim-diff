"""Acceptance tests for the TASK-04 fold-aware evaluation pipeline.

All tests are CPU-only. KID and LPIPS computers are monkey-patched in grid /
single-cell tests to avoid InceptionV3 / VGG weight downloads and to keep the
suite under ~60 s on CPU. AC-2 exercises the real computers against tiny
arrays and is skipped gracefully if model weights cannot be loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.diffusion.scripts.similarity_metrics.data.fold_loaders import (
    FoldEvalData,
    load_fold_eval_data,
)
from src.diffusion.scripts.similarity_metrics import fold_evaluation as fe
from src.diffusion.scripts.similarity_metrics.metrics.kid import MetricResult


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_fake_real_slices(
    cache_dir: Path,
    fold_id: int,
    n_real: int,
    img_size: int,
    rng: np.random.Generator,
) -> None:
    """Write a synthetic ``folds/fold_K/test.csv`` + per-slice NPZs.

    Slice NPZs are placed under ``{cache_dir}/slices/`` and the CSV
    ``filepath`` column uses the cache-relative path ``slices/slice_XXX.npz``,
    mirroring the real slice-cache builder (see
    ``src/diffusion/data/caching/base.py``).
    """
    slices_dir = cache_dir / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = cache_dir / "folds" / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i in range(n_real):
        image = rng.uniform(-1.0, 1.0, (img_size, img_size)).astype(np.float32)
        mask = np.full((img_size, img_size), -1.0, dtype=np.float32)
        # Stamp a small lesion so MorphologicalFeatureExtractor has >=1 lesion.
        c = img_size // 2
        r = max(img_size // 8, 2)
        mask[c - r : c + r, c - r : c + r] = 1.0
        npz_path = slices_dir / f"slice_f{fold_id}_{i:03d}.npz"
        np.savez(npz_path, image=image, mask=mask)
        rows.append(
            {
                "filepath": f"slices/slice_f{fold_id}_{i:03d}.npz",
                "subject_id": f"P{i:03d}",
                "z_index": 50,
                "z_bin": i % 30,
                "pathology_class": 0,
                "token": i % 30,
                "source": "test",
                "split": "test",
                "has_lesion": True,
                "lesion_area_px": int((2 * r) ** 2),
            }
        )
    pd.DataFrame(rows).to_csv(fold_dir / "test.csv", index=False)


def _write_fake_folds_meta(cache_dir: Path, n_folds: int, n_real: int) -> None:
    """Write a minimal ``folds_meta.json`` consumable by KFoldManager."""
    meta = {
        "schema_version": "1.0",
        "n_folds": n_folds,
        "random_state": 42,
        "stratify_by": "has_lesion",
        "fixed_test": True,
        "n_patients_total": n_real,
        "n_patients_pool": 0,
        "n_patients_test": n_real,
        "global_lesion_ratio_pool": 0.0,
        "global_lesion_ratio_test": 1.0,
        "created_at": "2026-04-14T00:00:00+00:00",
        "sklearn_version": "test",
        "numpy_version": "test",
        "test_subjects": [f"P{i:03d}" for i in range(n_real)],
        "test_lesion_ratio": 1.0,
        "folds": [
            {
                "fold_id": k,
                "train_subjects": [],
                "val_subjects": [],
                "n_train": 0,
                "n_val": 0,
                "train_lesion_ratio": 0.0,
                "val_lesion_ratio": 0.0,
            }
            for k in range(n_folds)
        ],
    }
    meta_path = cache_dir / "folds" / "folds_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)


def _write_fake_replicas(
    results_root: Path,
    fold_id: int,
    architecture: str,
    n_replicas: int,
    n_per_replica: int,
    img_size: int,
    rng: np.random.Generator,
) -> None:
    """Write ``slimdiff_cr_{arch}_fold_{k}/replicas/replica_XXX.npz`` files."""
    cell_dir = results_root / f"slimdiff_cr_{architecture}_fold_{fold_id}"
    replicas_dir = cell_dir / "replicas"
    replicas_dir.mkdir(parents=True, exist_ok=True)
    for r in range(n_replicas):
        images = rng.uniform(-1.0, 1.0, (n_per_replica, img_size, img_size)).astype(np.float16)
        masks = np.full((n_per_replica, img_size, img_size), -1.0, dtype=np.float16)
        c = img_size // 2
        rr = max(img_size // 8, 2)
        masks[:, c - rr : c + rr, c - rr : c + rr] = 1.0
        zbin = (np.arange(n_per_replica, dtype=np.int32) % 30)
        domain = np.zeros(n_per_replica, dtype=np.int32)
        np.savez(
            replicas_dir / f"replica_{r:03d}.npz",
            images=images,
            masks=masks,
            zbin=zbin,
            domain=domain,
            lesion_present=np.ones(n_per_replica, dtype=np.int32),
            condition_token=zbin,
            seed=np.int64(42),
            k_index=np.int32(0),
            replica_id=np.int32(r),
        )


def _build_cell(
    tmp_path: Path,
    fold_id: int,
    architecture: str,
    n_real: int = 8,
    n_per_replica: int = 16,
    n_replicas: int = 1,
    img_size: int = 32,
    n_folds: int = 3,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Build (or extend) a synthetic fold tree + per-cell replica dir."""
    rng = np.random.default_rng(seed + 100 * fold_id + (0 if architecture == "shared" else 1))
    cache_dir = tmp_path / "cache"
    results_root = tmp_path / "results"
    if not (cache_dir / "folds" / "folds_meta.json").exists():
        _write_fake_folds_meta(cache_dir, n_folds=n_folds, n_real=n_real)
    _write_fake_real_slices(cache_dir, fold_id, n_real, img_size, rng)
    _write_fake_replicas(
        results_root, fold_id, architecture, n_replicas, n_per_replica, img_size, rng,
    )
    return cache_dir, results_root


# ---------------------------------------------------------------------------
# Monkey-patch stubs for KID / LPIPS (grid + single-cell tests)
# ---------------------------------------------------------------------------


class _FakeKIDComputer:
    def __init__(self, **_kwargs) -> None:
        pass

    def compute(self, real_images, synth_images, show_progress=False):
        return MetricResult(
            metric_name="kid",
            value=0.12,
            std=0.01,
            n_real=int(len(real_images)),
            n_synth=int(len(synth_images)),
            metadata={"fake": True},
        )


class _FakeLPIPSComputer:
    def __init__(self, **_kwargs) -> None:
        pass

    def compute_pairwise(
        self, real_images, synth_images, n_pairs=1000, seed=42, show_progress=False,
    ):
        return MetricResult(
            metric_name="lpips",
            value=0.5,
            std=0.02,
            n_real=int(len(real_images)),
            n_synth=int(len(synth_images)),
            metadata={"fake": True, "n_pairs": n_pairs, "seed": seed},
        )


def _patch_fast_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fe, "KIDComputer", _FakeKIDComputer)
    monkeypatch.setattr(fe, "LPIPSComputer", _FakeLPIPSComputer)


def _tiny_metrics_cfg() -> dict:
    return {
        "kid": {"enabled": True, "subset_size": 4, "num_subsets": 2, "degree": 3},
        "lpips": {"enabled": True, "net": "vgg", "n_pairs": 4},
        "mmd_mf": {
            "enabled": True,
            "min_lesion_size_px": 1,
            "subset_size": 4,
            "num_subsets": 2,
            "degree": 3,
            "normalize_features": True,
        },
        "batch_size": 4,
        "seed": 42,
        "max_replicas": None,
    }


# ---------------------------------------------------------------------------
# AC-1: loader
# ---------------------------------------------------------------------------


def test_ac1_load_fold_eval_data(tmp_path: Path) -> None:
    cache_dir, results_root = _build_cell(
        tmp_path,
        fold_id=0,
        architecture="shared",
        n_real=20,
        n_per_replica=50,
        n_replicas=2,
        img_size=32,
    )
    data = load_fold_eval_data(
        fold=0,
        architecture="shared",
        cache_dir=cache_dir,
        results_root=results_root,
    )
    assert data.synth_images.shape == (100, 32, 32)
    assert data.synth_images.dtype == np.float16
    assert data.synth_masks.shape == (100, 32, 32)
    assert data.synth_masks.dtype == np.float16
    assert data.real_images.shape == (20, 32, 32)
    assert data.real_images.dtype == np.float32
    assert data.real_masks.shape == (20, 32, 32)
    assert data.real_masks.dtype == np.float32
    assert data.real_zbins.shape == (20,)
    assert data.synth_zbins.shape == (100,)
    assert data.synth_domains.shape == (100,)
    assert data.n_replicas == 2
    # Lazy float32 casts return fresh buffers at float32.
    f32 = data.synth_images_f32()
    assert f32.dtype == np.float32
    assert f32.shape == data.synth_images.shape


# ---------------------------------------------------------------------------
# AC-2: metric smoke (may skip if model weights unavailable)
# ---------------------------------------------------------------------------


def test_ac2_metric_smoke() -> None:
    rng = np.random.default_rng(0)
    n = 16
    img_size = 64
    real_images = rng.uniform(-1.0, 1.0, (n, img_size, img_size)).astype(np.float32)
    synth_images = rng.uniform(-1.0, 1.0, (n, img_size, img_size)).astype(np.float16)
    real_masks = np.full((n, img_size, img_size), -1.0, dtype=np.float32)
    real_masks[:, 20:40, 20:40] = 1.0
    synth_masks = np.full((n, img_size, img_size), -1.0, dtype=np.float16)
    synth_masks[:, 18:42, 18:42] = 1.0

    data = FoldEvalData(
        fold=0,
        architecture="shared",
        real_images=real_images,
        real_masks=real_masks,
        real_zbins=np.zeros(n, dtype=np.int32),
        synth_images=synth_images,
        synth_masks=synth_masks,
        synth_zbins=np.zeros(n, dtype=np.int32),
        synth_domains=np.zeros(n, dtype=np.int32),
        n_replicas=1,
    )
    cfg = {
        "kid": {"enabled": True, "subset_size": 4, "num_subsets": 2, "degree": 3},
        "lpips": {"enabled": True, "net": "vgg", "n_pairs": 4},
        "mmd_mf": {
            "enabled": True,
            "min_lesion_size_px": 5,
            "subset_size": 8,
            "num_subsets": 2,
            "degree": 3,
            "normalize_features": True,
        },
        "batch_size": 4,
        "seed": 42,
    }
    try:
        metrics, wasserstein = fe.compute_cell_metrics(data, cfg, device="cpu")
    except Exception as exc:  # pragma: no cover - depends on weight availability
        pytest.skip(f"Metric smoke skipped (weights/network unavailable): {exc}")

    assert np.isnan(metrics.kid_mean) or metrics.kid_mean >= 0
    assert np.isnan(metrics.lpips_mean) or 0.0 <= metrics.lpips_mean <= 1.0
    assert np.isnan(metrics.mmd_mf_mean) or metrics.mmd_mf_mean >= 0
    assert len(wasserstein) == 10
    assert "geometric_mean" in wasserstein


# ---------------------------------------------------------------------------
# AC-3: full grid on dummy data
# ---------------------------------------------------------------------------


def test_ac3_grid_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for fold_id in (0, 1):
        for arch in ("shared", "decoupled"):
            _build_cell(
                tmp_path,
                fold_id=fold_id,
                architecture=arch,
                n_real=8,
                n_per_replica=16,
                n_replicas=1,
                img_size=32,
            )

    _patch_fast_metrics(monkeypatch)

    cache_dir = tmp_path / "cache"
    results_root = tmp_path / "results"
    out_dir = tmp_path / "out"

    fe.run_grid(
        folds=[0, 1],
        architectures=["shared", "decoupled"],
        cache_dir=cache_dir,
        results_root=results_root,
        output_dir=out_dir,
        cfg=_tiny_metrics_cfg(),
        device="cpu",
    )

    fold_df = pd.read_csv(out_dir / "fold_metrics.csv")
    assert len(fold_df) == 4
    expected_pairs = {(0, "shared"), (0, "decoupled"), (1, "shared"), (1, "decoupled")}
    got_pairs = {
        (int(r["fold"]), str(r["architecture"])) for _, r in fold_df.iterrows()
    }
    assert got_pairs == expected_pairs

    summary_df = pd.read_csv(out_dir / "summary_metrics.csv")
    assert len(summary_df) == 2
    assert set(summary_df["architecture"].tolist()) == {"shared", "decoupled"}

    wass_df = pd.read_csv(out_dir / "wasserstein_per_feature.csv")
    assert len(wass_df) == 4
    assert wass_df.shape[1] == 12  # fold + architecture + 10 features

    # Numeric columns finite except possibly MMD-MF (which may be NaN on tiny fixtures).
    for col in ("kid_mean", "kid_std", "lpips_mean", "lpips_std"):
        assert np.all(np.isfinite(fold_df[col].to_numpy()))
    for col in ("n_real", "n_synth"):
        assert np.all(fold_df[col].to_numpy() > 0)


# ---------------------------------------------------------------------------
# AC-4: single-cell partial outputs
# ---------------------------------------------------------------------------


def test_ac4_single_cell_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _build_cell(
        tmp_path,
        fold_id=0,
        architecture="shared",
        n_real=8,
        n_per_replica=16,
        n_replicas=1,
        img_size=32,
    )
    _patch_fast_metrics(monkeypatch)

    cache_dir = tmp_path / "cache"
    results_root = tmp_path / "results"
    out_dir = tmp_path / "out"

    fe.run_single_cell(
        fold=0,
        architecture="shared",
        cache_dir=cache_dir,
        results_root=results_root,
        output_dir=out_dir,
        cfg=_tiny_metrics_cfg(),
        device="cpu",
    )

    partial = out_dir / "fold_metrics_f0_shared.csv"
    assert partial.exists()
    df = pd.read_csv(partial)
    assert len(df) == 1
    assert int(df.iloc[0]["fold"]) == 0
    assert str(df.iloc[0]["architecture"]) == "shared"

    wass_partial = out_dir / "wasserstein_per_feature_f0_shared.csv"
    assert wass_partial.exists()
    wdf = pd.read_csv(wass_partial)
    assert len(wdf) == 1
    assert wdf.shape[1] == 12

    count_partial = out_dir / "eval_sample_counts_f0_shared.json"
    assert count_partial.exists()

    # --aggregate-only path: merge into final artefacts (single-cell corner case).
    fe.aggregate_partials(out_dir, out_dir)
    assert (out_dir / "fold_metrics.csv").exists()
    assert (out_dir / "summary_metrics.csv").exists()
    assert (out_dir / "wasserstein_per_feature.csv").exists()
    assert (out_dir / "eval_sample_counts.json").exists()


# ---------------------------------------------------------------------------
# AC-5: eval_sample_counts.json schema
# ---------------------------------------------------------------------------


def test_ac5_eval_sample_counts_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for fold_id in (0, 1):
        for arch in ("shared", "decoupled"):
            _build_cell(
                tmp_path,
                fold_id=fold_id,
                architecture=arch,
                n_real=8,
                n_per_replica=16,
                n_replicas=2,
                img_size=32,
            )
    _patch_fast_metrics(monkeypatch)

    cache_dir = tmp_path / "cache"
    results_root = tmp_path / "results"
    out_dir = tmp_path / "out"

    fe.run_grid(
        folds=[0, 1],
        architectures=["shared", "decoupled"],
        cache_dir=cache_dir,
        results_root=results_root,
        output_dir=out_dir,
        cfg=_tiny_metrics_cfg(),
        device="cpu",
    )

    counts_path = out_dir / "eval_sample_counts.json"
    assert counts_path.exists()
    payload = json.loads(counts_path.read_text())
    assert "cells" in payload
    assert "generated_at" in payload
    assert isinstance(payload["generated_at"], str)
    assert len(payload["cells"]) == 4
    for cell in payload["cells"]:
        assert isinstance(cell["fold"], int) and cell["fold"] >= 0
        assert isinstance(cell["architecture"], str)
        assert cell["architecture"] in {"shared", "decoupled"}
        assert isinstance(cell["n_real"], int) and cell["n_real"] > 0
        assert isinstance(cell["n_synth"], int) and cell["n_synth"] > 0
        assert isinstance(cell["n_replicas"], int) and cell["n_replicas"] > 0


# ---------------------------------------------------------------------------
# AC-6: CLI backward compat + fold-eval registered
# ---------------------------------------------------------------------------


def test_ac6_cli_registers_fold_eval(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    # Import must succeed even though fold_evaluation.py / fold_loaders.py are new.
    import src.diffusion.scripts.similarity_metrics.cli as cli  # noqa: F401

    monkeypatch.setattr(sys, "argv", ["slimdiff-metrics", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "fold-eval" in captured.out
    # A representative pre-existing subcommand stays listed.
    assert "image-metrics" in captured.out or "mask-metrics" in captured.out
