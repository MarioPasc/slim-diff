"""Unit tests for :mod:`src.diffusion.data.kfold`.

CPU-only; no NPZ loading required (all tests operate on CSV fixtures). The
fixture constructs a synthetic mini-cache of 30 patients with a roughly
60/50 pool/test lesion ratio so that stratified folding is non-trivial.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from src.diffusion.data.kfold import (
    FoldAssignment,
    KFoldManager,
    main as kfold_main,
)

CACHE_FIELDNAMES: tuple[str, ...] = (
    "filepath",
    "subject_id",
    "z_index",
    "z_bin",
    "pathology_class",
    "token",
    "source",
    "split",
    "has_lesion",
    "lesion_area_px",
)


# --------------------------------------------------------------------- fixture


def _build_rows_for_subject(
    subject_id: str,
    split: str,
    has_lesion: bool,
    n_slices: int = 6,
    z_bins: int = 30,
) -> list[dict[str, str]]:
    """Generate ``n_slices`` rows for one subject, ~half with lesions if the
    subject is a lesion patient (otherwise all rows are non-lesion).
    """
    rows: list[dict[str, str]] = []
    for z_idx in range(n_slices):
        z_bin = z_idx % z_bins
        slice_has_lesion = has_lesion and (z_idx % 2 == 0)
        pathology = 1 if slice_has_lesion else 0
        token = z_bin + pathology * z_bins
        rows.append(
            {
                "filepath": (
                    f"slices/{subject_id}_z{z_idx:03d}"
                    f"_bin{z_bin:02d}_c{pathology}.npz"
                ),
                "subject_id": subject_id,
                "z_index": str(z_idx),
                "z_bin": str(z_bin),
                "pathology_class": str(pathology),
                "token": str(token),
                "source": "epilepsy",
                "split": split,
                "has_lesion": "True" if slice_has_lesion else "False",
                "lesion_area_px": "42" if slice_has_lesion else "0",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CACHE_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)


def _make_cache(cache_dir: Path) -> dict[str, list[str]]:
    """Create a mock cache at ``cache_dir`` and return a mapping from
    source-split name to the list of subject IDs written into it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 20 train+val pool patients: 12 with lesions, 8 without.
    pool_subjects = [f"sub-{i:03d}" for i in range(20)]
    pool_labels = {
        s: (i < 12) for i, s in enumerate(pool_subjects)  # first 12 lesion
    }

    # Split the pool roughly 12/8 across original train.csv / val.csv so that
    # the fixture stresses the manager's pool-derivation logic.
    train_source = pool_subjects[:12]
    val_source = pool_subjects[12:]

    # 10 held-out test patients: 5 lesion / 5 non-lesion.
    test_source = [f"sub-t{i:02d}" for i in range(10)]
    test_labels = {s: (i < 5) for i, s in enumerate(test_source)}

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    for s in train_source:
        train_rows.extend(
            _build_rows_for_subject(s, "train", has_lesion=pool_labels[s])
        )
    for s in val_source:
        val_rows.extend(
            _build_rows_for_subject(s, "val", has_lesion=pool_labels[s])
        )
    for s in test_source:
        test_rows.extend(
            _build_rows_for_subject(s, "test", has_lesion=test_labels[s])
        )

    _write_csv(cache_dir / "train.csv", train_rows)
    _write_csv(cache_dir / "val.csv", val_rows)
    _write_csv(cache_dir / "test.csv", test_rows)

    return {
        "train": train_source,
        "val": val_source,
        "test": test_source,
    }


@pytest.fixture
def mock_cache(tmp_path: Path) -> tuple[Path, dict[str, list[str]]]:
    cache_dir = tmp_path / "cache"
    subjects_by_split = _make_cache(cache_dir)
    return cache_dir, subjects_by_split


# ------------------------------------------------------------------- helpers


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _count_total_rows(cache_dir: Path) -> int:
    return sum(
        len(_load_csv_rows(cache_dir / f"{s}.csv"))
        for s in ("train", "val", "test")
    )


def _pool_lesion_ratio(
    manager: KFoldManager, subjects: list[str]
) -> float:
    rows, _ = manager._load_rows()
    labels = manager._patient_has_lesion(rows)
    if not subjects:
        return 0.0
    return sum(labels.get(s, 0) for s in subjects) / len(subjects)


# -------------------------------------------------------------------- tests


def test_no_patient_leakage(mock_cache):
    cache_dir, _ = mock_cache
    manager = KFoldManager(cache_dir, n_folds=3, seed=42)
    for fold in manager.create_folds():
        train_set = set(fold.train_subjects)
        val_set = set(fold.val_subjects)
        test_set = set(fold.test_subjects)
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)


def test_fixed_test_set_across_folds(mock_cache):
    cache_dir, subjects_by_split = mock_cache
    manager = KFoldManager(cache_dir, n_folds=3, seed=42)
    folds = manager.create_folds()
    expected_test = tuple(sorted(subjects_by_split["test"]))
    assert all(f.test_subjects == expected_test for f in folds)
    # All three fold objects must literally agree.
    assert folds[0].test_subjects == folds[1].test_subjects == folds[2].test_subjects


def test_every_pool_patient_in_exactly_one_val(mock_cache):
    cache_dir, subjects_by_split = mock_cache
    manager = KFoldManager(cache_dir, n_folds=3, seed=42)
    folds = manager.create_folds()
    pool = set(subjects_by_split["train"]) | set(subjects_by_split["val"])

    counted: dict[str, int] = {s: 0 for s in pool}
    for fold in folds:
        for s in fold.val_subjects:
            counted[s] += 1

    assert set(counted) == pool
    assert all(v == 1 for v in counted.values()), (
        f"Each pool patient must land in exactly one val fold: {counted}"
    )


def test_stratification_balance(mock_cache):
    cache_dir, subjects_by_split = mock_cache
    manager = KFoldManager(cache_dir, n_folds=3, seed=42)
    folds = manager.create_folds()
    pool = list(
        set(subjects_by_split["train"]) | set(subjects_by_split["val"])
    )
    global_ratio = _pool_lesion_ratio(manager, pool)
    assert global_ratio > 0.0

    for fold in folds:
        val_ratio = _pool_lesion_ratio(manager, list(fold.val_subjects))
        # Per spec: within 20% of the global pool ratio.
        assert abs(val_ratio - global_ratio) / global_ratio < 0.20, (
            f"Fold {fold.fold_id}: val lesion ratio {val_ratio:.3f} vs "
            f"global {global_ratio:.3f}"
        )


def test_csv_columns_match(mock_cache):
    cache_dir, _ = mock_cache
    manager = KFoldManager(cache_dir, n_folds=3, seed=42)
    manager.write_fold_csvs()

    source_header = _load_csv_rows(cache_dir / "train.csv")
    assert source_header, "source train.csv should not be empty"
    # Grab headers by re-opening the file with csv.reader to preserve order.
    with open(cache_dir / "train.csv", "r", newline="") as f:
        source_cols = next(csv.reader(f))

    for fold_id in range(3):
        for split in ("train", "val", "test"):
            fold_csv = cache_dir / "folds" / f"fold_{fold_id}" / f"{split}.csv"
            with open(fold_csv, "r", newline="") as f:
                fold_cols = next(csv.reader(f))
            assert fold_cols == source_cols, (
                f"Column mismatch in {fold_csv}: {fold_cols} != {source_cols}"
            )
            assert tuple(fold_cols) == CACHE_FIELDNAMES


def test_split_column_rewritten(mock_cache):
    cache_dir, _ = mock_cache
    manager = KFoldManager(cache_dir, n_folds=3, seed=42)
    manager.write_fold_csvs()
    for fold_id in range(3):
        for split in ("train", "val", "test"):
            fold_csv = cache_dir / "folds" / f"fold_{fold_id}" / f"{split}.csv"
            rows = _load_csv_rows(fold_csv)
            assert rows, f"{fold_csv} is empty"
            assert all(row["split"] == split for row in rows), (
                f"Split column not rewritten in {fold_csv}"
            )


def test_determinism(mock_cache):
    cache_dir, _ = mock_cache
    m1 = KFoldManager(cache_dir, n_folds=3, seed=42)
    m2 = KFoldManager(cache_dir, n_folds=3, seed=42)
    f1 = m1.create_folds()
    f2 = m2.create_folds()
    for a, b in zip(f1, f2):
        assert a == b


def test_slice_counts_sum(mock_cache):
    cache_dir, _ = mock_cache
    manager = KFoldManager(cache_dir, n_folds=3, seed=42)
    manager.write_fold_csvs()

    total = _count_total_rows(cache_dir)
    n_tests = []
    for fold_id in range(3):
        fold_dir = cache_dir / "folds" / f"fold_{fold_id}"
        n_train = len(_load_csv_rows(fold_dir / "train.csv"))
        n_val = len(_load_csv_rows(fold_dir / "val.csv"))
        n_test = len(_load_csv_rows(fold_dir / "test.csv"))
        assert n_train + n_val + n_test == total, (
            f"Fold {fold_id}: {n_train}+{n_val}+{n_test} != {total}"
        )
        n_tests.append(n_test)
    assert len(set(n_tests)) == 1, (
        f"Per-fold test-slice counts must be identical: {n_tests}"
    )


def test_json_roundtrip(mock_cache):
    cache_dir, _ = mock_cache
    m1 = KFoldManager(cache_dir, n_folds=3, seed=42)
    m1.write_fold_csvs()
    meta_path = cache_dir / "folds" / "folds_meta.json"
    assert meta_path.exists()

    m2 = KFoldManager.from_meta_json(meta_path)
    for fold_id in range(3):
        assert m1.get_fold(fold_id) == m2.get_fold(fold_id)

    # Sanity: top-level metadata fields present.
    with open(meta_path, "r") as f:
        meta = json.load(f)
    for required_key in (
        "schema_version", "n_folds", "random_state", "stratify_by",
        "fixed_test", "test_subjects", "folds",
    ):
        assert required_key in meta, f"Missing key {required_key} in meta"
    assert meta["fixed_test"] is True


def test_cli_idempotent(mock_cache, monkeypatch, caplog):
    cache_dir, _ = mock_cache
    argv = [
        "slimdiff-kfold",
        "--cache-dir", str(cache_dir),
        "--n-folds", "3",
        "--seed", "42",
    ]

    # First invocation: writes the tree.
    monkeypatch.setattr(sys, "argv", argv)
    rc = kfold_main()
    assert rc == 0
    meta_path = cache_dir / "folds" / "folds_meta.json"
    assert meta_path.exists()
    first_meta = meta_path.read_text()

    # Second invocation: idempotent no-op (exit 0, meta untouched).
    monkeypatch.setattr(sys, "argv", argv)
    rc = kfold_main()
    assert rc == 0
    assert meta_path.read_text() == first_meta


def test_no_slice_appears_twice(mock_cache):
    cache_dir, _ = mock_cache
    manager = KFoldManager(cache_dir, n_folds=3, seed=42)
    manager.write_fold_csvs()
    for fold_id in range(3):
        fold_dir = cache_dir / "folds" / f"fold_{fold_id}"
        all_paths: list[str] = []
        for split in ("train", "val", "test"):
            all_paths.extend(
                row["filepath"] for row in _load_csv_rows(fold_dir / f"{split}.csv")
            )
        assert len(all_paths) == len(set(all_paths)), (
            f"Fold {fold_id}: duplicate filepath rows across splits."
        )


def test_pool_derivation(mock_cache):
    cache_dir, subjects_by_split = mock_cache
    manager = KFoldManager(cache_dir, n_folds=3, seed=42)
    folds = manager.create_folds()

    all_subjects = set()
    for fold in folds:
        all_subjects |= set(fold.train_subjects)
        all_subjects |= set(fold.val_subjects)
        all_subjects |= set(fold.test_subjects)

    expected_all = (
        set(subjects_by_split["train"])
        | set(subjects_by_split["val"])
        | set(subjects_by_split["test"])
    )
    assert all_subjects == expected_all

    fixed_test = set(folds[0].test_subjects)
    assert fixed_test == set(subjects_by_split["test"])

    pool_seen = set()
    for fold in folds:
        pool_seen |= set(fold.train_subjects)
        pool_seen |= set(fold.val_subjects)
    assert pool_seen == (expected_all - fixed_test)
