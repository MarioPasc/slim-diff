"""Patient-level stratified K-fold splitter for SLIM-Diff.

Produces per-fold ``train.csv`` / ``val.csv`` / ``test.csv`` triplets that are
drop-in replacements for the single-split CSVs written by the slice-cache
builder (``src/diffusion/data/caching/base.py::write_index_csv``). The training
runner selects a fold by pointing ``cfg.data.cache_dir`` at
``{cache_dir}/folds/fold_{k}``; no changes to ``dataset.py`` or
``create_dataloader`` are required.

Design decisions (ICIP 2026 camera-ready):

* The subjects listed in the source ``test.csv`` are held **FIXED** as the
  test set for every fold. Only the ``train + val`` union is 3-folded — the
  held-out third of each fold becomes that fold's validation set.
* Stratification: binary ``has_lesion`` at patient level (positive if ≥1 slice
  row has ``has_lesion=True``).
* The full slice pool is materialised in memory by concatenating the existing
  ``train.csv``/``val.csv``/``test.csv``; no ``master.csv`` artifact is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from src.diffusion.utils.logging import setup_logger

logger = logging.getLogger(__name__)

_META_FILENAME = "folds_meta.json"
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FoldAssignment:
    """Immutable per-fold patient assignment.

    Subject tuples are always sorted to guarantee run-to-run stability of
    JSON outputs and hashes.

    Attributes:
        fold_id: 0-indexed fold identifier.
        train_subjects: Patients assigned to this fold's train split.
        val_subjects:   Patients assigned to this fold's validation split.
        test_subjects:  Patients in the (fixed) test split. Identical across
            all folds when produced by :class:`KFoldManager` in its default
            fixed-test mode.
    """

    fold_id: int
    train_subjects: tuple[str, ...]
    val_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]

    @property
    def n_train(self) -> int:
        return len(self.train_subjects)

    @property
    def n_val(self) -> int:
        return len(self.val_subjects)

    @property
    def n_test(self) -> int:
        return len(self.test_subjects)


class KFoldManager:
    """Manages a patient-level stratified K-fold split over a slice cache.

    The cache directory must already contain ``train.csv``, ``val.csv``, and
    ``test.csv`` written by the slice-cache builder. On :meth:`create_folds`
    the manager reads all three, holds the subjects in the source ``test.csv``
    as a fixed test set, then applies
    :class:`sklearn.model_selection.StratifiedKFold` to the remaining patient
    pool using binary ``has_lesion`` (patient positive if ≥1 slice row has
    ``has_lesion=True``) as the stratification label. The held-out fold of
    each split becomes that fold's validation set.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        n_folds: int = 3,
        seed: int = 42,
        source_csvs: tuple[str, ...] = ("train.csv", "val.csv", "test.csv"),
        fixed_test_from: str = "test.csv",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        self.source_csvs = tuple(source_csvs)
        self.fixed_test_from = str(fixed_test_from)

        if self.fixed_test_from not in self.source_csvs:
            raise ValueError(
                f"fixed_test_from={self.fixed_test_from!r} must appear in "
                f"source_csvs={self.source_csvs!r}."
            )

        self._rows: list[dict[str, str]] | None = None
        self._fieldnames: tuple[str, ...] | None = None
        self._folds: list[FoldAssignment] | None = None
        self._stratified: bool = True

    # ------------------------------------------------------------------ load

    def _load_rows(self) -> tuple[list[dict[str, str]], tuple[str, ...]]:
        """Load and concatenate rows from the three source CSVs.

        Returns:
            Tuple ``(rows, fieldnames)`` where ``rows`` is the concatenated
            list of dictionaries in on-disk order and ``fieldnames`` is the
            header of the first source CSV (used as the canonical field order
            when writing fold CSVs).
        """
        if self._rows is not None and self._fieldnames is not None:
            return self._rows, self._fieldnames

        all_rows: list[dict[str, str]] = []
        canonical_fields: tuple[str, ...] | None = None

        for name in self.source_csvs:
            csv_path = self.cache_dir / name
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Required source CSV not found: {csv_path}"
                )
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = tuple(reader.fieldnames or ())
                if not fieldnames:
                    raise ValueError(f"Empty header in {csv_path}")
                if canonical_fields is None:
                    canonical_fields = fieldnames
                elif fieldnames != canonical_fields:
                    raise ValueError(
                        f"Field mismatch in {csv_path}: {fieldnames} != "
                        f"{canonical_fields}"
                    )
                for row in reader:
                    all_rows.append(dict(row))

        assert canonical_fields is not None
        for required in ("subject_id", "split", "has_lesion"):
            if required not in canonical_fields:
                raise ValueError(
                    f"Source CSV header missing required column {required!r}; "
                    f"got {canonical_fields}."
                )

        self._rows = all_rows
        self._fieldnames = canonical_fields
        return all_rows, canonical_fields

    # ------------------------------------------------------------- fold build

    @staticmethod
    def _patient_has_lesion(rows: Iterable[dict[str, str]]) -> dict[str, int]:
        """Binary ``has_lesion`` per patient computed over ``rows``.

        Consistent with :func:`~src.diffusion.data.splits.compute_subject_characteristics_from_csv`:
        a patient is positive (``1``) if at least one row has
        ``has_lesion`` truthy (case-insensitive ``"true"``).
        """
        labels: dict[str, int] = {}
        for row in rows:
            sid = row["subject_id"]
            is_lesion = row.get("has_lesion", "").strip().lower() == "true"
            # Record the first time seen, then upgrade to 1 on any lesion row.
            labels[sid] = max(labels.get(sid, 0), int(is_lesion))
        return labels

    def create_folds(self) -> list[FoldAssignment]:
        """Compute fold assignments. Idempotent — subsequent calls return the
        cached result unchanged.
        """
        if self._folds is not None:
            return self._folds

        rows, _ = self._load_rows()

        # 1. Fixed test subjects: from rows with split matching the source
        #    test file's implicit split name ("test").
        test_source_split = Path(self.fixed_test_from).stem  # "test"
        test_subjects = sorted({
            row["subject_id"]
            for row in rows
            if row["split"] == test_source_split
        })
        if not test_subjects:
            raise ValueError(
                f"No rows found with split={test_source_split!r}; cannot "
                f"derive a fixed test set from {self.fixed_test_from}."
            )

        # 2. Full patient set and pool (train+val union).
        all_subjects = sorted({row["subject_id"] for row in rows})
        pool = sorted(set(all_subjects) - set(test_subjects))

        # 3. Subject-disjointness between source splits: the same patient must
        #    not straddle train/val/test in the source cache.
        src_membership: dict[str, set[str]] = {}
        for row in rows:
            src_membership.setdefault(row["subject_id"], set()).add(row["split"])
        overlapping = {s: m for s, m in src_membership.items() if len(m) > 1}
        if overlapping:
            raise ValueError(
                "Source CSVs contain subjects present in multiple splits. "
                f"Offenders: {sorted(overlapping.items())[:5]} ..."
            )

        # 4. Binary has_lesion label per pool patient.
        pool_rows = [r for r in rows if r["subject_id"] in set(pool)]
        labels_map = self._patient_has_lesion(pool_rows)
        y = np.array([labels_map[s] for s in pool], dtype=int)

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        min_class = min(n_pos, n_neg)

        # 5. K-fold on the pool. When both classes are present with ≥ n_folds
        #    members each, stratify; when the pool is single-class (common for
        #    disease-only cohorts like the FCD epilepsy dataset where every
        #    patient has a lesion), fall back to plain KFold — stratification
        #    is trivial and StratifiedKFold refuses to split.
        if min_class == 0:
            logger.warning(
                "Pool is single-class (%d has_lesion, %d no_lesion); using "
                "plain KFold — stratification is degenerate.", n_pos, n_neg,
            )
            splitter = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.seed,
            )
            self._stratified = False
        elif min_class < self.n_folds:
            raise ValueError(
                f"StratifiedKFold infeasible: minimum class count {min_class} "
                f"< n_folds {self.n_folds}. Pool has {n_pos} has_lesion and "
                f"{n_neg} no_lesion patients."
            )
        else:
            splitter = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.seed,
            )
            self._stratified = True

        pool_arr = np.asarray(pool)
        folds: list[FoldAssignment] = []
        for fold_id, (train_idx, val_idx) in enumerate(
            splitter.split(np.zeros(len(pool)), y)
        ):
            val_subjects = tuple(sorted(pool_arr[val_idx].tolist()))
            train_subjects = tuple(sorted(pool_arr[train_idx].tolist()))
            folds.append(
                FoldAssignment(
                    fold_id=fold_id,
                    train_subjects=train_subjects,
                    val_subjects=val_subjects,
                    test_subjects=tuple(test_subjects),
                )
            )

        # Sanity: val sizes balanced to within one patient.
        n_val = [f.n_val for f in folds]
        if max(n_val) - min(n_val) > 1:
            logger.warning(
                "Uneven val-fold sizes %s; StratifiedKFold normally keeps this "
                "within 1.", n_val,
            )

        self._folds = folds
        return folds

    # ------------------------------------------------------- query accessors

    def get_fold(self, fold_id: int) -> FoldAssignment:
        folds = self.create_folds()
        if not 0 <= fold_id < len(folds):
            raise IndexError(
                f"fold_id {fold_id} out of range [0, {len(folds)})."
            )
        return folds[fold_id]

    def get_cache_dir_for_fold(self, fold_id: int) -> Path:
        """Directory containing ``train.csv``/``val.csv``/``test.csv`` for a
        given fold. Used by training runners to set ``cfg.data.cache_dir``.
        """
        return self.cache_dir / "folds" / f"fold_{fold_id}"

    # ------------------------------------------------------------- write out

    def write_fold_csvs(self, output_dir: Path | None = None) -> None:
        """Write per-fold CSV triplets to disk.

        Writes atomically: first produces a ``folds.tmp/`` subtree, then swaps
        it into place via :func:`os.replace` to avoid half-written fold dirs.

        Args:
            output_dir: Target directory. Defaults to ``{cache_dir}/folds``.
        """
        folds = self.create_folds()
        rows, fieldnames = self._load_rows()

        out = Path(output_dir) if output_dir is not None else (
            self.cache_dir / "folds"
        )
        tmp = out.parent / (out.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)

        priors_src = self.cache_dir / "zbin_priors_brain_roi.npz"

        for fold in folds:
            fold_dir = tmp / f"fold_{fold.fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_set = set(fold.train_subjects)
            val_set = set(fold.val_subjects)
            test_set = set(fold.test_subjects)

            # `filepath` in source CSVs is stored relative to the source
            # cache_dir. Per-fold CSVs live one or more levels deeper (default
            # `{cache_dir}/folds/fold_N/`), and `SlicesCSVDataset` resolves
            # rows via `self.cache_dir / row["filepath"]`. Rewrite the column
            # as a path relative to the fold dir so the join still lands on
            # the real NPZ. Absolute filepaths are preserved verbatim.
            rel_prefix = os.path.relpath(self.cache_dir, fold_dir)

            # Partition rows; rewrite the "split" column so downstream
            # consumers reading row["split"] see a consistent value, and
            # rebase "filepath" onto the fold dir.
            per_split: dict[str, list[dict[str, str]]] = {
                "train": [], "val": [], "test": [],
            }
            for row in rows:
                sid = row["subject_id"]
                new_row = dict(row)
                fp = new_row.get("filepath", "")
                if fp and not os.path.isabs(fp):
                    new_row["filepath"] = os.path.normpath(
                        os.path.join(rel_prefix, fp)
                    )
                if sid in train_set:
                    new_row["split"] = "train"
                    per_split["train"].append(new_row)
                elif sid in val_set:
                    new_row["split"] = "val"
                    per_split["val"].append(new_row)
                elif sid in test_set:
                    new_row["split"] = "test"
                    per_split["test"].append(new_row)
                # Subjects not in any partition are dropped (should not occur
                # because train∪val∪test = all_subjects by construction).

            for split_name, split_rows in per_split.items():
                csv_path = fold_dir / f"{split_name}.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(fieldnames))
                    writer.writeheader()
                    writer.writerows(split_rows)

            # Symlink the priors file into the fold dir so training pipelines
            # that look it up under cfg.data.cache_dir find it. Use a relative
            # link so the fold dir stays portable.
            if priors_src.exists():
                link_path = fold_dir / priors_src.name
                target = Path("..") / ".." / priors_src.name
                try:
                    link_path.symlink_to(target)
                except (OSError, NotImplementedError) as exc:
                    logger.warning(
                        "Could not symlink priors into %s (%s); copying.",
                        link_path, exc,
                    )
                    shutil.copy2(priors_src, link_path)

        # Write meta JSON inside the tmp tree first.
        self._write_meta_json_to(tmp / _META_FILENAME)

        # Atomic swap: remove existing out, then move tmp into place.
        if out.exists():
            shutil.rmtree(out)
        os.replace(tmp, out)
        logger.info("Wrote %d folds to %s", len(folds), out)

    def write_meta_json(self, output_path: Path | None = None) -> None:
        """Write ``folds_meta.json``. Defaults to ``{cache_dir}/folds/folds_meta.json``."""
        if output_path is None:
            output_path = self.cache_dir / "folds" / _META_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_meta_json_to(output_path)

    def _write_meta_json_to(self, path: Path) -> None:
        folds = self.create_folds()
        rows, _ = self._load_rows()

        all_subjects = sorted({row["subject_id"] for row in rows})
        test_set = set(folds[0].test_subjects)
        pool = sorted(set(all_subjects) - test_set)

        all_labels = self._patient_has_lesion(rows)

        def ratio(subjects: Iterable[str]) -> float:
            subs = list(subjects)
            if not subs:
                return 0.0
            return round(
                sum(all_labels.get(s, 0) for s in subs) / len(subs), 6,
            )

        sk_version = _safe_pkg_version("sklearn")
        np_version = _safe_pkg_version("numpy")

        meta: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "n_folds": self.n_folds,
            "random_state": self.seed,
            "stratify_by": "has_lesion" if self._stratified else None,
            "fixed_test": True,
            "n_patients_total": len(all_subjects),
            "n_patients_pool": len(pool),
            "n_patients_test": len(test_set),
            "global_lesion_ratio_pool": ratio(pool),
            "global_lesion_ratio_test": ratio(test_set),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sk_version,
            "numpy_version": np_version,
            "test_subjects": list(folds[0].test_subjects),
            "test_lesion_ratio": ratio(test_set),
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "train_subjects": list(f.train_subjects),
                    "val_subjects": list(f.val_subjects),
                    "n_train": f.n_train,
                    "n_val": f.n_val,
                    "train_lesion_ratio": ratio(f.train_subjects),
                    "val_lesion_ratio": ratio(f.val_subjects),
                }
                for f in folds
            ],
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=False)

    # ----------------------------------------------------------- round-trip

    @classmethod
    def from_meta_json(cls, meta_path: Path | str) -> "KFoldManager":
        """Reconstruct a manager from a previously written ``folds_meta.json``.

        The returned instance's ``cache_dir`` is inferred as the parent of
        the parent of ``meta_path`` (i.e. ``folds/folds_meta.json`` lives at
        ``{cache_dir}/folds/folds_meta.json``). Fold assignments are pinned
        to the serialized JSON — ``create_folds`` returns them without
        re-running :class:`StratifiedKFold`.
        """
        meta_path = Path(meta_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)

        cache_dir = meta_path.parent.parent
        manager = cls(
            cache_dir=cache_dir,
            n_folds=int(meta["n_folds"]),
            seed=int(meta["random_state"]),
        )
        test_subjects = tuple(meta["test_subjects"])
        manager._folds = [
            FoldAssignment(
                fold_id=int(fold["fold_id"]),
                train_subjects=tuple(fold["train_subjects"]),
                val_subjects=tuple(fold["val_subjects"]),
                test_subjects=test_subjects,
            )
            for fold in meta["folds"]
        ]
        return manager


# ---------------------------------------------------------------- helpers


def _safe_pkg_version(pkg: str) -> str | None:
    try:
        mod = __import__(pkg)
        return str(getattr(mod, "__version__", "unknown"))
    except ImportError:
        return None


# ---------------------------------------------------------------- CLI


def _existing_meta_matches(meta_path: Path, n_folds: int, seed: int) -> bool:
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(meta.get("n_folds", -1)) == int(n_folds)
        and int(meta.get("random_state", -1)) == int(seed)
    )


def main() -> int:
    """Argparse entry point for ``slimdiff-kfold``."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute a patient-level stratified K-fold split over a SLIM-Diff "
            "slice cache. The source test.csv subjects are held fixed across "
            "all folds; only the train+val union is k-folded."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  slimdiff-kfold \\\n"
            "      --cache-dir /path/to/slice_cache \\\n"
            "      --n-folds 3 --seed 42\n"
        ),
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Target directory (default: {cache-dir}/folds).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing fold tree.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logger(level=getattr(logging, args.log_level))

    cache_dir = args.cache_dir
    if not cache_dir.is_dir():
        logger.error("cache-dir does not exist or is not a directory: %s", cache_dir)
        return 1
    for name in ("train.csv", "val.csv", "test.csv"):
        if not (cache_dir / name).exists():
            logger.error("Missing required source CSV: %s", cache_dir / name)
            return 1

    output_dir = args.output_dir if args.output_dir is not None else (
        cache_dir / "folds"
    )
    meta_path = output_dir / _META_FILENAME

    # Idempotency check.
    if meta_path.exists():
        if _existing_meta_matches(meta_path, args.n_folds, args.seed):
            if not args.force:
                logger.info(
                    "Fold tree already exists at %s with matching parameters; "
                    "skipping. Pass --force to rewrite.", output_dir,
                )
                return 0
        else:
            if not args.force:
                logger.error(
                    "Existing %s has different n_folds or random_state than "
                    "requested. Pass --force to overwrite.", meta_path,
                )
                return 3

    try:
        manager = KFoldManager(
            cache_dir=cache_dir, n_folds=args.n_folds, seed=args.seed,
        )
        manager.create_folds()
    except ValueError as exc:
        msg = str(exc)
        logger.error("%s", msg)
        if "infeasible" in msg:
            return 2
        return 1
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    manager.write_fold_csvs(output_dir=output_dir)
    logger.info("K-fold split written: %d folds at %s", args.n_folds, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
