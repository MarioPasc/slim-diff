"""Fold-aware data loading for the ICIP 2026 camera-ready evaluation.

Consumes the outputs produced by:

* ``src/diffusion/data/kfold.py`` (TASK-02): per-fold
  ``{cache_dir}/folds/fold_K/test.csv`` and ``{cache_dir}/folds/folds_meta.json``.
* ``src/diffusion/training/runners/generate_replicas.py`` (TASK-03): per-cell
  ``{results_root}/slimdiff_cr_{arch}_fold_{k}/replicas/replica_{r:03d}.npz``.

Single public surface: :class:`FoldEvalData` + :func:`load_fold_eval_data`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.diffusion.data.kfold import KFoldManager

logger = logging.getLogger(__name__)


# The default slots in __init__'s cell_dir_template — see run_generate_replicas
# and slurm/camera_ready/*/train_generate.sh for the actual on-disk naming.
DEFAULT_CELL_DIR_TEMPLATE: str = (
    "{results_root}/slimdiff_cr_{architecture}_fold_{fold}"
)


class FoldLoaderError(RuntimeError):
    """Raised when fold-aware loading cannot proceed (missing files etc)."""


@dataclass
class FoldEvalData:
    """Container for one ``(fold, architecture)`` evaluation cell.

    Images are kept in their on-disk precision (float16 for synthetic, float32
    for real) to keep peak RSS bounded at ~9 GB even with 180 k synthetic
    samples. Metric computers that require float32 use the ``*_f32`` helpers,
    which allocate a transient copy per call.

    Attributes
    ----------
    fold : int
        0-indexed fold identifier.
    architecture : str
        ``"shared"`` or ``"decoupled"``.
    real_images, real_masks : NDArray[float32]
        ``(N_real, H, W)`` in ``[-1, 1]``. Real masks are ``{-1, +1}`` binary.
    real_zbins : NDArray[int32]
        Z-bin per real slice.
    synth_images, synth_masks : NDArray[float16]
        ``(N_synth, H, W)`` in ``[-1, 1]``. Synthetic masks are CONTINUOUS
        (not thresholded); MMD-MF's feature extractor binarises internally at
        ``mask > 0`` (equivalent to ``tau = 0``).
    synth_zbins, synth_domains : NDArray[int32]
        Generation conditioning metadata per synthetic sample.
    n_replicas : int
        Number of replicas merged into the synthetic arrays.
    """

    fold: int
    architecture: str
    real_images: NDArray[np.float32]
    real_masks: NDArray[np.float32]
    real_zbins: NDArray[np.int32]
    synth_images: NDArray[np.float16]
    synth_masks: NDArray[np.float16]
    synth_zbins: NDArray[np.int32]
    synth_domains: NDArray[np.int32] = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    n_replicas: int = 0

    @property
    def n_real(self) -> int:
        return int(self.real_images.shape[0])

    @property
    def n_synth(self) -> int:
        return int(self.synth_images.shape[0])

    def synth_images_f32(self) -> NDArray[np.float32]:
        """Transient float32 copy of the synthetic images for metric calls."""
        return self.synth_images.astype(np.float32, copy=True)

    def synth_masks_f32(self) -> NDArray[np.float32]:
        """Transient float32 copy of the synthetic masks for metric calls."""
        return self.synth_masks.astype(np.float32, copy=True)


# --------------------------------------------------------------------------- real


def _resolve_filepath(
    raw: str,
    fold_dir: Path,
    cache_dir: Path,
) -> Path:
    """Resolve a fold-CSV ``filepath`` value to an absolute path.

    The slice-cache builder writes ``filepath`` as ``"slices/<name>.npz"``
    relative to the ORIGINAL ``cache_dir``. The K-fold manager inherits the
    column verbatim when writing ``folds/fold_K/test.csv``, so resolving
    relative to ``fold_dir`` (e.g. ``cache_dir/folds/fold_K``) would miss the
    NPZs by two directory levels. This helper tries the cache-root resolution
    first, falls back to the fold-dir resolution (useful for synthetic test
    fixtures that stage slices directly under the fold dir), then raises.
    """
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    candidates: list[Path] = [cache_dir / p, fold_dir / p]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Slice NPZ not found; tried {candidates} for raw filepath {raw!r}"
    )


def _load_real_from_fold_csv(
    csv_path: Path,
    cache_dir: Path,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32]]:
    """Load per-fold real test slices.

    The CSV schema is the one written by
    ``src/diffusion/data/caching/base.py::write_index_csv``:
    columns ``filepath, subject_id, z_index, z_bin, pathology_class, token,
    source, split, has_lesion, lesion_area_px``. ``filepath`` is relative to
    the original ``cache_dir`` (the fold CSV inherits the column verbatim);
    each ``.npz`` holds both ``"image"`` and ``"mask"`` arrays.

    Parameters
    ----------
    csv_path : Path
        Absolute path to ``{cache_dir}/folds/fold_K/test.csv``.
    cache_dir : Path
        Original slice-cache root (used to resolve ``row["filepath"]``).

    Returns
    -------
    images : NDArray[float32], shape (N, H, W)
    masks  : NDArray[float32], shape (N, H, W)
    zbins  : NDArray[int32],   shape (N,)

    Raises
    ------
    FoldLoaderError
        If ``csv_path`` does not exist or yields no valid slice rows.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FoldLoaderError(f"Per-fold test CSV not found: {csv_path}")

    fold_dir = csv_path.parent
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise FoldLoaderError(f"Per-fold test CSV is empty: {csv_path}")

    images_list: list[NDArray[np.float32]] = []
    masks_list: list[NDArray[np.float32]] = []
    zbins_list: list[int] = []

    for _, row in df.iterrows():
        try:
            npz_path = _resolve_filepath(row["filepath"], fold_dir, cache_dir)
        except FileNotFoundError as exc:
            logger.warning(str(exc))
            continue

        data = np.load(npz_path)
        images_list.append(np.asarray(data["image"], dtype=np.float32))
        masks_list.append(np.asarray(data["mask"], dtype=np.float32))
        zbins_list.append(int(row["z_bin"]))

    if not images_list:
        raise FoldLoaderError(
            f"No real slices could be loaded from {csv_path}; check filepath column."
        )

    images = np.stack(images_list, axis=0)
    masks = np.stack(masks_list, axis=0)
    zbins = np.asarray(zbins_list, dtype=np.int32)
    return images, masks, zbins


# ------------------------------------------------------------------------ synth


def _load_replicas(
    replicas_dir: Path,
    max_replicas: int | None = None,
) -> tuple[
    NDArray[np.float16],
    NDArray[np.float16],
    NDArray[np.int32],
    NDArray[np.int32],
    int,
]:
    """Load and concatenate all (or the first ``max_replicas``) replica NPZs.

    The replica NPZ schema is produced by
    ``src/diffusion/training/runners/generate_replicas.py:528``:
    keys ``images``, ``masks``, ``zbin`` (singular), ``domain`` (singular), and
    other metadata. Images and masks are ``(N, H, W)`` float16 in ``[-1, 1]``;
    masks are continuous (unthresholded).

    Returns
    -------
    synth_images, synth_masks : NDArray[float16], shape (N_total, H, W)
    synth_zbins, synth_domains : NDArray[int32],  shape (N_total,)
    n_replicas : int
    """
    replicas_dir = Path(replicas_dir)
    if not replicas_dir.exists():
        raise FoldLoaderError(f"Replicas directory not found: {replicas_dir}")

    files = sorted(replicas_dir.glob("replica_*.npz"))
    if not files:
        raise FoldLoaderError(f"No replica_*.npz files in {replicas_dir}")
    if max_replicas is not None:
        files = files[: max(int(max_replicas), 0)]
    if not files:
        raise FoldLoaderError(
            f"max_replicas={max_replicas} leaves zero replicas under {replicas_dir}"
        )

    images_parts: list[NDArray[np.float16]] = []
    masks_parts: list[NDArray[np.float16]] = []
    zbin_parts: list[NDArray[np.int32]] = []
    domain_parts: list[NDArray[np.int32]] = []

    for path in files:
        data = np.load(path)
        images_parts.append(np.asarray(data["images"], dtype=np.float16))
        masks_parts.append(np.asarray(data["masks"], dtype=np.float16))
        zbin_parts.append(np.asarray(data["zbin"], dtype=np.int32))
        domain_parts.append(np.asarray(data["domain"], dtype=np.int32))

    synth_images = np.concatenate(images_parts, axis=0)
    synth_masks = np.concatenate(masks_parts, axis=0)
    synth_zbins = np.concatenate(zbin_parts, axis=0)
    synth_domains = np.concatenate(domain_parts, axis=0)
    return synth_images, synth_masks, synth_zbins, synth_domains, len(files)


# ------------------------------------------------------------------------ public


def resolve_cell_dir(
    results_root: Path,
    architecture: str,
    fold: int,
    cell_dir_template: str | None = None,
) -> Path:
    """Format the per-cell directory from the template.

    The default template mirrors the on-disk layout produced by
    ``slurm/camera_ready/*/train_generate.sh``:
    ``{results_root}/slimdiff_cr_{architecture}_fold_{fold}``.
    """
    template = cell_dir_template or DEFAULT_CELL_DIR_TEMPLATE
    return Path(
        template.format(
            results_root=str(results_root),
            architecture=architecture,
            fold=int(fold),
        )
    )


def load_fold_eval_data(
    fold: int,
    architecture: str,
    cache_dir: Path | str,
    results_root: Path | str,
    max_replicas: int | None = None,
    cell_dir_template: str | None = None,
) -> FoldEvalData:
    """Load real test slices and merged synthetic replicas for one cell.

    Parameters
    ----------
    fold : int
        Fold identifier (0-indexed; typically 0, 1, or 2).
    architecture : str
        ``"shared"`` or ``"decoupled"``.
    cache_dir : Path
        Slice cache root; must contain ``folds/folds_meta.json`` and
        ``folds/fold_{fold}/test.csv``.
    results_root : Path
        Parent directory of the per-cell ``slimdiff_cr_*`` directories.
    max_replicas : int | None
        Cap on the number of replicas merged (useful for smoke tests).
    cell_dir_template : str | None
        Override the default ``{results_root}/slimdiff_cr_{architecture}_fold_{fold}``
        template if the on-disk layout differs.

    Returns
    -------
    FoldEvalData

    Raises
    ------
    FoldLoaderError
        If required files are missing.
    """
    cache_dir = Path(cache_dir)
    results_root = Path(results_root)

    if architecture not in ("shared", "decoupled"):
        raise ValueError(
            f"architecture must be 'shared' or 'decoupled', got {architecture!r}"
        )

    meta_path = cache_dir / "folds" / "folds_meta.json"
    if not meta_path.exists():
        raise FoldLoaderError(
            f"Fold meta JSON not found at {meta_path}. Run `slimdiff-kfold` first."
        )

    manager = KFoldManager.from_meta_json(meta_path)
    fold_dir = manager.get_cache_dir_for_fold(fold)
    fold_csv = fold_dir / "test.csv"

    real_images, real_masks, real_zbins = _load_real_from_fold_csv(fold_csv, cache_dir)

    cell_dir = resolve_cell_dir(
        results_root=results_root,
        architecture=architecture,
        fold=fold,
        cell_dir_template=cell_dir_template,
    )
    synth_images, synth_masks, synth_zbins, synth_domains, n_replicas = _load_replicas(
        cell_dir / "replicas", max_replicas=max_replicas
    )

    logger.info(
        "Loaded fold=%d architecture=%s: n_real=%d n_synth=%d n_replicas=%d",
        fold, architecture, real_images.shape[0], synth_images.shape[0], n_replicas,
    )

    return FoldEvalData(
        fold=int(fold),
        architecture=architecture,
        real_images=real_images,
        real_masks=real_masks,
        real_zbins=real_zbins,
        synth_images=synth_images,
        synth_masks=synth_masks,
        synth_zbins=synth_zbins,
        synth_domains=synth_domains,
        n_replicas=int(n_replicas),
    )
