"""Shared plotting utilities for the TASK-06 camera-ready figures.

Loads real and synthetic samples from the camera-ready layout:

* Synthetic replicas: ``{results_root}/slimdiff_cr_{arch}_fold_{k}/replicas/replica_*.npz``
  with keys ``images``, ``masks``, ``zbin``, ``lesion_present``, ``domain``
  (float16 images/masks in ``[-1, 1]``, ``domain`` = 0/1 control/epilepsy).
* Real test slices: ``{cache_dir}/folds/fold_{k}/test.csv``. The ``filepath``
  column is CSV-relative (``../../slices/...``); we resolve by joining the
  fold directory before ``cache_dir``.

The selection logic ranks each synthetic sample by MSE to the nearest real
slice in the same ``(zbin, condition)`` bucket. ``median`` picks the index at
rank ``N/2`` (representative), ``best`` picks rank 0, ``worst`` picks rank ``N-1``.
``random`` is seeded by ``numpy.random.default_rng(seed)``.

LPIPS is intentionally omitted: the MSE proxy is sufficient for selecting a
representative "middle-of-the-pack" sample for qualitative figures, and the
proxy avoids a GPU dependency on figure generation jobs.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# Type aliases & constants
# =============================================================================

SelectionMode = Literal["median", "best", "random", "worst"]

CELL_DIR_TEMPLATE: str = "slimdiff_cr_{architecture}_fold_{fold}"

# IEEE ICIP column widths (from similarity_metrics/plotting/settings.py).
IEEE_SINGLE_COL_IN: float = 3.39
IEEE_DOUBLE_COL_IN: float = 7.0
IEEE_MAX_HEIGHT_IN: float = 9.0


# =============================================================================
# Errors & data containers
# =============================================================================


class FigureUtilsError(RuntimeError):
    """Raised when figure utilities cannot locate required inputs."""


@dataclass
class SelectedSamples:
    """Container for a block of selected samples grouped by z-bin."""

    images: NDArray[np.float32]
    masks: NDArray[np.float32]
    zbins: NDArray[np.int32]
    scores: NDArray[np.float32]
    indices: NDArray[np.int64]
    source_ids: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.images.shape[0])


# =============================================================================
# Colour / value transforms
# =============================================================================


def rescale_to_display(image: NDArray) -> NDArray:
    """Map ``[-1, 1]`` to ``[0, 1]`` for matplotlib display.

    Parameters
    ----------
    image
        Any-shape float array whose nominal range is ``[-1, 1]``.

    Returns
    -------
    NDArray
        Same-shape float32 array clipped to ``[0, 1]``.
    """
    x = np.asarray(image, dtype=np.float32)
    return np.clip((x + 1.0) * 0.5, 0.0, 1.0)


def binarise_mask(mask: NDArray, tau: float = 0.0) -> NDArray:
    """Binarise a continuous mask at threshold ``tau`` (in ``[-1, 1]`` space).

    Parameters
    ----------
    mask
        Mask with values in ``[-1, 1]`` (continuous from the model) or in
        ``{-1, +1}`` (real dataset).
    tau
        Decision threshold in the input space.

    Returns
    -------
    NDArray[bool]
        Boolean lesion mask (``True`` where ``mask > tau``).
    """
    return np.asarray(mask) > tau


def overlay_mask_on_image(
    image: NDArray,
    mask: NDArray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
    tau: float = 0.0,
) -> NDArray:
    """Blend a binary mask on top of a grayscale FLAIR slice.

    Parameters
    ----------
    image
        Grayscale slice in ``[-1, 1]`` or ``[0, 1]``, shape ``(H, W)``.
    mask
        Mask in ``[-1, 1]`` (binarised at ``tau``) or a pre-binarised boolean
        array, shape ``(H, W)``.
    alpha
        Blend weight for the overlay colour.
    color
        ``(R, G, B)`` triplet in 0-255 range.
    tau
        Threshold for binarisation (ignored if ``mask.dtype == bool``).

    Returns
    -------
    NDArray[float32]
        RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.
    """
    img = np.asarray(image)
    if img.ndim == 3:
        img = img.squeeze()
    if img.min() < 0.0:
        img = rescale_to_display(img)
    rgb = np.stack([img, img, img], axis=-1).astype(np.float32)

    m = np.asarray(mask)
    if m.ndim == 3:
        m = m.squeeze()
    binary = m if m.dtype == np.bool_ else binarise_mask(m, tau=tau)

    if binary.any():
        c = np.asarray(color, dtype=np.float32) / 255.0
        for channel in range(3):
            rgb[..., channel] = np.where(
                binary,
                (1.0 - alpha) * rgb[..., channel] + alpha * c[channel],
                rgb[..., channel],
            )
    return rgb


# =============================================================================
# IEEE style helpers
# =============================================================================


def setup_ieee_style() -> None:
    """Configure matplotlib for IEEE 2-column figure output.

    Prefers the project-level ``apply_ieee_style`` when importable; falls back
    to a minimal local configuration so the figure scripts remain usable
    even if the similarity-metrics subpackage is unavailable.
    """
    try:
        from src.diffusion.scripts.similarity_metrics.plotting.settings import (
            apply_ieee_style,
        )

        apply_ieee_style()
    except Exception:  # pragma: no cover - fallback path
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "font.size": 8,
                "axes.labelsize": 8,
                "axes.titlesize": 9,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.02,
            }
        )

    # Panel-grid figures should not have a background grid.
    plt.rcParams.update({"axes.grid": False})


def add_row_label(ax: Axes, text: str, fontsize: int = 7) -> None:
    """Add a rotated row label to the left of ``ax``.

    Uses the axes y-label slot rather than a free-floating text so the label
    stays aligned when ``bbox_inches='tight'`` crops the figure.
    """
    ax.set_ylabel(
        text,
        fontsize=fontsize,
        rotation=90,
        ha="center",
        va="center",
        labelpad=4,
    )


def strip_axis(ax: Axes) -> None:
    """Remove ticks, labels and spines for an image-only panel."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# =============================================================================
# Path resolution
# =============================================================================


def resolve_cell_dir(results_root: Path, architecture: str, fold: int) -> Path:
    """Resolve the per-cell directory ``slimdiff_cr_{arch}_fold_{k}``.

    Supports two layouts transparently:

    * ``results_root`` directly contains the ``slimdiff_cr_*`` directories.
    * ``results_root`` has a ``runs/`` subdirectory that does (the camera-ready
      layout on the Sandisk).
    """
    cell_name = CELL_DIR_TEMPLATE.format(architecture=architecture, fold=fold)
    candidates = [results_root / cell_name, results_root / "runs" / cell_name]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise FigureUtilsError(
        f"Cell directory not found for arch={architecture} fold={fold}; "
        f"tried {[str(c) for c in candidates]}"
    )


def _resolve_fold_csv_filepath(raw: str, fold_dir: Path, cache_dir: Path) -> Path:
    """Resolve a fold-CSV ``filepath`` column value to an absolute path.

    The K-fold manager inherits ``filepath`` from the base cache CSV; on our
    layout that value is either ``../../slices/...`` (resolved against the
    fold dir) or ``slices/...`` (resolved against the cache root). Both are
    tried so the helper works for fixtures staged either way.
    """
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    for cand in (fold_dir / p, cache_dir / p):
        if cand.exists():
            return cand
    raise FigureUtilsError(f"Cannot resolve slice filepath: {raw!r} (fold_dir={fold_dir})")


# =============================================================================
# Real sample loading
# =============================================================================


def _condition_col(df: pd.DataFrame) -> str:
    for col in ("pathology_class", "domain", "condition"):
        if col in df.columns:
            return col
    raise FigureUtilsError(
        f"Fold CSV is missing a condition column (expected one of "
        f"pathology_class/domain/condition); got {df.columns.tolist()}"
    )


def load_real_samples(
    cache_dir: Path,
    fold: int,
    zbins: list[int],
    condition: int,
    n_samples: int = 2,
    rng: np.random.Generator | None = None,
    tau: float = 0.0,
    require_lesion: bool | None = None,
) -> SelectedSamples:
    """Load ``n_samples`` real slices per z-bin for a given condition.

    Parameters
    ----------
    cache_dir
        Root of the slice cache (``.../slice_cache``).
    fold
        Fold index.
    zbins
        Requested z-bins (samples are drawn from each independently).
    condition
        ``pathology_class`` value (0 = control, 1 = epilepsy).
    n_samples
        Samples per z-bin.
    rng
        Random generator used only when more than ``n_samples`` candidates
        exist per bucket and we need to subsample deterministically.
    tau
        Mask binarisation threshold (used to shortlist lesion-present
        slices when ``require_lesion=True``).
    require_lesion
        ``None``: no filter. ``True``: only slices with lesion pixels
        (``has_lesion=True`` or binarised mask area > 0). ``False``: only
        slices without any lesion. When ``condition == 1`` (epilepsy) and the
        caller does not override, we default to ``True`` so epilepsy rows
        always exhibit a lesion.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    fold_dir = cache_dir / "folds" / f"fold_{fold}"
    csv_path = fold_dir / "test.csv"
    if not csv_path.exists():
        # Some fixtures stage CSVs directly under cache_dir (no folds/ layer).
        alt = cache_dir / f"fold_{fold}" / "test.csv"
        if alt.exists():
            csv_path = alt
            fold_dir = alt.parent
        else:
            raise FigureUtilsError(f"Fold CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    cond_col = _condition_col(df)
    df = df[df[cond_col] == condition]

    if require_lesion is None and condition == 1 and "has_lesion" in df.columns:
        require_lesion = True
    if require_lesion is not None and "has_lesion" in df.columns:
        wanted = bool(require_lesion)
        df = df[df["has_lesion"].astype(bool) == wanted]

    images: list[NDArray] = []
    masks: list[NDArray] = []
    kept_zbins: list[int] = []
    ids: list[str] = []

    for zbin in zbins:
        candidates = df[df["z_bin"] == zbin]
        if candidates.empty:
            logger.warning(
                "No real slices for fold=%d zbin=%d condition=%d", fold, zbin, condition
            )
            continue
        n = min(n_samples, len(candidates))
        idx = rng.choice(len(candidates), size=n, replace=False)
        for r in candidates.iloc[idx].itertuples(index=False):
            fp = _resolve_fold_csv_filepath(getattr(r, "filepath"), fold_dir, cache_dir)
            arr = np.load(fp)
            images.append(arr["image"].astype(np.float32))
            masks.append(
                arr["mask"].astype(np.float32)
                if "mask" in arr.files
                else np.full_like(arr["image"], -1.0, dtype=np.float32)
            )
            kept_zbins.append(zbin)
            ids.append(str(getattr(r, "subject_id", fp.stem)))

    if not images:
        raise FigureUtilsError(
            f"load_real_samples returned 0 samples for fold={fold} condition={condition} "
            f"zbins={zbins}"
        )
    return SelectedSamples(
        images=np.stack(images),
        masks=np.stack(masks),
        zbins=np.asarray(kept_zbins, dtype=np.int32),
        scores=np.full(len(images), np.nan, dtype=np.float32),
        indices=np.arange(len(images), dtype=np.int64),
        source_ids=ids,
    )


# =============================================================================
# Synthetic sample loading & selection
# =============================================================================


_REPLICA_RE = re.compile(r"replica_(\d+)\.npz$")


def _load_replica(npz: Path) -> dict[str, NDArray]:
    data = np.load(npz, allow_pickle=False)
    return {
        "images": data["images"],
        "masks": data["masks"],
        "zbin": data["zbin"],
        "lesion_present": data["lesion_present"],
        "domain": data["domain"],
    }


def load_replicas_concat(
    run_dir: Path,
    max_replicas: int | None = None,
) -> dict[str, NDArray]:
    """Concatenate all replicas under ``run_dir/replicas`` into one dict.

    The concatenation preserves the per-replica source index in an extra
    ``replica_idx`` field so callers can annotate provenance.
    """
    rep_dir = run_dir / "replicas"
    files = sorted(rep_dir.glob("replica_*.npz"))
    if not files:
        raise FigureUtilsError(f"No replica_*.npz found under {rep_dir}")
    if max_replicas is not None:
        files = files[:max_replicas]

    images: list[NDArray] = []
    masks: list[NDArray] = []
    zbins: list[NDArray] = []
    lesions: list[NDArray] = []
    domains: list[NDArray] = []
    rep_ids: list[NDArray] = []
    for f in files:
        m = _REPLICA_RE.search(f.name)
        rid = int(m.group(1)) if m else -1
        d = _load_replica(f)
        n = d["images"].shape[0]
        images.append(d["images"])
        masks.append(d["masks"])
        zbins.append(d["zbin"])
        lesions.append(d["lesion_present"])
        domains.append(d["domain"])
        rep_ids.append(np.full(n, rid, dtype=np.int32))
    return {
        "images": np.concatenate(images, axis=0),
        "masks": np.concatenate(masks, axis=0),
        "zbin": np.concatenate(zbins, axis=0).astype(np.int32),
        "lesion_present": np.concatenate(lesions, axis=0).astype(np.int32),
        "domain": np.concatenate(domains, axis=0).astype(np.int32),
        "replica_idx": np.concatenate(rep_ids, axis=0),
    }


def _mse_to_nearest(
    synth: NDArray,
    reals: NDArray,
) -> NDArray:
    """Per-synthetic MSE to its nearest real reference.

    ``synth``: ``(N, H, W)``, ``reals``: ``(M, H, W)``. Computation is done
    in float32 with an inner loop over synth samples to keep peak memory at
    ``O(M * H * W)`` rather than ``O(N * M * H * W)``.
    """
    s = synth.astype(np.float32, copy=False).reshape(synth.shape[0], -1)
    r = reals.astype(np.float32, copy=False).reshape(reals.shape[0], -1)
    out = np.empty(s.shape[0], dtype=np.float32)
    # ||s-r||^2 = ||s||^2 + ||r||^2 - 2 s·r, vectorised per batch.
    r_sq = (r * r).sum(axis=1)  # (M,)
    for i in range(s.shape[0]):
        diff = r_sq + (s[i] * s[i]).sum() - 2.0 * r @ s[i]
        out[i] = float(diff.min()) / float(s.shape[1])
    return out


def _pick_indices(
    scores: NDArray,
    n: int,
    mode: SelectionMode,
    rng: np.random.Generator,
) -> NDArray:
    """Return ``n`` indices into ``scores`` according to ``mode``.

    For ``median`` we pick the ``n`` samples closest to the global median
    score, which matches the spec's "median-ranked" selection while still
    returning distinct samples.
    """
    n = min(n, scores.shape[0])
    order = np.argsort(scores, kind="stable")
    if mode == "best":
        return order[:n]
    if mode == "worst":
        return order[-n:][::-1]
    if mode == "random":
        return np.sort(rng.choice(scores.shape[0], size=n, replace=False))
    if mode == "median":
        mid = order.shape[0] // 2
        lo = max(0, mid - n // 2)
        hi = lo + n
        if hi > order.shape[0]:
            hi = order.shape[0]
            lo = hi - n
        return order[lo:hi]
    raise FigureUtilsError(f"Unknown selection mode: {mode}")


def load_synthetic_samples(
    results_root: Path,
    fold: int,
    architecture: str,
    zbins: list[int],
    condition: int,
    n_samples: int = 2,
    selection_mode: SelectionMode = "median",
    reference_images: NDArray | None = None,
    cache_dir: Path | None = None,
    rng: np.random.Generator | None = None,
    max_replicas: int | None = None,
    replicas: dict[str, NDArray] | None = None,
) -> SelectedSamples:
    """Select synthetic samples per z-bin according to ``selection_mode``.

    The selection ranks candidates by MSE to the nearest real slice in the
    ``(fold, zbin, condition)`` bucket. If ``reference_images`` is provided,
    it is used verbatim (one reference set applied across z-bins); otherwise
    the real slices for each z-bin are loaded from ``cache_dir``.

    Parameters
    ----------
    results_root
        Directory containing the ``slimdiff_cr_*`` subdirs (either directly
        or via a ``runs/`` layer).
    fold, architecture
        Cell identifiers.
    zbins, condition, n_samples
        Per-bucket request.
    selection_mode
        ``median`` (default), ``best``, ``random`` or ``worst``.
    reference_images
        Optional ``(R, H, W)`` reference real slices; pooled across z-bins.
    cache_dir
        Required if ``reference_images`` is ``None`` (for real-sample lookup).
    rng
        Deterministic generator for ``random`` mode.
    max_replicas
        Limit the number of loaded replicas (useful for tests).
    replicas
        Pre-loaded replica dict (skips ``load_replicas_concat`` when
        provided). Useful to avoid re-reading 24 × 9000-sample ``.npz`` files.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if replicas is None:
        cell_dir = resolve_cell_dir(results_root, architecture, fold)
        replicas = load_replicas_concat(cell_dir, max_replicas=max_replicas)

    out_imgs: list[NDArray] = []
    out_masks: list[NDArray] = []
    out_zbins: list[int] = []
    out_scores: list[float] = []
    out_indices: list[int] = []
    out_ids: list[str] = []

    for zbin in zbins:
        bucket = np.where(
            (replicas["zbin"] == zbin) & (replicas["domain"] == condition)
        )[0]
        if bucket.size == 0:
            logger.warning(
                "No synthetic samples for arch=%s fold=%d zbin=%d condition=%d",
                architecture,
                fold,
                zbin,
                condition,
            )
            continue
        cand_imgs = replicas["images"][bucket]

        if reference_images is not None:
            ref = reference_images
        elif cache_dir is not None:
            ref_block = load_real_samples(
                cache_dir=cache_dir,
                fold=fold,
                zbins=[zbin],
                condition=condition,
                n_samples=max(8, n_samples * 4),
                rng=rng,
                require_lesion=None,
            )
            ref = ref_block.images
        else:
            # Fall back to using the bucket's own centroid as a pseudo reference.
            ref = cand_imgs.astype(np.float32).mean(axis=0, keepdims=True)

        if selection_mode in ("random",):
            scores = np.zeros(cand_imgs.shape[0], dtype=np.float32)
        else:
            scores = _mse_to_nearest(cand_imgs, ref)
        pick_local = _pick_indices(scores, n_samples, selection_mode, rng)
        global_idx = bucket[pick_local]

        out_imgs.append(replicas["images"][global_idx].astype(np.float32))
        out_masks.append(replicas["masks"][global_idx].astype(np.float32))
        out_zbins.extend([int(zbin)] * pick_local.size)
        out_scores.extend(scores[pick_local].tolist())
        out_indices.extend(global_idx.tolist())
        for gi in global_idx:
            out_ids.append(f"rep{int(replicas['replica_idx'][gi])}_idx{int(gi)}")

    if not out_imgs:
        raise FigureUtilsError(
            f"load_synthetic_samples returned 0 samples for arch={architecture} "
            f"fold={fold} condition={condition} zbins={zbins}"
        )
    return SelectedSamples(
        images=np.concatenate(out_imgs, axis=0),
        masks=np.concatenate(out_masks, axis=0),
        zbins=np.asarray(out_zbins, dtype=np.int32),
        scores=np.asarray(out_scores, dtype=np.float32),
        indices=np.asarray(out_indices, dtype=np.int64),
        source_ids=out_ids,
    )


# =============================================================================
# Sidecar / provenance
# =============================================================================


def dump_selection_sidecar(
    output_path: Path,
    payload: dict,
) -> Path:
    """Write a ``{output}.json`` sidecar listing every selected sample.

    Used by all three figure scripts so selection can be audited in a
    reviewer-facing artefact.
    """
    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)

    def _jsonify(obj):
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        return obj

    sidecar.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True))
    return sidecar


def selected_to_dict(sel: SelectedSamples) -> dict:
    """Serialisable summary of a :class:`SelectedSamples` block."""
    return {
        "n": len(sel),
        "zbins": sel.zbins.tolist(),
        "indices": sel.indices.tolist(),
        "scores": sel.scores.tolist(),
        "source_ids": list(sel.source_ids),
    }
