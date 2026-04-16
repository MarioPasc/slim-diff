"""Shared vs Decoupled bottleneck preliminary analysis for ICIP 2026 camera-ready.

Compares training convergence, image quality (KID), mask morphology (MMD-MF),
and lesion-level statistics across the 3-fold cross-validation grid.
Handles any number of available replicas (>= 1).

Usage:
    conda run -n slimdiff python scripts/camera_ready_analysis.py
    conda run -n slimdiff python scripts/camera_ready_analysis.py --skip-kid  # CPU-only, fast
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from scipy.ndimage import label as scipy_label
from scipy.stats import mannwhitneyu, wilcoxon

# ---------------------------------------------------------------------------
# Paths  (edit these if your layout differs)
# ---------------------------------------------------------------------------
RESULTS_ROOT = Path(
    "/media/mpascual/Sandisk2TB/completed/jsddpm/results/epilepsy/icip2026/camera_ready/runs"
)
DATA_ROOT = Path(
    "/media/mpascual/Sandisk2TB/completed/jsddpm/data/epilepsy/slice_cache"
)
OUTPUT_ROOT = Path(
    "/media/mpascual/Sandisk2TB/completed/jsddpm/results/epilepsy/icip2026/camera_ready/analysis"
)

ARCHITECTURES = ["shared", "decoupled"]
FOLDS = [0, 1, 2]
RUN_TEMPLATE = "slimdiff_cr_{arch}_fold_{fold}"

# Mask binarisation threshold (model outputs continuous [-1,1])
TAU = 0.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RealData:
    """Container for real test-set slices for one fold."""
    images: NDArray[np.float32]   # (N, 160, 160)
    masks: NDArray[np.float32]    # (N, 160, 160)
    zbins: NDArray[np.int32]
    has_lesion: NDArray[np.bool_]
    lesion_area_px: NDArray[np.int32]


@dataclass
class SynthData:
    """Container for merged synthetic replicas for one cell."""
    images: NDArray[np.float32]
    masks: NDArray[np.float32]
    zbins: NDArray[np.int32]
    lesion_present: NDArray[np.uint8]
    n_replicas: int


def load_real_test_data(fold: int) -> RealData:
    """Load real test slices for a fold from the slice cache."""
    csv_path = DATA_ROOT / "folds" / f"fold_{fold}" / "test.csv"
    df = pd.read_csv(csv_path)
    # filepaths are relative to the CSV directory (e.g. ../../slices/...)
    csv_dir = csv_path.parent

    images, masks, zbins = [], [], []
    has_lesion, lesion_area = [], []

    for _, row in df.iterrows():
        npz = np.load((csv_dir / row["filepath"]).resolve())
        images.append(npz["image"])
        masks.append(npz["mask"])
        zbins.append(row["z_bin"])
        has_lesion.append(bool(row["has_lesion"]))
        lesion_area.append(int(row["lesion_area_px"]))

    return RealData(
        images=np.stack(images).astype(np.float32),
        masks=np.stack(masks).astype(np.float32),
        zbins=np.array(zbins, dtype=np.int32),
        has_lesion=np.array(has_lesion, dtype=np.bool_),
        lesion_area_px=np.array(lesion_area, dtype=np.int32),
    )


def load_synth_data(arch: str, fold: int) -> SynthData:
    """Load and merge all available replicas for one (arch, fold) cell."""
    run_name = RUN_TEMPLATE.format(arch=arch, fold=fold)
    replicas_dir = RESULTS_ROOT / run_name / "replicas"

    npz_files = sorted(replicas_dir.glob("replica_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No replicas in {replicas_dir}")

    all_images, all_masks, all_zbins, all_lesion = [], [], [], []
    for f in npz_files:
        d = np.load(f)
        all_images.append(d["images"].astype(np.float32))
        all_masks.append(d["masks"].astype(np.float32))
        all_zbins.append(d["zbin"].astype(np.int32))
        all_lesion.append(d["lesion_present"].astype(np.uint8))

    n_replicas = len(npz_files)
    log.info("  %s fold_%d: loaded %d replica(s), %d samples each",
             arch, fold, n_replicas, all_images[0].shape[0])

    return SynthData(
        images=np.concatenate(all_images, axis=0),
        masks=np.concatenate(all_masks, axis=0),
        zbins=np.concatenate(all_zbins, axis=0),
        lesion_present=np.concatenate(all_lesion, axis=0),
        n_replicas=n_replicas,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Training convergence
# ═══════════════════════════════════════════════════════════════════════════

def load_training_curves() -> pd.DataFrame:
    """Load epoch-level val metrics from all 6 cells."""
    rows = []
    for arch in ARCHITECTURES:
        for fold in FOLDS:
            run_name = RUN_TEMPLATE.format(arch=arch, fold=fold)
            csv_path = RESULTS_ROOT / run_name / "csv_logs" / "performance.csv"
            df = pd.read_csv(csv_path)
            # Keep only epoch-level rows (those with val/loss)
            edf = df.dropna(subset=["val/loss"]).copy()
            edf["arch"] = arch
            edf["fold"] = fold
            rows.append(edf)
    return pd.concat(rows, ignore_index=True)


def plot_convergence(curves: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Plot val loss curves and return convergence summary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "val/loss": "Total validation loss",
        "val/loss_image": "Image validation loss",
        "val/loss_mask": "Mask validation loss",
    }

    arch_colors = {"shared": "#004488", "decoupled": "#BB5566"}
    fold_styles = {0: "-", 1: "--", 2: ":"}

    for col, title in metrics.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for arch in ARCHITECTURES:
            for fold in FOLDS:
                sub = curves[(curves["arch"] == arch) & (curves["fold"] == fold)]
                ax.plot(
                    sub["epoch"], sub[col],
                    color=arch_colors[arch],
                    linestyle=fold_styles[fold],
                    linewidth=1.0,
                    alpha=0.85,
                    label=f"{arch} fold {fold}",
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(col.replace("val/", ""))
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = col.replace("/", "_")
        fig.savefig(out_dir / f"{fname}.png", dpi=200)
        fig.savefig(out_dir / f"{fname}.pdf")
        plt.close(fig)

    # Mean curves (average over folds) with std band
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, title) in zip(axes, metrics.items()):
        for arch in ARCHITECTURES:
            arch_data = curves[curves["arch"] == arch]
            grouped = arch_data.groupby("epoch")[col]
            mean = grouped.mean()
            std = grouped.std()
            ax.plot(mean.index, mean.values, color=arch_colors[arch],
                    linewidth=1.5, label=arch)
            ax.fill_between(mean.index, (mean - std).values, (mean + std).values,
                            color=arch_colors[arch], alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(col.split("/")[-1])
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Mean validation loss across folds (shaded = 1 std)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "val_loss_mean_std.png", dpi=200)
    fig.savefig(out_dir / "val_loss_mean_std.pdf")
    plt.close(fig)

    # Convergence summary table
    summary_rows = []
    for arch in ARCHITECTURES:
        for fold in FOLDS:
            sub = curves[(curves["arch"] == arch) & (curves["fold"] == fold)]
            best_idx = sub["val/loss"].idxmin()
            best_row = sub.loc[best_idx]
            summary_rows.append({
                "arch": arch,
                "fold": fold,
                "best_epoch": int(best_row["epoch"]),
                "best_val_loss": best_row["val/loss"],
                "best_val_loss_image": best_row["val/loss_image"],
                "best_val_loss_mask": best_row["val/loss_mask"],
                "total_epochs": int(sub["epoch"].max()),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "convergence_summary.csv", index=False)
    log.info("Convergence summary:\n%s", summary.to_string(index=False))
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# 2. KID (image quality)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_features_chunked(
    images: NDArray[np.float32],
    model: "torch.nn.Module",
    device: str,
    batch_size: int = 32,
    chunk_size: int = 2000,
) -> "torch.Tensor":
    """Extract InceptionV3 features without materialising all preprocessed images.

    Preprocesses and forwards in chunks of `chunk_size` images so that the
    peak RAM spike is ~chunk_size * 3 * 299 * 299 bytes rather than N * ...
    """
    import torch
    import torch.nn.functional as F

    all_features: list[torch.Tensor] = []
    N = len(images)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = images[start:end]  # (C, 160, 160) float32

        # Preprocess: [-1,1] → [0,1] → (C,3,299,299) uint8
        prep = (chunk + 1.0) / 2.0
        np.clip(prep, 0.0, 1.0, out=prep)
        t = torch.from_numpy(prep).float().unsqueeze(1).repeat(1, 3, 1, 1)
        t = F.interpolate(t, size=(299, 299), mode="bilinear", align_corners=False)
        t = (t * 255).to(torch.uint8)

        # Forward in mini-batches
        with torch.no_grad():
            for i in range(0, t.shape[0], batch_size):
                batch = t[i : i + batch_size].to(device)
                feat = model(batch)
                if isinstance(feat, tuple):
                    feat = feat[0]
                if feat.dim() > 2:
                    feat = feat.view(feat.size(0), -1)
                all_features.append(feat.cpu())

        del t, prep
        torch.cuda.empty_cache()

    return torch.cat(all_features, dim=0)


def compute_kid_for_cell(
    real_images: NDArray[np.float32],
    synth_images: NDArray[np.float32],
    device: str = "cuda",
    batch_size: int = 32,
    subset_size: int = 1000,
    num_subsets: int = 100,
) -> tuple[float, float]:
    """Compute KID between real and synthetic image distributions.

    Uses a memory-efficient two-stage approach:
    1. Extract InceptionV3 features in small chunks (peak ~500 MB extra RAM).
    2. Compute polynomial-kernel MMD on CPU from cached features.

    Returns:
        (kid_mean, kid_std) across subsets.
    """
    import torch
    from torchmetrics.image.inception import InceptionScore
    from src.diffusion.scripts.similarity_metrics.metrics.kid import KIDComputer

    # Lazy-load InceptionV3 once
    if not hasattr(compute_kid_for_cell, "_model"):
        is_module = InceptionScore(normalize=True)
        compute_kid_for_cell._model = is_module.inception.eval().to(device)
    model = compute_kid_for_cell._model

    log.info("    Extracting real features (%d images)...", len(real_images))
    real_feat = _extract_features_chunked(real_images, model, device, batch_size)
    log.info("    Extracting synth features (%d images)...", len(synth_images))
    synth_feat = _extract_features_chunked(synth_images, model, device, batch_size)

    computer = KIDComputer(
        subset_size=min(subset_size, len(real_images), len(synth_images)),
        num_subsets=num_subsets,
        degree=3,
        device="cpu",
        batch_size=batch_size,
    )
    result = computer.compute_from_features(real_feat, synth_feat)
    return result.value, result.std


def _plot_kid(df: pd.DataFrame | None, out_dir: Path) -> None:
    """Plot KID results if available."""
    if df is None or len(df) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    arch_colors = {"shared": "#004488", "decoupled": "#BB5566"}
    x_offsets = {"shared": -0.15, "decoupled": 0.15}
    for arch in ARCHITECTURES:
        sub = df[df["arch"] == arch]
        x = sub["fold"].values + x_offsets[arch]
        ax.errorbar(x, sub["kid_mean"], yerr=sub["kid_std"],
                     fmt="o", color=arch_colors[arch], label=arch,
                     capsize=4, markersize=6)
    ax.set_xlabel("Fold")
    ax.set_ylabel("KID")
    ax.set_title("Kernel Inception Distance (lower = better)")
    ax.set_xticks(FOLDS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "kid_per_fold.png", dpi=200)
    fig.savefig(out_dir / "kid_per_fold.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Mask morphology (MMD-MF + per-feature Wasserstein)
# ═══════════════════════════════════════════════════════════════════════════

def _count_lesions_per_mask(masks: NDArray[np.float32], tau: float = TAU, min_px: int = 5) -> NDArray[np.int32]:
    """Count connected components per mask."""
    from scipy.ndimage import sum as ndi_sum
    counts = np.zeros(masks.shape[0], dtype=np.int32)
    for i in range(masks.shape[0]):
        binary = (masks[i] > tau).astype(np.uint8)
        labeled, n = scipy_label(binary)
        if n > 0:
            areas = ndi_sum(binary, labeled, range(1, n + 1))
            counts[i] = int(np.sum(np.array(areas) >= min_px))
    return counts


def _lesion_areas(masks: NDArray[np.float32], tau: float = TAU, min_px: int = 5) -> NDArray[np.float64]:
    """Extract all lesion areas (in pixels) from a set of masks."""
    from scipy.ndimage import sum as ndi_sum
    areas = []
    for i in range(masks.shape[0]):
        binary = (masks[i] > tau).astype(np.uint8)
        labeled, n = scipy_label(binary)
        if n > 0:
            region_areas = ndi_sum(binary, labeled, range(1, n + 1))
            for a in region_areas:
                if a >= min_px:
                    areas.append(float(a))
    return np.array(areas, dtype=np.float64)


def _plot_mmd(mmd_df: pd.DataFrame, wass_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot MMD-MF and Wasserstein heatmap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arch_colors = {"shared": "#004488", "decoupled": "#BB5566"}
    x_offsets = {"shared": -0.15, "decoupled": 0.15}

    fig, ax = plt.subplots(figsize=(5, 4))
    for arch in ARCHITECTURES:
        sub = mmd_df[mmd_df["arch"] == arch]
        x = sub["fold"].values + x_offsets[arch]
        ax.errorbar(x, sub["mmd_mf"], yerr=sub["mmd_mf_std"],
                     fmt="s", color=arch_colors[arch], label=arch,
                     capsize=4, markersize=6)
    ax.set_xlabel("Fold")
    ax.set_ylabel("MMD-MF")
    ax.set_title("Mask Morphology Distance (lower = better)")
    ax.set_xticks(FOLDS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mmd_mf_per_fold.png", dpi=200)
    fig.savefig(out_dir / "mmd_mf_per_fold.pdf")
    plt.close(fig)

    wass_no_geom = wass_df[wass_df["feature"] != "geometric_mean"]
    pivot = wass_no_geom.groupby(["arch", "feature"])["wasserstein"].mean().unstack("feature")
    fig, ax = plt.subplots(figsize=(9, 3))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Per-feature Wasserstein distance (mean over folds, lower = better)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "wasserstein_heatmap.png", dpi=200)
    fig.savefig(out_dir / "wasserstein_heatmap.pdf")
    plt.close(fig)


def _plot_lesion_stats(lesion_df: pd.DataFrame, area_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot lesion prevalence and area distributions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arch_colors = {"shared": "#004488", "decoupled": "#BB5566", "real": "#228833"}

    # Prevalence bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_width = 0.25
    x = np.arange(len(FOLDS))
    real_prevs = [lesion_df[lesion_df["fold"] == f]["real_prevalence"].iloc[0] for f in FOLDS]
    ax.bar(x - bar_width, real_prevs, bar_width, color=arch_colors["real"], label="Real")
    for i, arch in enumerate(ARCHITECTURES):
        sub = lesion_df[lesion_df["arch"] == arch]
        ax.bar(x + i * bar_width, sub["synth_prevalence_mask"].values, bar_width,
               color=arch_colors[arch], label=f"Synth ({arch})")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Lesion prevalence")
    ax.set_title("Fraction of slices containing lesions")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in FOLDS])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "lesion_prevalence.png", dpi=200)
    fig.savefig(out_dir / "lesion_prevalence.pdf")
    plt.close(fig)

    # Area distribution (log scale)
    if len(area_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for ax, fold in zip(axes, FOLDS):
            fold_data = area_df[area_df["fold"] == fold]
            for src_label, color in [("real", "#228833"), ("shared", "#004488"), ("decoupled", "#BB5566")]:
                if src_label == "real":
                    sub = fold_data[fold_data["source"] == "real"]
                else:
                    sub = fold_data[(fold_data["source"] == "synth") & (fold_data["arch"] == src_label)]
                if len(sub) > 0:
                    ax.hist(np.log10(sub["area"].values + 1), bins=40, alpha=0.5,
                            color=color, label=src_label, density=True)
            ax.set_xlabel("log10(area + 1)")
            ax.set_title(f"Fold {fold}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Density")
        fig.suptitle("Lesion area distributions", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / "lesion_area_distribution.png", dpi=200)
        fig.savefig(out_dir / "lesion_area_distribution.pdf")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Summary + statistical comparison
# ═══════════════════════════════════════════════════════════════════════════

def build_summary(
    convergence: pd.DataFrame,
    kid_df: pd.DataFrame | None,
    mmd_df: pd.DataFrame,
    lesion_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Build a summary table comparing shared vs decoupled across folds."""
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("SHARED vs DECOUPLED BOTTLENECK — PRELIMINARY ANALYSIS")
    lines.append("=" * 72)

    # Convergence
    lines.append("\n--- Convergence ---")
    for arch in ARCHITECTURES:
        sub = convergence[convergence["arch"] == arch]
        lines.append(f"  {arch:>10s}: best_epoch = {sub['best_epoch'].values}  "
                     f"val_loss = {sub['best_val_loss'].values.round(6)}")
    shared_epochs = convergence[convergence["arch"] == "shared"]["best_epoch"].values
    decoupled_epochs = convergence[convergence["arch"] == "decoupled"]["best_epoch"].values
    lines.append(f"  Mean best epoch: shared={shared_epochs.mean():.1f}, "
                 f"decoupled={decoupled_epochs.mean():.1f}")

    # KID
    if kid_df is not None:
        lines.append("\n--- KID (Image Quality, lower = better) ---")
        for arch in ARCHITECTURES:
            sub = kid_df[kid_df["arch"] == arch]
            vals = sub["kid_mean"].values
            lines.append(f"  {arch:>10s}: per-fold = {vals.round(6)}  "
                         f"mean = {vals.mean():.6f}")
        shared_kid = kid_df[kid_df["arch"] == "shared"]["kid_mean"].values
        decoupled_kid = kid_df[kid_df["arch"] == "decoupled"]["kid_mean"].values
        diff = decoupled_kid.mean() - shared_kid.mean()
        lines.append(f"  Delta (decoupled - shared): {diff:+.6f}")
        if len(shared_kid) >= 3:
            # Paired comparison across folds
            try:
                stat, p = wilcoxon(shared_kid, decoupled_kid)
                lines.append(f"  Wilcoxon signed-rank: W={stat:.3f}, p={p:.4f}")
            except ValueError:
                pass

    # MMD-MF
    lines.append("\n--- MMD-MF (Mask Morphology, lower = better) ---")
    for arch in ARCHITECTURES:
        sub = mmd_df[mmd_df["arch"] == arch]
        vals = sub["mmd_mf"].values
        lines.append(f"  {arch:>10s}: per-fold = {vals.round(6)}  "
                     f"mean = {vals.mean():.6f}")
    shared_mmd = mmd_df[mmd_df["arch"] == "shared"]["mmd_mf"].values
    decoupled_mmd = mmd_df[mmd_df["arch"] == "decoupled"]["mmd_mf"].values
    diff = decoupled_mmd.mean() - shared_mmd.mean()
    lines.append(f"  Delta (decoupled - shared): {diff:+.6f}")
    if len(shared_mmd) >= 3:
        try:
            stat, p = wilcoxon(shared_mmd, decoupled_mmd)
            lines.append(f"  Wilcoxon signed-rank: W={stat:.3f}, p={p:.4f}")
        except ValueError:
            pass

    # Lesion stats
    lines.append("\n--- Lesion Prevalence (gap to real, lower = better) ---")
    for arch in ARCHITECTURES:
        sub = lesion_df[lesion_df["arch"] == arch]
        gaps = sub["prevalence_gap"].values
        lines.append(f"  {arch:>10s}: per-fold gap = {gaps.round(4)}  "
                     f"mean = {gaps.mean():.4f}")

    lines.append("\n--- Lesion Area (median, px) ---")
    for arch in ARCHITECTURES:
        sub = lesion_df[lesion_df["arch"] == arch]
        lines.append(f"  {arch:>10s}: synth_median = {sub['synth_median_area'].values.round(1)}  "
                     f"(real_median = {sub['real_median_area'].values.round(1)})")

    # Replica count note
    lines.append("\n--- Replica Counts ---")
    for arch in ARCHITECTURES:
        for fold in FOLDS:
            key = (arch, fold)
            if kid_df is not None:
                row = kid_df[(kid_df["arch"] == arch) & (kid_df["fold"] == fold)]
                n = row["n_replicas"].values[0] if len(row) > 0 else "?"
            else:
                row = mmd_df[(mmd_df["arch"] == arch) & (mmd_df["fold"] == fold)]
                n = row["n_replicas"].values[0] if len(row) > 0 else "?"
            lines.append(f"  {arch} fold_{fold}: {n} replica(s)")

    lines.append("\n" + "=" * 72)

    report = "\n".join(lines)
    print(report)
    (out_dir / "summary_report.txt").write_text(report)
    log.info("Summary written to %s", out_dir / "summary_report.txt")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Shared vs Decoupled analysis")
    parser.add_argument("--skip-kid", action="store_true",
                        help="Skip KID computation (GPU-heavy)")
    parser.add_argument("--skip-lesion-stats", action="store_true",
                        help="Skip lesion-level statistics (slow on large replica sets)")
    parser.add_argument("--device", default="cuda",
                        help="Device for KID (default: cuda)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    t0 = time.time()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    log.info("Loading real test data (fixed test set, 16 patients)...")
    # Test set is the same across folds, but load per-fold CSVs for consistency
    real_data: dict[int, RealData] = {}
    for fold in FOLDS:
        real_data[fold] = load_real_test_data(fold)
        log.info("  fold_%d: %d test slices, %d with lesions",
                 fold, len(real_data[fold].images), real_data[fold].has_lesion.sum())

    # ------------------------------------------------------------------
    # 1. Training convergence
    # ------------------------------------------------------------------
    log.info("=== 1. Training Convergence ===")
    curves = load_training_curves()
    convergence = plot_convergence(curves, out_dir / "convergence")

    # ------------------------------------------------------------------
    # Process each cell lazily to keep RAM in check (~2 GB per cell)
    # ------------------------------------------------------------------
    kid_rows: list[dict] = []
    mmd_rows: list[dict] = []
    wass_rows: list[dict] = []
    lesion_rows: list[dict] = []
    area_rows: list[dict] = []

    for arch in ARCHITECTURES:
        for fold in FOLDS:
            log.info("--- Loading %s fold_%d ---", arch, fold)
            synth = load_synth_data(arch, fold)
            real = real_data[fold]

            # 2. KID
            if not args.skip_kid:
                import torch
                log.info("  KID: %s fold_%d ...", arch, fold)
                kid_mean, kid_std = compute_kid_for_cell(
                    real.images, synth.images,
                    device=args.device, batch_size=32,
                    subset_size=min(1000, len(real.images), len(synth.images)),
                )
                kid_rows.append({
                    "arch": arch, "fold": fold,
                    "kid_mean": kid_mean, "kid_std": kid_std,
                    "n_real": len(real.images), "n_synth": len(synth.images),
                    "n_replicas": synth.n_replicas,
                })
                log.info("    KID = %.6f +/- %.6f", kid_mean, kid_std)
                torch.cuda.empty_cache()

            # 3. MMD-MF
            log.info("  MMD-MF: %s fold_%d ...", arch, fold)
            synth_masks_bin = np.where(synth.masks > TAU, 1.0, -1.0).astype(np.float32)
            from src.diffusion.scripts.similarity_metrics.metrics.mask_morphology import (
                MaskMorphologyDistanceComputer,
            )
            computer = MaskMorphologyDistanceComputer(
                min_lesion_size_px=5, subset_size=500, num_subsets=100,
                degree=3, normalize_features=True,
            )
            result, real_feat, synth_feat = computer.compute(
                real.masks, synth_masks_bin, show_progress=True,
            )
            mmd_rows.append({
                "arch": arch, "fold": fold,
                "mmd_mf": result.value, "mmd_mf_std": result.std,
                "n_real_lesions": result.metadata.get("n_real_lesions", 0),
                "n_synth_lesions": result.metadata.get("n_synth_lesions", 0),
                "n_replicas": synth.n_replicas,
            })
            log.info("    MMD-MF = %.6f +/- %.6f", result.value, result.std)
            wass = computer.compute_per_feature_wasserstein(
                real_features=real_feat, synth_features=synth_feat,
            )
            for feat_name, dist in wass.items():
                wass_rows.append({"arch": arch, "fold": fold,
                                  "feature": feat_name, "wasserstein": dist})

            # 4. Lesion stats
            if not args.skip_lesion_stats:
                log.info("  Lesion stats: %s fold_%d ...", arch, fold)
                real_prevalence = real.has_lesion.mean()
                real_areas = _lesion_areas(real.masks)
                synth_has_lesion = np.array([
                    (synth_masks_bin[i] > 0).any() for i in range(len(synth_masks_bin))
                ])
                synth_prevalence_mask = synth_has_lesion.mean()
                synth_areas = _lesion_areas(synth_masks_bin)
                real_counts = _count_lesions_per_mask(real.masks)
                synth_counts = _count_lesions_per_mask(synth_masks_bin)
                real_mean_count = real_counts[real_counts > 0].mean() if (real_counts > 0).any() else 0.0
                synth_mean_count = synth_counts[synth_counts > 0].mean() if (synth_counts > 0).any() else 0.0
                lesion_rows.append({
                    "arch": arch, "fold": fold,
                    "real_prevalence": real_prevalence,
                    "synth_prevalence_token": synth.lesion_present.mean(),
                    "synth_prevalence_mask": synth_prevalence_mask,
                    "prevalence_gap": abs(real_prevalence - synth_prevalence_mask),
                    "real_mean_area": real_areas.mean() if len(real_areas) > 0 else 0.0,
                    "synth_mean_area": synth_areas.mean() if len(synth_areas) > 0 else 0.0,
                    "real_median_area": np.median(real_areas) if len(real_areas) > 0 else 0.0,
                    "synth_median_area": np.median(synth_areas) if len(synth_areas) > 0 else 0.0,
                    "real_n_lesions": len(real_areas),
                    "synth_n_lesions": len(synth_areas),
                    "real_mean_lesions_per_mask": real_mean_count,
                    "synth_mean_lesions_per_mask": synth_mean_count,
                    "n_replicas": synth.n_replicas,
                })
                for a in real_areas:
                    area_rows.append({"source": "real", "arch": "real", "fold": fold, "area": a})
                for a in synth_areas:
                    area_rows.append({"source": "synth", "arch": arch, "fold": fold, "area": a})

            # Free synth data for this cell
            del synth, synth_masks_bin
            import gc; gc.collect()

    # ------------------------------------------------------------------
    # Assemble DataFrames and save outputs
    # ------------------------------------------------------------------
    kid_df = pd.DataFrame(kid_rows) if kid_rows else None
    mmd_df = pd.DataFrame(mmd_rows)
    wass_df = pd.DataFrame(wass_rows)
    lesion_df = pd.DataFrame(lesion_rows) if lesion_rows else pd.DataFrame([
        {"arch": a, "fold": f, "prevalence_gap": 0.0,
         "real_median_area": 0.0, "synth_median_area": 0.0}
        for a in ARCHITECTURES for f in FOLDS
    ])

    # Save CSVs
    if kid_df is not None and len(kid_df) > 0:
        kid_dir = out_dir / "kid"; kid_dir.mkdir(parents=True, exist_ok=True)
        kid_df.to_csv(kid_dir / "kid_results.csv", index=False)
    mmd_dir = out_dir / "mask_morphology"; mmd_dir.mkdir(parents=True, exist_ok=True)
    mmd_df.to_csv(mmd_dir / "mmd_mf_results.csv", index=False)
    wass_df.to_csv(mmd_dir / "wasserstein_per_feature.csv", index=False)
    if lesion_rows:
        ls_dir = out_dir / "lesion_stats"; ls_dir.mkdir(parents=True, exist_ok=True)
        lesion_df.to_csv(ls_dir / "lesion_stats.csv", index=False)

    # Generate plots
    _plot_kid(kid_df, out_dir / "kid")
    _plot_mmd(mmd_df, wass_df, out_dir / "mask_morphology")
    if lesion_rows:
        _plot_lesion_stats(lesion_df, pd.DataFrame(area_rows), out_dir / "lesion_stats")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    log.info("=== 5. Building Summary ===")
    build_summary(convergence, kid_df, mmd_df, lesion_df, out_dir)

    elapsed = time.time() - t0
    log.info("Done in %.1f s. Outputs at: %s", elapsed, out_dir)


if __name__ == "__main__":
    main()
