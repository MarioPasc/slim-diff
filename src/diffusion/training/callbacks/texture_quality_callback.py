"""Texture quality tracking callback for JS-DDPM training.

Monitors texture fidelity metrics during training by comparing generated samples
against real validation data. Tracks:
- LBP code 8 fraction (uniform flat regions - key Component B marker)
- Wavelet HH energy ratio at Level 1 (fine texture - Component A marker)

These metrics directly address the two-component artifact structure identified
in the XAI diagnostic analysis:
- Component A: High-frequency deficit (wavelet HH ratio < 0.85)
- Component B: Low-frequency texture anomaly (LBP code 8 deficit)

Only runs on rank 0 in DDP to avoid redundant computation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig
from scipy.stats import ks_2samp

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from src.diffusion.model.factory import DiffusionSampler
from src.diffusion.utils.zbin_priors import get_anatomical_priors_as_input

logger = logging.getLogger(__name__)


def compute_lbp_histogram(
    images: np.ndarray,
    radius: int = 1,
    n_points: int = 8,
) -> np.ndarray:
    """Compute LBP histogram for a batch of images.

    Args:
        images: Batch of 2D images, shape (N, H, W).
        radius: LBP radius.
        n_points: Number of points for LBP.

    Returns:
        Normalized histogram of LBP codes, shape (n_points + 2,).
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image required for LBP computation")

    n_bins = n_points + 2  # uniform LBP has n_points + 2 bins
    hist_sum = np.zeros(n_bins, dtype=np.float64)

    for img in images:
        # Normalize to [0, 255] for LBP
        img_uint8 = ((img + 1) / 2 * 255).astype(np.uint8)
        lbp = local_binary_pattern(img_uint8, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist_sum += hist.astype(np.float64)

    # Normalize
    hist_sum = hist_sum / (hist_sum.sum() + 1e-12)
    return hist_sum


def compute_wavelet_hh_energy(
    images: np.ndarray,
    wavelet: str = "db4",
    level: int = 1,
) -> float:
    """Compute mean HH subband energy at specified level.

    Args:
        images: Batch of 2D images, shape (N, H, W).
        wavelet: Wavelet name.
        level: Decomposition level (1 = finest).

    Returns:
        Mean HH energy (mean squared coefficient).
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets required for wavelet analysis")

    hh_energies = []
    for img in images:
        coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
        # coeffs[0] = approximation at coarsest level
        # coeffs[1] = (LH, HL, HH) at level `level`
        # For level=1, coeffs[1] is the finest detail
        _, _, hh = coeffs[1]
        energy = np.mean(hh ** 2)
        hh_energies.append(energy)

    return float(np.mean(hh_energies))


class TextureQualityCallback(Callback):
    """Track texture quality metrics during training.

    Generates synthetic samples and compares texture characteristics
    against real validation data. Computes:
    - LBP code 8 fraction ratio (synth/real) - target: 1.0
    - Wavelet HH L1 energy ratio (synth/real) - target: > 0.85

    Only runs on global rank 0 in DDP to avoid duplicate computation.
    """

    def __init__(
        self,
        cfg: DictConfig,
        log_every_n_epochs: int = 25,
        n_samples: int = 200,
        lbp_radius: int = 1,
        lbp_n_points: int = 8,
        wavelet: str = "db4",
        compute_full_texture: bool = False,
    ) -> None:
        """Initialize the callback.

        Args:
            cfg: Configuration object.
            log_every_n_epochs: Frequency of texture quality evaluation.
            n_samples: Number of samples to generate for comparison.
            lbp_radius: LBP radius parameter.
            lbp_n_points: Number of LBP points.
            wavelet: Wavelet name for DWT.
            compute_full_texture: If True, compute full GLCM/texture suite.
                                 If False, only compute key markers (faster).
        """
        super().__init__()
        self.cfg = cfg
        self.log_every_n_epochs = log_every_n_epochs
        self.n_samples = n_samples
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.wavelet = wavelet
        self.compute_full_texture = compute_full_texture

        # Output directory for CSV logs
        self.output_dir = Path(cfg.experiment.output_dir) / "texture_quality"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sampler (created lazily)
        self._sampler: DiffusionSampler | None = None

        # Z-bin priors for anatomical conditioning
        self._zbin_priors = None
        self._use_anatomical_conditioning = cfg.model.get("anatomical_conditioning", False)

        # Check dependencies
        if not PYWT_AVAILABLE:
            logger.warning("PyWavelets not available - wavelet metrics disabled")
        if not SKIMAGE_AVAILABLE:
            logger.warning("scikit-image not available - LBP metrics disabled")

        logger.info(
            f"TextureQualityCallback initialized: "
            f"every_n_epochs={log_every_n_epochs}, n_samples={n_samples}, "
            f"lbp_radius={lbp_radius}, wavelet={wavelet}"
        )

    def _get_sampler(self, pl_module: pl.LightningModule) -> DiffusionSampler:
        """Get or create the diffusion sampler."""
        if self._sampler is None:
            anatomical_encoder = getattr(pl_module, "_anatomical_encoder", None)
            self._sampler = DiffusionSampler(
                model=pl_module.model,
                scheduler=pl_module.inferer,
                cfg=self.cfg,
                device=pl_module.device,
                anatomical_encoder=anatomical_encoder,
            )
        return self._sampler

    def _load_zbin_priors(self, pl_module: pl.LightningModule) -> None:
        """Load z-bin priors from the pl_module if available."""
        if self._zbin_priors is None:
            self._zbin_priors = getattr(pl_module, "_zbin_priors", None)

    def _generate_samples(
        self,
        pl_module: pl.LightningModule,
        tokens: list[int],
        z_bins: list[int],
    ) -> np.ndarray:
        """Generate synthetic samples for given tokens.

        Args:
            pl_module: Lightning module.
            tokens: List of conditioning tokens.
            z_bins: List of z-bin values (for anatomical priors).

        Returns:
            Array of generated images, shape (N, H, W).
        """
        sampler = self._get_sampler(pl_module)
        self._load_zbin_priors(pl_module)

        samples = []
        with torch.no_grad():
            for token, z_bin in zip(tokens, z_bins):
                # Get anatomical prior if needed
                anatomical_mask = None
                if self._use_anatomical_conditioning and self._zbin_priors is not None:
                    anatomical_mask = get_anatomical_priors_as_input(
                        [z_bin],
                        self._zbin_priors,
                        device=pl_module.device,
                    ).squeeze(0)

                sample = sampler.sample_single(token, anatomical_mask=anatomical_mask)
                # Extract image channel (channel 0)
                samples.append(sample[0].cpu().numpy())

        return np.stack(samples, axis=0)

    def _collect_real_samples(
        self,
        trainer: pl.Trainer,
        n_samples: int,
    ) -> tuple[np.ndarray, list[int], list[int]]:
        """Collect real samples from validation loader.

        Args:
            trainer: Lightning trainer.
            n_samples: Number of samples to collect.

        Returns:
            Tuple of (images, tokens, z_bins).
        """
        val_loader = trainer.val_dataloaders
        if val_loader is None:
            raise RuntimeError("No validation dataloader available")

        images = []
        tokens = []
        z_bins = []

        for batch in val_loader:
            batch_images = batch["image"].cpu().numpy()
            batch_tokens = batch["token"].cpu().numpy().tolist()
            batch_zbins = batch["metadata"]["z_bin"]
            if isinstance(batch_zbins, torch.Tensor):
                batch_zbins = batch_zbins.cpu().numpy().tolist()

            for i in range(len(batch_images)):
                images.append(batch_images[i, 0])  # Channel 0
                tokens.append(batch_tokens[i])
                z_bins.append(batch_zbins[i])

                if len(images) >= n_samples:
                    break

            if len(images) >= n_samples:
                break

        return np.stack(images[:n_samples], axis=0), tokens[:n_samples], z_bins[:n_samples]

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Compute and log texture quality metrics.

        Only runs on global rank 0 in DDP.

        Args:
            trainer: Lightning trainer.
            pl_module: Lightning module.
        """
        current_epoch = trainer.current_epoch

        # Check frequency
        if current_epoch % self.log_every_n_epochs != 0:
            return

        # Only run on rank 0 in DDP
        if not trainer.is_global_zero:
            logger.debug(f"Skipping texture quality on rank {trainer.global_rank}")
            return

        logger.info(f"Computing texture quality metrics at epoch {current_epoch}")

        pl_module.eval()

        try:
            # Collect real samples
            real_images, tokens, z_bins = self._collect_real_samples(
                trainer, self.n_samples
            )

            # Generate synthetic samples with matching tokens
            synth_images = self._generate_samples(pl_module, tokens, z_bins)

            metrics = {"epoch": current_epoch}

            # Compute LBP metrics
            if SKIMAGE_AVAILABLE:
                real_lbp_hist = compute_lbp_histogram(
                    real_images, self.lbp_radius, self.lbp_n_points
                )
                synth_lbp_hist = compute_lbp_histogram(
                    synth_images, self.lbp_radius, self.lbp_n_points
                )

                # LBP code 8 is the key marker for Component B
                # Code 8 represents uniform flat regions
                lbp_code8_real = real_lbp_hist[8] if len(real_lbp_hist) > 8 else 0
                lbp_code8_synth = synth_lbp_hist[8] if len(synth_lbp_hist) > 8 else 0
                lbp_code8_ratio = lbp_code8_synth / (lbp_code8_real + 1e-12)

                # KS test on full histograms
                lbp_ks_stat, lbp_ks_pval = ks_2samp(real_lbp_hist, synth_lbp_hist)

                metrics.update({
                    "texture/lbp_code8_real": lbp_code8_real,
                    "texture/lbp_code8_synth": lbp_code8_synth,
                    "texture/lbp_code8_ratio": lbp_code8_ratio,
                    "texture/lbp_ks_statistic": lbp_ks_stat,
                })

                logger.info(
                    f"  LBP code 8: real={lbp_code8_real:.4f}, synth={lbp_code8_synth:.4f}, "
                    f"ratio={lbp_code8_ratio:.4f} (target: 1.0)"
                )

            # Compute wavelet metrics
            if PYWT_AVAILABLE:
                real_hh_energy = compute_wavelet_hh_energy(
                    real_images, self.wavelet, level=1
                )
                synth_hh_energy = compute_wavelet_hh_energy(
                    synth_images, self.wavelet, level=1
                )

                hh_ratio = synth_hh_energy / (real_hh_energy + 1e-12)

                metrics.update({
                    "texture/wavelet_hh_real": real_hh_energy,
                    "texture/wavelet_hh_synth": synth_hh_energy,
                    "texture/wavelet_hh_ratio": hh_ratio,
                })

                logger.info(
                    f"  Wavelet HH L1: real={real_hh_energy:.6f}, synth={synth_hh_energy:.6f}, "
                    f"ratio={hh_ratio:.4f} (target: >0.85)"
                )

            # Compute full texture suite if requested
            if self.compute_full_texture:
                self._compute_full_texture_metrics(
                    real_images, synth_images, metrics
                )

            # Log to trainer
            for key, value in metrics.items():
                if key != "epoch":
                    pl_module.log(key, value, sync_dist=False, rank_zero_only=True)

            # Log to wandb if available
            if hasattr(trainer.logger, "experiment"):
                import wandb
                trainer.logger.experiment.log(metrics)

            # Save to CSV
            self._save_metrics_csv(metrics)

        except Exception as e:
            logger.error(f"Error computing texture quality metrics: {e}")
            import traceback
            traceback.print_exc()

    def _compute_full_texture_metrics(
        self,
        real_images: np.ndarray,
        synth_images: np.ndarray,
        metrics: dict[str, Any],
    ) -> None:
        """Compute additional texture metrics (GLCM, multi-level wavelet).

        Args:
            real_images: Real image batch.
            synth_images: Synthetic image batch.
            metrics: Dictionary to update with new metrics.
        """
        # Multi-level wavelet analysis
        if PYWT_AVAILABLE:
            for level in [2, 3, 4]:
                real_hh = compute_wavelet_hh_energy(real_images, self.wavelet, level)
                synth_hh = compute_wavelet_hh_energy(synth_images, self.wavelet, level)
                ratio = synth_hh / (real_hh + 1e-12)
                metrics[f"texture/wavelet_hh_L{level}_ratio"] = ratio

        # Additional LBP radii
        if SKIMAGE_AVAILABLE:
            for radius in [2, 3]:
                n_points = 8 * radius
                real_hist = compute_lbp_histogram(real_images, radius, n_points)
                synth_hist = compute_lbp_histogram(synth_images, radius, n_points)
                ks_stat, _ = ks_2samp(real_hist, synth_hist)
                metrics[f"texture/lbp_r{radius}_ks"] = ks_stat

    def _save_metrics_csv(self, metrics: dict[str, Any]) -> None:
        """Save metrics to CSV file."""
        import csv

        csv_path = self.output_dir / "texture_quality_metrics.csv"
        file_exists = csv_path.exists()

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(metrics.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

        logger.debug(f"Saved texture metrics to {csv_path}")


def build_texture_quality_callback(cfg: DictConfig) -> TextureQualityCallback | None:
    """Build texture quality callback from config.

    Args:
        cfg: Configuration object.

    Returns:
        TextureQualityCallback if enabled, None otherwise.
    """
    callback_cfg = cfg.logging.callbacks.get("texture_quality", {})

    if not callback_cfg.get("enabled", False):
        return None

    return TextureQualityCallback(
        cfg=cfg,
        log_every_n_epochs=callback_cfg.get("log_every_n_epochs", 25),
        n_samples=callback_cfg.get("n_samples", 200),
        lbp_radius=callback_cfg.get("lbp_radius", 1),
        lbp_n_points=callback_cfg.get("lbp_n_points", 8),
        wavelet=callback_cfg.get("wavelet", "db4"),
        compute_full_texture=callback_cfg.get("compute_full_texture", False),
    )
