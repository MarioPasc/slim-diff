"""Tests for reconstruction-based validation metrics.

This test verifies that the reconstruction-based validation approach works correctly:
1. Timestep selection divides T into 4 parts correctly
2. Denoising from a timestep produces reasonable reconstructions
3. Metrics are computed at all 4 timesteps with correct naming
4. The validation step returns metrics with timestep suffixes
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from src.diffusion.losses.diffusion_losses import DiffusionLoss
from src.diffusion.model.factory import build_model, build_scheduler
from src.diffusion.training.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


@pytest.fixture
def test_config():
    """Create a test configuration for reconstruction validation tests."""
    cfg = OmegaConf.create({
        "experiment": {
            "name": "test_reconstruction",
            "output_dir": "./test_outputs",
            "seed": 42,
        },
        "data": {
            "root_dir": "/tmp",
            "cache_dir": "/tmp/cache",
            "epilepsy": {"name": "test_epi", "modality_index": 0},
            "control": {"name": "test_ctl", "modality_index": 0},
            "splits": {
                "use_predefined_test": True,
                "val_fraction": 0.1,
                "control_test_fraction": 0.15,
                "seed": 42,
            },
            "transforms": {
                "target_spacing": [1.875, 1.875, 1.875],
                "roi_size": [128, 128, 128],
                "intensity_norm": {
                    "type": "percentile",
                    "lower": 0.5,
                    "upper": 99.5,
                    "b_min": -1.0,
                    "b_max": 1.0,
                    "clip": True,
                },
            },
            "slice_sampling": {
                "z_range": [0, 127],
                "filter_empty_brain": True,
                "brain_threshold": -0.9,
                "brain_min_fraction": 0.05,
            },
            "lesion_oversampling": {
                "enabled": True,
                "weight": 5.0,
            },
        },
        "conditioning": {
            "z_bins": 50,
            "use_sinusoidal": False,
            "max_z": 127,
            "cfg": {
                "enabled": False,
                "null_token": 100,
                "dropout_prob": 0.1,
            },
        },
        "model": {
            "type": "DiffusionModelUNet",
            "spatial_dims": 2,
            "in_channels": 2,
            "out_channels": 2,
            "channels": [32, 64, 64],  # Smaller for testing
            "attention_levels": [False, False, True],
            "num_res_blocks": 1,
            "num_head_channels": 16,
            "norm_name": "GROUP",
            "norm_num_groups": 8,
            "use_class_embedding": True,
            "dropout": 0.0,
            "resblock_updown": False,
            "with_conditioning": False,
            "anatomical_conditioning": False,
        },
        "scheduler": {
            "type": "DDPM",
            "num_train_timesteps": 1000,  # Standard 1000 timesteps
            "schedule": "linear_beta",
            "beta_start": 0.0015,
            "beta_end": 0.0195,
            "prediction_type": "epsilon",
            "clip_sample": True,
            "clip_sample_range": 1.0,
        },
        "sampler": {
            "type": "DDIM",
            "num_inference_steps": 50,  # Fewer for testing
            "eta": 0.0,
            "guidance_scale": 1.0,
        },
        "training": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "optimizer": {
                "type": "AdamW",
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "lr_scheduler": {
                "type": "CosineAnnealingLR",
                "T_max": None,
                "eta_min": 1e-6,
            },
            "max_epochs": 1,
            "max_steps": None,
            "precision": "32",
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
            "accumulate_grad_batches": 1,
            "val_check_interval": 1.0,
            "check_val_every_n_epoch": 1,
            "early_stopping": {"enabled": False},
        },
        "loss": {
            "uncertainty_weighting": {
                "enabled": False,
                "initial_log_vars": [0.0, 0.0],
                "learnable": True,
            },
            "lesion_weighted_mask": {
                "enabled": False,
                "lesion_weight": 2.0,
                "background_weight": 1.0,
            },
        },
        "logging": {
            "log_every_n_steps": 1,
            "logger": {"type": "tensorboard"},
            "checkpointing": {
                "save_top_k": 1,
                "monitor": "val/loss",
                "mode": "min",
                "save_last": True,
                "every_n_epochs": 1,
                "filename": "test",
            },
        },
        "visualization": {
            "enabled": False,
        },
        "postprocessing": {
            "zbin_priors": {
                "enabled": False,
                "apply_to": [],
            },
        },
        "generation": {
            "n_per_condition": 1,
            "z_bins": None,
            "classes": [0, 1],
            "output_format": "npz",
            "save_individual": True,
            "create_index_csv": True,
        },
    })
    return cfg


class TestReconstructionTimesteps:
    """Tests for reconstruction timestep selection."""

    def test_timestep_division_1000(self, test_config):
        """Test that T=1000 divides into [250, 500, 750, 999]."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST: Timestep division for T=1000")
        logger.info("=" * 60)

        # Mock the lightning module to test _get_reconstruction_timesteps
        test_config.scheduler.num_train_timesteps = 1000
        
        from src.diffusion.training.lit_modules import JSDDPMLightningModule
        
        module = JSDDPMLightningModule(test_config)
        timesteps = module._get_reconstruction_timesteps()
        
        logger.info(f"Computed timesteps: {timesteps}")
        
        # Expected: T/4=250, T/2=500, 3T/4=750, T-1=999
        expected = [250, 500, 750, 999]
        assert timesteps == expected, f"Expected {expected}, got {timesteps}"
        
        logger.info("✓ Timestep division for T=1000 passed")

    def test_timestep_division_100(self, test_config):
        """Test that T=100 divides into [25, 50, 75, 99]."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST: Timestep division for T=100")
        logger.info("=" * 60)

        test_config.scheduler.num_train_timesteps = 100
        
        from src.diffusion.training.lit_modules import JSDDPMLightningModule
        
        module = JSDDPMLightningModule(test_config)
        timesteps = module._get_reconstruction_timesteps()
        
        logger.info(f"Computed timesteps: {timesteps}")
        
        # Expected: T/4=25, T/2=50, 3T/4=75, T-1=99
        expected = [25, 50, 75, 99]
        assert timesteps == expected, f"Expected {expected}, got {timesteps}"
        
        logger.info("✓ Timestep division for T=100 passed")

    def test_timesteps_are_valid_range(self, test_config):
        """Test that all timesteps are within valid scheduler range [0, T-1]."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST: Timesteps within valid range")
        logger.info("=" * 60)

        from src.diffusion.training.lit_modules import JSDDPMLightningModule
        
        for T in [100, 500, 1000, 2000]:
            test_config.scheduler.num_train_timesteps = T
            module = JSDDPMLightningModule(test_config)
            timesteps = module._get_reconstruction_timesteps()
            
            for t in timesteps:
                assert 0 <= t < T, f"Timestep {t} out of range [0, {T-1}] for T={T}"
            
            logger.info(f"T={T}: timesteps {timesteps} all valid")
        
        logger.info("✓ All timesteps within valid range")


class TestReconstructionMetrics:
    """Tests for reconstruction-based metric computation."""

    @pytest.fixture
    def metrics_calc(self):
        """Create a MetricsCalculator instance."""
        return MetricsCalculator(data_range=2.0)

    def test_metrics_returned_with_timestep_suffix(self, test_config, metrics_calc):
        """Test that metrics are computed with timestep suffix."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST: Metrics with timestep suffix")
        logger.info("=" * 60)

        # Create synthetic data
        B, H, W = 2, 128, 128
        pred_image = torch.randn(B, 1, H, W) * 0.5
        target_image = torch.randn(B, 1, H, W) * 0.5
        pred_mask = torch.randn(B, 1, H, W).sign()
        target_mask = torch.randn(B, 1, H, W).sign()

        # Compute metrics for each timestep
        reconstruction_timesteps = [250, 500, 750, 999]
        all_metrics = {}
        
        for t_val in reconstruction_timesteps:
            metrics_t = metrics_calc.compute_all(
                pred_image, target_image,
                pred_mask, target_mask,
            )
            
            # Store with timestep suffix (as done in validation_step)
            for metric_name, value in metrics_t.items():
                all_metrics[f"{metric_name}_t{t_val}"] = value

        logger.info(f"Metrics keys: {list(all_metrics.keys())}")

        # Check all expected keys exist
        expected_metrics = ["psnr", "ssim", "dice", "hd95"]
        for t in reconstruction_timesteps:
            for metric in expected_metrics:
                key = f"{metric}_t{t}"
                assert key in all_metrics, f"Missing metric: {key}"
                logger.info(f"  {key}: {all_metrics[key]:.4f}")

        logger.info("✓ Metrics with timestep suffix passed")

    def test_reconstruction_improves_with_lower_t(self, test_config):
        """Test that reconstruction quality improves with lower timesteps (less noise)."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST: Reconstruction quality vs timestep")
        logger.info("=" * 60)

        # Create original image
        B, H, W = 4, 128, 128
        x0 = torch.randn(B, 2, H, W) * 0.5  # Original clean image+mask
        
        # Build scheduler to get noise schedule
        scheduler = build_scheduler(test_config)
        
        metrics_calc = MetricsCalculator(data_range=2.0)
        
        psnr_values = []
        timesteps_to_test = [100, 250, 500, 750, 999]
        
        for t_val in timesteps_to_test:
            # Add noise at timestep t
            t_tensor = torch.full((B,), t_val, dtype=torch.long)
            noise = torch.randn_like(x0)
            
            # Get alpha_bar at this timestep
            alpha_bar_t = scheduler.alphas_cumprod[t_tensor]
            while alpha_bar_t.dim() < x0.dim():
                alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
            
            x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
            
            # Compute PSNR between noisy and original (proxy for reconstruction difficulty)
            metrics = metrics_calc.compute_all(
                x_t[:, 0:1], x0[:, 0:1],
                x_t[:, 1:2], x0[:, 1:2],
            )
            psnr_values.append(metrics["psnr"])
            
            logger.info(f"t={t_val}: PSNR={metrics['psnr']:.2f} dB, alpha_bar={alpha_bar_t[0,0,0,0].item():.4f}")

        # Verify PSNR decreases as timestep increases (more noise = worse PSNR)
        for i in range(len(psnr_values) - 1):
            assert psnr_values[i] > psnr_values[i + 1], \
                f"PSNR should decrease with higher t: t={timesteps_to_test[i]} ({psnr_values[i]:.2f}) " \
                f"vs t={timesteps_to_test[i+1]} ({psnr_values[i+1]:.2f})"

        logger.info("✓ Reconstruction quality decreases with higher timesteps (more noise)")


class TestValidationStepIntegration:
    """Integration tests for the full validation step."""

    def test_validation_step_returns_timestep_metrics(self, test_config):
        """Test that validation_step returns metrics with timestep suffixes."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST: Validation step returns timestep metrics")
        logger.info("=" * 60)

        # Use smaller timesteps for faster test
        test_config.scheduler.num_train_timesteps = 100
        test_config.sampler.num_inference_steps = 10
        
        from src.diffusion.training.lit_modules import JSDDPMLightningModule
        
        module = JSDDPMLightningModule(test_config)
        module.eval()
        
        # Create a mock batch
        B, H, W = 2, 128, 128
        batch = {
            "image": torch.randn(B, 1, H, W),
            "mask": torch.randn(B, 1, H, W).sign(),
            "token": torch.randint(0, 100, (B,)),
            "metadata": {"z_bin": [0, 1]},
        }
        
        # Mock the log method to capture logged metrics
        logged_metrics = {}
        def mock_log(name, value, **kwargs):
            logged_metrics[name] = value.item() if hasattr(value, 'item') else value
        module.log = mock_log
        
        # Run validation step
        with torch.no_grad():
            result = module.validation_step(batch, batch_idx=0)
        
        logger.info(f"Result keys: {list(result.keys())}")
        logger.info(f"Logged metrics: {list(logged_metrics.keys())}")
        
        # Check that timestep-specific metrics are in the result
        expected_timesteps = module._get_reconstruction_timesteps()
        
        for t in expected_timesteps:
            # Check result dict
            assert f"psnr_t{t}" in result, f"Missing psnr_t{t} in result"
            assert f"ssim_t{t}" in result, f"Missing ssim_t{t} in result"
            
            # Check logged metrics
            assert f"val/psnr_t{t}" in logged_metrics, f"Missing val/psnr_t{t} in logged"
            assert f"val/ssim_t{t}" in logged_metrics, f"Missing val/ssim_t{t} in logged"
            
            logger.info(f"t={t}: PSNR={result[f'psnr_t{t}']:.2f}, SSIM={result[f'ssim_t{t}']:.4f}")
        
        # Verify loss is still computed
        assert "loss" in result, "Missing loss in result"
        assert "val/loss" in logged_metrics, "Missing val/loss in logged"
        
        logger.info("✓ Validation step returns timestep metrics")

    def test_validation_step_metric_values_are_reasonable(self, test_config):
        """Test that computed metric values are in reasonable ranges."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST: Validation metric values are reasonable")
        logger.info("=" * 60)

        # Use smaller timesteps for faster test
        test_config.scheduler.num_train_timesteps = 100
        test_config.sampler.num_inference_steps = 10
        
        from src.diffusion.training.lit_modules import JSDDPMLightningModule
        
        module = JSDDPMLightningModule(test_config)
        module.eval()
        
        # Create a mock batch
        B, H, W = 2, 128, 128
        batch = {
            "image": torch.randn(B, 1, H, W),
            "mask": torch.randn(B, 1, H, W).sign(),
            "token": torch.randint(0, 100, (B,)),
            "metadata": {"z_bin": [0, 1]},
        }
        
        # Disable logging
        module.log = lambda *args, **kwargs: None
        
        # Run validation step
        with torch.no_grad():
            result = module.validation_step(batch, batch_idx=0)
        
        expected_timesteps = module._get_reconstruction_timesteps()
        
        for t in expected_timesteps:
            psnr = result[f"psnr_t{t}"]
            ssim = result[f"ssim_t{t}"]
            
            # PSNR should be in reasonable range (even with random model, not too extreme)
            assert -10 < psnr < 50, f"PSNR at t={t} out of range: {psnr}"
            
            # SSIM should be in [0, 1] (or slightly negative for very bad reconstructions)
            assert -0.5 < ssim < 1.0, f"SSIM at t={t} out of range: {ssim}"
            
            logger.info(f"t={t}: PSNR={psnr:.2f}, SSIM={ssim:.4f} (reasonable)")
        
        logger.info("✓ All metric values are in reasonable ranges")


class TestCSVLogging:
    """Tests for CSV logging of timestep-specific metrics."""

    def test_csv_callback_handles_timestep_metrics(self, test_config, tmp_path):
        """Test that CSVLoggingCallback correctly handles new timestep metric columns."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST: CSV callback handles timestep metrics")
        logger.info("=" * 60)

        from src.diffusion.training.callbacks.csv_callback import CSVLoggingCallback
        
        # Update config to use tmp_path
        test_config.experiment.output_dir = str(tmp_path)
        
        callback = CSVLoggingCallback(test_config)
        
        # Simulate metrics that would be logged
        timesteps = [250, 500, 750, 999]
        metrics_row = {
            "epoch": 0,
            "step": 100,
            "val/loss": 0.5,
        }
        
        # Add timestep-specific metrics
        for t in timesteps:
            metrics_row[f"val/psnr_t{t}"] = 20.0 + t / 100  # Dummy values
            metrics_row[f"val/ssim_t{t}"] = 0.5 + t / 2000
            metrics_row[f"val/dice_t{t}"] = 0.7 + t / 5000
            metrics_row[f"val/hd95_t{t}"] = 10.0 - t / 200
        
        # Update internal metric names tracking
        callback._all_metric_names.update(metrics_row.keys())
        callback._all_metric_names.discard("epoch")
        callback._all_metric_names.discard("step")
        
        # Write row
        callback._write_row(metrics_row)
        
        # Read back and verify
        import csv
        with open(callback.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 1, "Should have written 1 row"
        
        row = rows[0]
        logger.info(f"CSV columns: {list(row.keys())}")
        
        # Verify timestep columns exist
        for t in timesteps:
            assert f"val/psnr_t{t}" in row, f"Missing val/psnr_t{t} column"
            assert f"val/ssim_t{t}" in row, f"Missing val/ssim_t{t} column"
            
        logger.info("✓ CSV callback handles timestep metrics correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
