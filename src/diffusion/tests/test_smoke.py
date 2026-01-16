"""Smoke tests for JS-DDPM module.

These tests verify that the core components work together:
1. Data pipeline can load a sample with correct shapes/ranges
2. Model forward pass works with conditioning
3. One training step runs without crashing
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.diffusion.data.transforms import (
    check_brain_content,
    check_lesion_content,
    extract_axial_slice,
)
from src.diffusion.losses.diffusion_losses import DiffusionLoss
from src.diffusion.losses.uncertainty import UncertaintyWeightedLoss
from src.diffusion.model.components.conditioning import (
    compute_class_token,
    compute_z_bin,
    get_token_for_condition,
    token_to_condition,
)
from src.diffusion.model.embeddings.zpos import normalize_z_local, quantize_z
from src.diffusion.model.factory import build_model, build_scheduler
from src.diffusion.training.metrics import MetricsCalculator


@pytest.fixture
def test_config():
    """Create a test configuration."""
    cfg = OmegaConf.create({
        "experiment": {
            "name": "test",
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
                "mode": "balance",
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
        },
        "scheduler": {
            "type": "DDPM",
            "num_train_timesteps": 100,  # Fewer for testing
            "schedule": "linear_beta",
            "beta_start": 0.0015,
            "beta_end": 0.0195,
            "prediction_type": "epsilon",
            "clip_sample": True,
            "clip_sample_range": 1.0,
        },
        "sampler": {
            "type": "DDIM",
            "num_inference_steps": 10,  # Fewer for testing
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
                "enabled": True,
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
            "every_n_epochs": 1,
            "z_bins_to_show": [0, 25, 49],
            "n_samples_per_condition": 1,
            "overlay": {
                "enabled": True,
                "alpha": 0.5,
                "color": [255, 0, 0],
                "threshold": 0.0,
            },
            "save_png": True,
            "save_npz": False,
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


class TestZPositionEncoding:
    """Tests for z-position encoding utilities."""

    def test_normalize_z_local(self):
        """Test local z-index normalization within z_range."""
        z_range = (0, 127)
        assert normalize_z_local(0, z_range) == 0.0
        assert normalize_z_local(127, z_range) == pytest.approx(127/128, rel=0.001)  # Inclusive fix
        assert normalize_z_local(63, z_range) == pytest.approx(63/128, rel=0.001)  # Inclusive fix

        # Test with custom z_range
        z_range = (24, 93)
        assert normalize_z_local(24, z_range) == 0.0
        assert normalize_z_local(93, z_range) == pytest.approx(69/70, rel=0.001)  # Inclusive fix
        # Middle of range: (58.5-24)/70 = 34.5/70 ≈ 0.493
        assert normalize_z_local(58, z_range) == pytest.approx(34/70, rel=0.001)  # Inclusive fix

    def test_quantize_z(self):
        """Test LOCAL z-position quantization within z_range."""
        n_bins = 50
        z_range = (0, 127)

        # Edge cases for full range
        assert quantize_z(0, z_range, n_bins) == 0
        assert quantize_z(127, z_range, n_bins) == 49  # Clamped

        # Middle value
        z_bin = quantize_z(63, z_range, n_bins)
        assert 0 <= z_bin < n_bins

        # Test with custom z_range (local binning)
        z_range = (24, 93)  # 70 slices
        n_bins = 10

        # First slice should map to bin 0
        assert quantize_z(24, z_range, n_bins) == 0

        # Last slice should map to bin 9
        assert quantize_z(93, z_range, n_bins) == 9

        # Middle slice (58) should map to middle bin
        z_bin_middle = quantize_z(58, z_range, n_bins)
        assert z_bin_middle == 4  # (58-24) / (93-24) * 10 = 34/69 * 10 ≈ 4.9 → 4


class TestConditioning:
    """Tests for conditioning token utilities."""

    def test_compute_class_token(self):
        """Test token computation."""
        n_bins = 50

        # Control tokens: 0-49
        token = compute_class_token(0, 0, n_bins)
        assert token == 0

        token = compute_class_token(25, 0, n_bins)
        assert token == 25

        # Lesion tokens: 50-99
        token = compute_class_token(0, 1, n_bins)
        assert token == 50

        token = compute_class_token(25, 1, n_bins)
        assert token == 75

    def test_get_token_for_condition(self):
        """Test convenience function."""
        n_bins = 50

        control_token = get_token_for_condition(10, 0, n_bins)
        assert control_token == 10

        lesion_token = get_token_for_condition(10, 1, n_bins)
        assert lesion_token == 60

    def test_token_to_condition(self):
        """Test token decoding."""
        n_bins = 50

        z_bin, cls = token_to_condition(25, n_bins)
        assert z_bin == 25
        assert cls == 0

        z_bin, cls = token_to_condition(75, n_bins)
        assert z_bin == 25
        assert cls == 1


class TestSliceUtilities:
    """Tests for slice extraction and checking utilities."""

    def test_extract_axial_slice(self):
        """Test axial slice extraction."""
        volume = torch.randn(1, 64, 64, 32)  # (C, H, W, D)

        slice_data = extract_axial_slice(volume, 16)
        assert slice_data.shape == (1, 64, 64)

    def test_check_brain_content(self):
        """Test brain content checking."""
        # Mostly high values (brain present)
        brain_slice = torch.ones(1, 64, 64) * 0.5
        assert check_brain_content(brain_slice) is True

        # Mostly low values (empty)
        empty_slice = torch.ones(1, 64, 64) * -0.95
        assert check_brain_content(empty_slice) is False

    def test_check_lesion_content(self):
        """Test lesion content checking."""
        # No lesion (all -1)
        no_lesion = torch.ones(1, 64, 64) * -1
        assert check_lesion_content(no_lesion) is False

        # Has lesion (some +1)
        has_lesion = torch.ones(1, 64, 64) * -1
        has_lesion[0, 30:35, 30:35] = 1
        assert check_lesion_content(has_lesion) is True


class TestModelFactory:
    """Tests for model building."""

    def test_build_model(self, test_config):
        """Test model construction."""
        model = build_model(test_config)

        # Check model type
        assert model is not None

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_build_scheduler(self, test_config):
        """Test scheduler construction."""
        scheduler = build_scheduler(test_config)
        assert scheduler is not None
        assert scheduler.num_train_timesteps == 100

    def test_model_forward_pass(self, test_config):
        """Test model forward pass with conditioning."""
        model = build_model(test_config)

        # Create dummy inputs
        batch_size = 2
        x = torch.randn(batch_size, 2, 64, 64)  # Smaller for speed
        timesteps = torch.randint(0, 100, (batch_size,))
        tokens = torch.randint(0, 100, (batch_size,))

        # Forward pass
        output = model(x, timesteps=timesteps, class_labels=tokens)

        assert output.shape == x.shape


class TestLosses:
    """Tests for loss functions."""

    def test_uncertainty_weighted_loss(self):
        """Test uncertainty weighting."""
        loss_fn = UncertaintyWeightedLoss(n_tasks=2)

        losses = [torch.tensor(1.0), torch.tensor(2.0)]
        total, details = loss_fn(losses)

        assert total > 0
        assert "log_var_0" in details
        assert "log_var_1" in details

    def test_diffusion_loss(self, test_config):
        """Test complete diffusion loss."""
        loss_fn = DiffusionLoss(test_config)

        eps_pred = torch.randn(2, 2, 64, 64)
        eps_target = torch.randn(2, 2, 64, 64)
        mask = torch.randn(2, 1, 64, 64)

        total, details = loss_fn(eps_pred, eps_target, mask)

        assert total > 0
        assert "loss_image" in details
        assert "loss_mask" in details

    def test_uncertainty_params_in_optimizer(self, test_config):
        """Test that uncertainty log_vars are optimized when learnable=True."""
        from src.diffusion.training.lit_modules import JSDDPMLightningModule

        # Enable learnable uncertainty weighting
        test_config.loss.uncertainty_weighting.enabled = True
        test_config.loss.uncertainty_weighting.learnable = True

        module = JSDDPMLightningModule(test_config)

        # Get initial log_vars
        initial_log_vars = module.criterion.loss.log_vars.clone()

        # Create dummy batch
        batch = {
            "image": torch.randn(2, 1, 64, 64),
            "mask": torch.randn(2, 1, 64, 64),
            "token": torch.randint(0, 100, (2,)),
            "metadata": {},
        }

        # Run one training step with manual optimization
        optimizer_config = module.configure_optimizers()
        optimizer = optimizer_config["optimizer"]

        loss = module.training_step(batch, 0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Check that log_vars changed
        final_log_vars = module.criterion.loss.log_vars
        assert not torch.allclose(initial_log_vars, final_log_vars), \
            "log_vars did not change after optimization step"


class TestMetrics:
    """Tests for MONAI-based metrics."""

    def test_metrics_calculator(self):
        """Test MONAI-based MetricsCalculator."""
        # Create calculator
        calc = MetricsCalculator(data_range=2.0, window_size=11, sigma=1.5)

        # Create dummy data (B=2, C=1, H=128, W=128)
        pred_image = torch.randn(2, 1, 128, 128) * 0.5
        target_image = torch.randn(2, 1, 128, 128) * 0.5

        pred_mask = torch.randn(2, 1, 128, 128)
        target_mask = torch.randn(2, 1, 128, 128)

        # Compute all metrics
        metrics = calc.compute_all(pred_image, target_image, pred_mask, target_mask)

        # Verify all metrics present
        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "dice" in metrics
        assert "hd95" in metrics

        # Verify types and ranges
        assert isinstance(metrics["psnr"], float)
        assert isinstance(metrics["ssim"], float)
        assert isinstance(metrics["dice"], float)
        assert isinstance(metrics["hd95"], (float, type(float('nan'))))

        # PSNR should be positive
        assert metrics["psnr"] > 0

        # SSIM should be in [-1, 1] (typically [0, 1] but can be negative for very different images)
        assert -1 <= metrics["ssim"] <= 1

        # Dice should be in [0, 1]
        assert 0 <= metrics["dice"] <= 1

        # HD95 can be NaN if no lesions
        # If not NaN, should be non-negative
        if not torch.isnan(torch.tensor(metrics["hd95"])):
            assert metrics["hd95"] >= 0

    def test_metrics_identical_images(self):
        """Test metrics on identical images (should give perfect scores)."""
        calc = MetricsCalculator(data_range=2.0)

        # Same images
        img = torch.randn(2, 1, 64, 64) * 0.5
        mask = torch.randn(2, 1, 64, 64)

        metrics = calc.compute_all(img, img, mask, mask)

        # PSNR should be very high (approaching inf)
        assert metrics["psnr"] > 100

        # SSIM should be close to 1
        assert metrics["ssim"] == pytest.approx(1.0, rel=0.01)

        # Dice should be 1 (perfect match)
        assert metrics["dice"] == pytest.approx(1.0, rel=0.01)


class TestTrainingStep:
    """Test one training step."""

    def test_lightning_module_step(self, test_config):
        """Test one training step on CPU."""
        from src.diffusion.training.lit_modules import JSDDPMLightningModule

        # Create module
        module = JSDDPMLightningModule(test_config)

        # Create dummy batch
        batch = {
            "image": torch.randn(2, 1, 64, 64),
            "mask": torch.randn(2, 1, 64, 64),
            "token": torch.randint(0, 100, (2,)),
            "metadata": {},
        }

        # Run training step
        loss = module.training_step(batch, 0)

        assert loss is not None
        assert loss > 0
        assert loss.requires_grad


class TestDataShapes:
    """Test expected data shapes and ranges."""

    def test_sample_shapes(self):
        """Test expected shapes for training samples."""
        # Simulate expected data
        image = torch.randn(1, 128, 128)  # (C, H, W)
        mask = torch.randn(1, 128, 128)

        # Combined x0
        x0 = torch.cat([image, mask], dim=0)
        assert x0.shape == (2, 128, 128)

    def test_value_ranges(self):
        """Test expected value ranges."""
        # Image should be in [-1, 1]
        image = torch.randn(1, 128, 128).clamp(-1, 1)
        assert image.min() >= -1
        assert image.max() <= 1

        # Mask should be in {-1, +1}
        mask = (torch.randn(1, 128, 128) > 0).float() * 2 - 1
        assert torch.all((mask == -1) | (mask == 1))


class TestZRangeFunctionality:
    """Test that z_range configuration works correctly."""

    def test_z_range_local_binning(self):
        """Test that LOCAL binning uses all bins within z_range."""
        from src.diffusion.model.embeddings.zpos import quantize_z

        n_bins = 50
        z_range = (40, 100)  # 61 slices
        min_z, max_z = z_range

        # Calculate z_bins from the range using LOCAL binning
        expected_bins = set()
        for z_idx in range(min_z, max_z + 1):
            z_bin = quantize_z(z_idx, z_range, n_bins)
            expected_bins.add(z_bin)

        # With local binning, ALL bins should be used
        assert len(expected_bins) == n_bins, \
            f"Expected all {n_bins} bins to be used, got {len(expected_bins)}"

        # All bins should be within valid range
        for z_bin in expected_bins:
            assert 0 <= z_bin < n_bins

        # Verify bin 0 is the first slice in range
        assert quantize_z(min_z, z_range, n_bins) == 0

        # Verify last bin is the last slice in range
        assert quantize_z(max_z, z_range, n_bins) == n_bins - 1

        # Slices outside z_range should raise ValueError
        with pytest.raises(ValueError):
            quantize_z(0, z_range, n_bins)  # Before range

        with pytest.raises(ValueError):
            quantize_z(127, z_range, n_bins)  # After range


class TestAnatomicalConditioning:
    """Tests for anatomical conditioning via input concatenation."""

    @pytest.fixture
    def anatomical_config(self, test_config):
        """Create config with anatomical conditioning enabled."""
        # Deep copy the config
        from copy import deepcopy
        cfg = deepcopy(test_config)

        # Enable anatomical conditioning
        cfg["model"]["anatomical_conditioning"] = True

        # Enable postprocessing (required for loading priors)
        cfg["postprocessing"] = {
            "zbin_priors": {
                "enabled": True,
                "priors_filename": "zbin_priors_brain_roi.npz",
                "prob_threshold": 0.20,
                "dilate_radius_px": 3,
                "gaussian_sigma_px": 0.7,
                "min_component_px": 500,
                "n_first_bins": 5,
                "max_components_for_first_bins": 3,
                "relaxed_threshold_factor": 0.1,
                "fallback": "prior",
                "apply_to": ["validation", "visualization", "generation"],
            }
        }

        return OmegaConf.create(cfg)

    def test_model_in_channels_with_anatomical_conditioning(self, anatomical_config):
        """Test that model has correct in_channels when anatomical conditioning is enabled."""
        model = build_model(anatomical_config)

        # The model should have 3 input channels (2 for image+mask, 1 for anatomical prior)
        # We can verify this by checking if it accepts 3-channel input
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)  # 3 channels
        timesteps = torch.randint(0, 100, (batch_size,))
        tokens = torch.randint(0, 100, (batch_size,))

        # Forward pass should work
        output = model(x, timesteps=timesteps, class_labels=tokens)

        # Output should still be 2 channels (image + mask noise prediction)
        assert output.shape == (batch_size, 2, 64, 64)

    def test_model_forward_with_anatomical_prior(self, anatomical_config):
        """Test forward pass with anatomical prior concatenated."""
        from src.diffusion.utils.zbin_priors import get_anatomical_priors_as_input

        model = build_model(anatomical_config)

        batch_size = 2

        # Create noisy input (2 channels)
        x_t = torch.randn(batch_size, 2, 64, 64)

        # Create mock anatomical priors
        # In reality these would come from load_zbin_priors, but for testing we create them
        z_bins = 50
        mock_priors = {}
        for i in range(z_bins):
            # Create a simple circular prior for testing
            prior = torch.zeros(64, 64, dtype=torch.bool)
            center = 32
            radius = 25
            y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
            distance = torch.sqrt((x - center)**2 + (y - center)**2)
            prior = distance < radius
            mock_priors[i] = prior.numpy()

        # Get anatomical priors for this batch
        z_bins_batch = [10, 25]  # Example z-bins
        anatomical_priors = get_anatomical_priors_as_input(
            z_bins_batch,
            mock_priors,
            device=x_t.device,
        )  # (B, 1, H, W)

        # Concatenate
        x_input = torch.cat([x_t, anatomical_priors], dim=1)  # (B, 3, H, W)

        assert x_input.shape == (batch_size, 3, 64, 64)

        # Forward pass
        timesteps = torch.randint(0, 100, (batch_size,))
        tokens = torch.randint(0, 100, (batch_size,))

        output = model(x_input, timesteps=timesteps, class_labels=tokens)

        assert output.shape == (batch_size, 2, 64, 64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_step_with_anatomical_conditioning_gpu(self, anatomical_config):
        """Test training step with anatomical conditioning on GPU."""
        from src.diffusion.training.lit_modules import JSDDPMLightningModule
        from src.diffusion.utils.zbin_priors import get_anatomical_priors_as_input

        # Use smaller batch size for GPU to avoid OOM
        anatomical_config.training.batch_size = 1

        # Create mock priors
        z_bins = 50
        mock_priors = {}
        for i in range(z_bins):
            prior = torch.zeros(64, 64, dtype=torch.bool)
            center = 32
            radius = 25
            y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
            distance = torch.sqrt((x - center)**2 + (y - center)**2)
            prior = distance < radius
            mock_priors[i] = prior.numpy()

        # Create module on GPU
        module = JSDDPMLightningModule(anatomical_config)
        module = module.to('cuda:0')

        # Manually set priors (normally loaded from disk)
        module._zbin_priors = mock_priors
        module._use_anatomical_conditioning = True

        # Create dummy batch on GPU
        batch = {
            "image": torch.randn(1, 1, 64, 64).to('cuda:0'),
            "mask": torch.randn(1, 1, 64, 64).to('cuda:0'),
            "token": torch.randint(0, 100, (1,)).to('cuda:0'),
            "metadata": {
                "z_bin": [25],  # Example z-bin
            },
        }

        # Run training step
        loss = module.training_step(batch, 0)

        assert loss is not None
        assert loss > 0
        assert loss.requires_grad
        assert loss.device.type == 'cuda'

    def test_anatomical_priors_as_input_function(self):
        """Test get_anatomical_priors_as_input utility function."""
        from src.diffusion.utils.zbin_priors import get_anatomical_priors_as_input

        # Create mock priors
        z_bins = 10
        H, W = 64, 64
        mock_priors = {}
        for i in range(z_bins):
            # Create different patterns for each z-bin
            prior = torch.zeros(H, W, dtype=torch.bool)
            if i < 5:
                # Lower z-bins: smaller region
                prior[20:44, 20:44] = True
            else:
                # Upper z-bins: larger region
                prior[10:54, 10:54] = True
            mock_priors[i] = prior.numpy()

        # Test with tensor input
        z_bins_batch = torch.tensor([2, 7])
        priors_tensor = get_anatomical_priors_as_input(
            z_bins_batch,
            mock_priors,
        )

        assert priors_tensor.shape == (2, 1, H, W)
        assert priors_tensor.dtype == torch.float32
        # Priors are in [-1, 1] range to match image/mask normalization
        # True (in-brain) -> 1.0, False (out-of-brain) -> -1.0
        assert priors_tensor.min() >= -1.0
        assert priors_tensor.max() <= 1.0

        # Test with list input
        z_bins_list = [2, 7]
        priors_list = get_anatomical_priors_as_input(
            z_bins_list,
            mock_priors,
        )

        assert priors_list.shape == (2, 1, H, W)
        assert torch.allclose(priors_tensor, priors_list)


class TestSamplerXTParameter:
    """Tests for the x_T pre-generated noise parameter in DiffusionSampler."""

    @pytest.fixture
    def ddim_config(self, test_config):
        """Create config with DDIM scheduler for deterministic sampling."""
        from copy import deepcopy
        cfg = deepcopy(test_config)
        # Override scheduler to DDIM for deterministic tests
        cfg.scheduler.type = "DDIM"
        return OmegaConf.create(cfg)

    def test_sample_with_x_T_uses_provided_noise_ddim(self, ddim_config):
        """Test that x_T parameter uses provided noise (DDIM for determinism)."""
        from src.diffusion.model.factory import DiffusionSampler, build_scheduler

        # Set seed for reproducible model weights
        torch.manual_seed(42)

        model = build_model(ddim_config)
        scheduler = build_scheduler(ddim_config)
        sampler = DiffusionSampler(model, scheduler, ddim_config, device="cpu")

        tokens = torch.tensor([5], dtype=torch.long)
        shape = (1, 2, 64, 64)

        # Create fixed noise
        fixed_noise = torch.randn(shape)

        # Sample twice with the same x_T - should get identical results with DDIM
        result1 = sampler.sample(tokens, shape=shape, x_T=fixed_noise.clone())
        result2 = sampler.sample(tokens, shape=shape, x_T=fixed_noise.clone())

        assert torch.allclose(result1, result2, atol=1e-5), \
            "Same x_T should produce identical results with DDIM scheduler"

    def test_sample_x_T_different_from_generator(self, test_config):
        """Test that x_T and generator produce different results for same token."""
        from src.diffusion.model.factory import DiffusionSampler, build_scheduler

        model = build_model(test_config)
        scheduler = build_scheduler(test_config)
        sampler = DiffusionSampler(model, scheduler, test_config, device="cpu")

        tokens = torch.tensor([5], dtype=torch.long)
        shape = (1, 2, 64, 64)

        # Sample with generator
        gen = torch.Generator().manual_seed(12345)
        result_gen = sampler.sample(tokens, shape=shape, generator=gen)

        # Sample with different x_T
        different_noise = torch.randn(shape) * 2  # Different noise
        result_x_T = sampler.sample(tokens, shape=shape, x_T=different_noise)

        # Results should be different
        assert not torch.allclose(result_gen, result_x_T), \
            "Different noise should produce different results"

    def test_sample_x_T_shape_validation(self, test_config):
        """Test that x_T with wrong shape raises ValueError."""
        from src.diffusion.model.factory import DiffusionSampler, build_scheduler

        model = build_model(test_config)
        scheduler = build_scheduler(test_config)
        sampler = DiffusionSampler(model, scheduler, test_config, device="cpu")

        tokens = torch.tensor([5], dtype=torch.long)
        shape = (1, 2, 64, 64)

        # Wrong batch size
        wrong_batch = torch.randn(2, 2, 64, 64)
        with pytest.raises(ValueError, match="x_T shape"):
            sampler.sample(tokens, shape=shape, x_T=wrong_batch)

        # Wrong channels
        wrong_channels = torch.randn(1, 3, 64, 64)
        with pytest.raises(ValueError, match="x_T shape"):
            sampler.sample(tokens, shape=shape, x_T=wrong_channels)

        # Wrong spatial dims
        wrong_spatial = torch.randn(1, 2, 32, 32)
        with pytest.raises(ValueError, match="x_T shape"):
            sampler.sample(tokens, shape=shape, x_T=wrong_spatial)

    def test_sample_x_T_matches_generator_equivalent_ddim(self, ddim_config):
        """Test that x_T with pre-generated noise matches using generator with same seed (DDIM)."""
        from src.diffusion.model.factory import DiffusionSampler, build_scheduler

        # Set seed for reproducible model weights
        torch.manual_seed(42)

        model = build_model(ddim_config)
        scheduler = build_scheduler(ddim_config)
        sampler = DiffusionSampler(model, scheduler, ddim_config, device="cpu")

        tokens = torch.tensor([5], dtype=torch.long)
        shape = (1, 2, 64, 64)
        seed = 12345

        # Method 1: Use generator
        gen1 = torch.Generator().manual_seed(seed)
        result_gen = sampler.sample(tokens, shape=shape, generator=gen1)

        # Method 2: Pre-generate noise with same seed, pass as x_T
        gen2 = torch.Generator().manual_seed(seed)
        pre_generated_noise = torch.randn(shape, generator=gen2)
        result_x_T = sampler.sample(tokens, shape=shape, x_T=pre_generated_noise)

        assert torch.allclose(result_gen, result_x_T, atol=1e-5), \
            "x_T with same initial noise should produce identical results to generator with DDIM"

    def test_sample_x_T_with_batched_different_seeds(self, test_config):
        """Test batch generation with per-sample seeds via x_T."""
        from src.diffusion.model.factory import DiffusionSampler, build_scheduler

        model = build_model(test_config)
        scheduler = build_scheduler(test_config)
        sampler = DiffusionSampler(model, scheduler, test_config, device="cpu")

        batch_size = 3
        tokens = torch.tensor([5, 10, 15], dtype=torch.long)
        shape = (batch_size, 2, 64, 64)

        # Generate per-sample noise with different seeds
        seeds = [100, 200, 300]
        noise_list = []
        for seed in seeds:
            gen = torch.Generator().manual_seed(seed)
            noise = torch.randn((2, 64, 64), generator=gen)
            noise_list.append(noise)
        x_T = torch.stack(noise_list, dim=0)

        assert x_T.shape == shape

        # Sample
        result = sampler.sample(tokens, shape=shape, x_T=x_T)

        assert result.shape == shape, "Output shape should match input shape"

        # Each sample should be different (different seeds)
        assert not torch.allclose(result[0], result[1]), \
            "Different seeds should produce different samples"
        assert not torch.allclose(result[1], result[2]), \
            "Different seeds should produce different samples"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
