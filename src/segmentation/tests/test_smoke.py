"""Smoke tests for segmentation module."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.segmentation.models.factory import build_model
from src.segmentation.utils.config import load_and_merge_configs


class TestConfiguration:
    """Test configuration loading and merging."""

    def test_load_master_config(self):
        """Test master config loads."""
        cfg = OmegaConf.load("src/segmentation/config/master.yaml")
        assert cfg.experiment.name is not None
        assert cfg.k_fold.n_folds == 5
        assert cfg.model.spatial_dims == 2

    def test_load_model_configs(self):
        """Test model configs load."""
        for model_name in ["unet", "dynunet", "unetplusplus", "swinunetr"]:
            cfg = OmegaConf.load(
                f"src/segmentation/config/models/{model_name}.yaml"
            )
            assert cfg.model.name is not None

    def test_merge_configs(self):
        """Test merging master + model configs."""
        cfg = load_and_merge_configs(
            master_path="src/segmentation/config/master.yaml",
            model_name="unet",
        )
        assert cfg.model.name.lower() == "unet"
        assert cfg.model.channels is not None
        assert cfg.k_fold.n_folds == 5


class TestModelFactory:
    """Test model instantiation."""

    @pytest.fixture
    def test_config_unet(self):
        """Create test config for UNet."""
        return load_and_merge_configs(
            master_path="src/segmentation/config/master.yaml",
            model_name="unet",
        )

    @pytest.fixture
    def test_config_dynunet(self):
        """Create test config for DynUNet."""
        return load_and_merge_configs(
            master_path="src/segmentation/config/master.yaml",
            model_name="dynunet",
        )

    def test_build_unet(self, test_config_unet):
        """Test UNet instantiation."""
        model = build_model(test_config_unet)
        assert model is not None

        # Test forward pass
        x = torch.randn(2, 1, 128, 128)
        y = model(x)
        assert y.shape == (2, 1, 128, 128)

    def test_build_dynunet(self, test_config_dynunet):
        """Test DynUNet instantiation."""
        model = build_model(test_config_dynunet)
        assert model is not None

        # Test forward pass
        x = torch.randn(2, 1, 128, 128)
        y = model(x)
        assert y.shape == (2, 1, 128, 128)


class TestDataPipeline:
    """Test data loading."""

    def test_mask_conversion(self):
        """Test mask conversion from {-1,+1} to {0,1}."""
        import numpy as np

        # Simulate mask in {-1, +1}
        mask = np.array([[-1, -1, 1], [-1, 1, 1]], dtype=np.float32)

        # Convert
        mask_binary = (mask > 0.0).astype(np.float32)

        # Check
        assert mask_binary.min() == 0.0
        assert mask_binary.max() == 1.0
        assert mask_binary.sum() == 3  # 3 ones (last two columns of second row + last column of first row)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
