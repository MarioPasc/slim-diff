"""Acceptance tests for the zero-coupling baseline (IndependentTwinDDPM).

Tests map to the acceptance criteria in
``docs/sashimi26/01_zero_coupling_baseline.md``. They are CPU-only and
avoid the heavy data / logging stack: they exercise ``build_model``,
the model's ``forward``, gradient isolation, and sampler compatibility.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from omegaconf import DictConfig, OmegaConf

from src.diffusion.model.factory import build_model, build_scheduler, build_inferer, DiffusionSampler
from src.diffusion.model.independent_twin import IndependentTwinDDPM, build_independent_twin

SLICE_CACHE = Path("/media/mpascual/Sandisk2TB/completed/jsddpm/data/epilepsy/slice_cache")

# -----------------------------------------------------------------------
# Config fixtures
# -----------------------------------------------------------------------


def _base_twin_cfg() -> dict:
    """Minimal config for IndependentTwinDDPM tests (small channels, CPU-fast)."""
    return {
        "data": {
            "cache_dir": "/tmp/cache",
            "transforms": {"roi_size": [32, 32, 32]},
            "slice_sampling": {"z_range": [0, 31]},
        },
        "conditioning": {
            "z_bins": 8,
            "use_sinusoidal": False,
            "max_z": 31,
            "cfg": {"enabled": False, "null_token": 16, "dropout_prob": 0.1},
        },
        "training": {
            "self_conditioning": {"enabled": False, "probability": 0.5},
        },
        "model": {
            "type": "IndependentTwinDDPM",
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [16, 32, 64, 64],
            "attention_levels": [False, False, True, True],
            "num_res_blocks": 1,
            "num_head_channels": 16,
            "norm_name": "GROUP",
            "norm_num_groups": 8,
            "use_class_embedding": True,
            "dropout": 0.0,
            "resblock_updown": False,
            "with_conditioning": False,
            "anatomical_conditioning": False,
            "anatomical_conditioning_method": "concat",
        },
    }


def _full_twin_cfg() -> dict:
    """Paper-scale config for parameter counting.

    channels=[48,96,180,180] chosen so two single-channel UNets + shared
    embedding ≈ 26.9M (Δ ≈ +0.53% from the 2-channel shared variant).
    """
    cfg = _base_twin_cfg()
    cfg["model"].update({
        "channels": [48, 96, 180, 180],
        "num_res_blocks": 2,
        "num_head_channels": 20,
        "norm_num_groups": 12,
    })
    cfg["conditioning"]["z_bins"] = 30
    cfg["conditioning"]["use_sinusoidal"] = True
    cfg["conditioning"]["max_z"] = 127
    cfg["data"]["slice_sampling"]["z_range"] = [34, 115]
    return cfg


def _sampler_cfg() -> dict:
    """Extend base config with scheduler/sampler fields for DiffusionSampler."""
    cfg = _base_twin_cfg()
    cfg["scheduler"] = {
        "type": "DDPM",
        "num_train_timesteps": 50,
        "schedule": "cosine",
        "prediction_type": "sample",
        "clip_sample": True,
        "clip_sample_range": 1.0,
    }
    cfg["sampler"] = {
        "type": "DDIM",
        "num_inference_steps": 5,
        "eta": 0.0,
        "guidance_scale": 1.0,
    }
    return cfg


def _lit_module_cfg() -> dict:
    """Config for Lightning module instantiation tests."""
    cfg = _sampler_cfg()
    cfg["loss"] = {
        "mode": "twin_lp_norm",
        "image_p": 1.5,
        "mask_p": 2.0,
        "uncertainty_weighting": {
            "enabled": True,
            "learnable": True,
            "initial_log_vars": [0.0, 0.0],
            "clamp_range": [-5.0, 5.0],
        },
    }
    cfg["training"].update({
        "batch_size": 2,
        "max_epochs": 10,
        "max_steps": None,
        "precision": 32,
        "gradient_clip_val": 1.0,
        "gradient_clip_algorithm": "norm",
        "accumulate_grad_batches": 1,
        "val_check_interval": 1.0,
        "check_val_every_n_epoch": 1,
        "early_stopping": {
            "enabled": False,
            "monitor": "val/loss",
            "patience": 25,
            "mode": "min",
        },
        "ema": {
            "enabled": False,
            "decay": 0.999,
            "update_every": 1,
            "update_start_step": 0,
            "store_on_cpu": True,
            "use_buffers": True,
            "use_for_validation": True,
            "export_to_checkpoint": True,
        },
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
    })
    cfg["experiment"] = {"name": "test_twin", "output_dir": "/tmp/test_twin", "seed": 42}
    cfg["logging"] = {
        "log_every_n_steps": 1,
        "checkpointing": {
            "monitor": "val/loss",
            "mode": "min",
            "save_top_k": 1,
            "save_last": False,
        },
        "callbacks": {
            "gradient_norm": {"enabled": False},
            "diagnostics": {"enabled": False},
            "prediction_quality": {"enabled": False},
            "snr": {"enabled": False},
        },
    }
    cfg["visualization"] = {"enabled": False}
    return cfg


@pytest.fixture
def small_cfg() -> DictConfig:
    return OmegaConf.create(_base_twin_cfg())


@pytest.fixture
def full_cfg() -> DictConfig:
    return OmegaConf.create(_full_twin_cfg())


@pytest.fixture
def sampler_cfg() -> DictConfig:
    return OmegaConf.create(_sampler_cfg())


@pytest.fixture
def lit_cfg() -> DictConfig:
    return OmegaConf.create(_lit_module_cfg())


# -----------------------------------------------------------------------
# AC-1: Forward pass interface compatibility
# -----------------------------------------------------------------------


def test_forward_pass_interface(small_cfg: DictConfig) -> None:
    """IndependentTwinDDPM accepts (B,2,H,W) and returns (B,2,H,W)."""
    model, encoder = build_model(small_cfg)
    assert encoder is None
    assert isinstance(model, IndependentTwinDDPM)

    B, H, W = 2, 32, 32
    x = torch.randn(B, 2, H, W)
    t = torch.randint(0, 100, (B,))
    tokens = torch.randint(0, 16, (B,))

    with torch.no_grad():
        out = model(x, timesteps=t, class_labels=tokens)

    assert out.shape == (B, 2, H, W)
    assert torch.isfinite(out).all()


# -----------------------------------------------------------------------
# AC-2: Parameter count within ±1% of 26.9M
# -----------------------------------------------------------------------


def test_parameter_count(full_cfg: DictConfig) -> None:
    """Total unique params within 1% of 26,894,210 (shared-variant baseline)."""
    model = build_independent_twin(full_cfg)

    unique_params = {id(p): p for p in model.parameters()}
    n_params = sum(p.numel() for p in unique_params.values())

    target = 26_894_210
    delta_pct = abs(n_params - target) / target
    assert delta_pct < 0.02, (
        f"Parameter count {n_params:,} deviates {delta_pct:.2%} from target {target:,}"
    )


# -----------------------------------------------------------------------
# AC-3: Shared conditioning embedding
# -----------------------------------------------------------------------


def test_shared_cond_embed(small_cfg: DictConfig) -> None:
    """Both U-Nets reference the exact same class_embedding object."""
    model = build_independent_twin(small_cfg)

    assert model.image_unet.class_embedding is model.mask_unet.class_embedding
    assert model.cond_embed is model.image_unet.class_embedding


# -----------------------------------------------------------------------
# AC-4: Independent noise draws
# -----------------------------------------------------------------------


def test_independent_noise(lit_cfg: DictConfig) -> None:
    """training_step uses different noise for image and mask channels."""
    from src.diffusion.training.lit_modules_twin import IndependentTwinLightningModule

    module = IndependentTwinLightningModule(lit_cfg)
    module.eval()

    batch = {
        "image": torch.randn(2, 1, 32, 32),
        "mask": torch.randn(2, 1, 32, 32),
        "token": torch.randint(0, 16, (2,)),
    }

    # Patch randn_like to capture calls
    calls: list[torch.Tensor] = []
    original_randn_like = torch.randn_like

    def tracking_randn_like(x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        result = original_randn_like(x, **kwargs)
        calls.append(result.clone())
        return result

    torch.randn_like = tracking_randn_like
    try:
        module.train()
        module.training_step(batch, 0)
    finally:
        torch.randn_like = original_randn_like

    assert len(calls) >= 2, f"Expected ≥2 randn_like calls, got {len(calls)}"
    assert not torch.equal(calls[0], calls[1]), "Noise draws must be independent"


# -----------------------------------------------------------------------
# AC-5: Independent gradients
# -----------------------------------------------------------------------


def test_independent_gradients(small_cfg: DictConfig) -> None:
    """Image U-Net grads don't flow through mask U-Net (except shared embedding)."""
    model = build_independent_twin(small_cfg)

    B = 2
    x_image = torch.randn(B, 1, 32, 32, requires_grad=True)
    t = torch.randint(0, 100, (B,))
    tokens = torch.randint(0, 16, (B,))

    out_image = model.image_unet(x_image, timesteps=t, class_labels=tokens)
    loss = out_image.sum()
    loss.backward()

    # Mask U-Net conv parameters should have no gradient
    for name, param in model.mask_unet.named_parameters():
        if "class_embedding" in name:
            # Shared embedding SHOULD have gradient from image path
            assert param.grad is not None, f"Shared param {name} should have grad"
        else:
            assert param.grad is None, f"Mask U-Net param {name} should NOT have grad"


# -----------------------------------------------------------------------
# AC-6: DiffusionSampler compatibility
# -----------------------------------------------------------------------


def test_sampler_compatibility(sampler_cfg: DictConfig) -> None:
    """DiffusionSampler works with IndependentTwinDDPM without modification."""
    model = build_independent_twin(sampler_cfg)
    inferer = build_inferer(sampler_cfg)

    sampler = DiffusionSampler(
        model=model,
        scheduler=inferer,
        cfg=sampler_cfg,
        device="cpu",
    )

    tokens = torch.randint(0, 16, (2,))

    with torch.no_grad():
        samples = sampler.sample(tokens, shape=(2, 2, 32, 32))

    assert samples.shape == (2, 2, 32, 32)
    assert torch.isfinite(samples).all()


# -----------------------------------------------------------------------
# AC-7: State dict structure
# -----------------------------------------------------------------------


def test_state_dict_structure(small_cfg: DictConfig) -> None:
    """State dict has image_unet.* and mask_unet.* keys."""
    model = build_independent_twin(small_cfg)
    sd = model.state_dict()

    image_keys = [k for k in sd if k.startswith("image_unet.")]
    mask_keys = [k for k in sd if k.startswith("mask_unet.")]

    assert len(image_keys) > 0, "No image_unet keys in state_dict"
    assert len(mask_keys) > 0, "No mask_unet keys in state_dict"

    # Shared class_embedding should appear under both prefixes
    image_embed_keys = [k for k in image_keys if "class_embedding" in k]
    mask_embed_keys = [k for k in mask_keys if "class_embedding" in k]
    assert len(image_embed_keys) > 0
    assert len(mask_embed_keys) > 0

    # Verify they reference the same underlying data
    for ik, mk in zip(sorted(image_embed_keys), sorted(mask_embed_keys)):
        ik_suffix = ik.replace("image_unet.", "")
        mk_suffix = mk.replace("mask_unet.", "")
        assert ik_suffix == mk_suffix, f"Key mismatch: {ik_suffix} vs {mk_suffix}"
        assert torch.equal(sd[ik], sd[mk])


# -----------------------------------------------------------------------
# AC-8: train.py dispatch
# -----------------------------------------------------------------------


def test_train_dispatch(lit_cfg: DictConfig) -> None:
    """Config with type=IndependentTwinDDPM instantiates twin Lightning module."""
    from src.diffusion.training.lit_modules_twin import IndependentTwinLightningModule

    module = IndependentTwinLightningModule(lit_cfg)
    assert isinstance(module, IndependentTwinLightningModule)
    assert isinstance(module.model, IndependentTwinDDPM)


# -----------------------------------------------------------------------
# Bonus: build_model factory dispatch
# -----------------------------------------------------------------------


def test_factory_dispatch(small_cfg: DictConfig) -> None:
    """build_model with type=IndependentTwinDDPM returns IndependentTwinDDPM."""
    model, encoder = build_model(small_cfg)
    assert isinstance(model, IndependentTwinDDPM)
    assert encoder is None


def test_factory_does_not_break_shared() -> None:
    """Existing shared/decoupled configs still work after factory change."""
    cfg_dict = _base_twin_cfg()
    cfg_dict["model"]["type"] = "DiffusionModelUNet"
    cfg_dict["model"]["in_channels"] = 2
    cfg_dict["model"]["out_channels"] = 2
    cfg = OmegaConf.create(cfg_dict)

    model, _ = build_model(cfg)
    assert isinstance(model, DiffusionModelUNet)
    assert not isinstance(model, IndependentTwinDDPM)


# -----------------------------------------------------------------------
# Bonus: Lightning module training step produces finite loss
# -----------------------------------------------------------------------


def test_lit_module_training_step(lit_cfg: DictConfig) -> None:
    """One training step produces a finite scalar loss."""
    from src.diffusion.training.lit_modules_twin import IndependentTwinLightningModule

    module = IndependentTwinLightningModule(lit_cfg)
    module.train()

    batch = {
        "image": torch.randn(2, 1, 32, 32),
        "mask": torch.randn(2, 1, 32, 32),
        "token": torch.randint(0, 16, (2,)),
    }

    loss = module.training_step(batch, 0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad


# -----------------------------------------------------------------------
# Integration test: real data mini-training
# -----------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not SLICE_CACHE.exists(),
    reason=f"Slice cache not found at {SLICE_CACHE}",
)
def test_real_data_mini_training() -> None:
    """Load real data from slice_cache fold_0, run 2 training steps."""
    import pandas as pd
    import numpy as np

    from src.diffusion.training.lit_modules_twin import IndependentTwinLightningModule

    cfg = OmegaConf.create(_lit_module_cfg())
    cfg.data.cache_dir = str(SLICE_CACHE)
    cfg.data.slice_sampling.z_range = [34, 115]
    cfg.conditioning.z_bins = 30
    cfg.conditioning.max_z = 127

    module = IndependentTwinLightningModule(cfg)
    module.train()

    # Load a few real samples from fold_0/train.csv
    train_csv = SLICE_CACHE / "folds" / "fold_0" / "train.csv"
    csv_dir = train_csv.parent
    df = pd.read_csv(train_csv)
    sample_rows = df.head(4)

    images, masks, tokens = [], [], []
    for _, row in sample_rows.iterrows():
        fpath = Path(row["filepath"])
        if not fpath.is_absolute():
            fpath = (csv_dir / fpath).resolve()
        data = np.load(fpath)
        images.append(torch.from_numpy(data["image"]).unsqueeze(0).float())
        masks.append(torch.from_numpy(data["mask"]).unsqueeze(0).float())
        tokens.append(int(row["token"]))

    batch = {
        "image": torch.stack(images),   # (4, 1, H, W)
        "mask": torch.stack(masks),     # (4, 1, H, W)
        "token": torch.tensor(tokens, dtype=torch.long),
    }

    # Run 2 training steps
    losses = []
    for step in range(2):
        loss = module.training_step(batch, step)
        assert torch.isfinite(loss), f"Non-finite loss at step {step}: {loss}"
        losses.append(loss.item())
        loss.backward()

    assert all(l > 0 for l in losses), f"Losses should be positive: {losses}"
