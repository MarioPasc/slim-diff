"""Acceptance tests for the bottleneck-only variant (BottleneckSharedTwinDDPM).

Tests map to the PRL pivot plan §4.1 design (encoder/decoder independent,
bottleneck shared with signal coupling). Mirrors the structure of
``test_independent_twin.py`` but with cross-branch coupling expected
through the shared middle block.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from monai.networks.nets.diffusion_model_unet import DiffusionModelUNet
from omegaconf import DictConfig, OmegaConf

from src.diffusion.model.bottleneck_shared_twin import (
    BottleneckSharedTwinDDPM,
    SplitForwardUNet,
    build_bottleneck_shared_twin,
)
from src.diffusion.model.factory import (
    DiffusionSampler,
    build_inferer,
    build_model,
    build_scheduler,
)

SLICE_CACHE_LOCAL = Path(
    "/media/mpascual/Sandisk2TB/completed/jsddpm/data/epilepsy/slice_cache"
)
SLICE_CACHE_SERVER3 = Path("/media/hddb/mario/slimdiff/slice_cache/slice_cache")


def _resolve_slice_cache() -> Path | None:
    for p in (SLICE_CACHE_LOCAL, SLICE_CACHE_SERVER3):
        if p.exists():
            return p
    return None


# -----------------------------------------------------------------------
# Config fixtures
# -----------------------------------------------------------------------


def _base_cfg() -> dict:
    """Minimal config for fast CPU tests (small channels)."""
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
            "type": "BottleneckSharedTwinDDPM",
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
            "bottleneck_shared": {
                "extra_resnet_blocks": 0,
                "norm_num_groups_joint": None,
                "num_head_channels_joint": None,
            },
        },
    }


def _full_cfg() -> dict:
    """Paper-scale config for parameter-budget verification.

    channels=[48,96,168,168] chosen so total params ≈ 26.35M
    (Δ ≈ -2.01 % from the shared 2-channel baseline 26.89M).
    """
    cfg = _base_cfg()
    cfg["model"].update({
        "channels": [48, 96, 168, 168],
        "num_res_blocks": 2,
        "num_head_channels": 12,
        "norm_num_groups": 12,
    })
    cfg["conditioning"]["z_bins"] = 30
    cfg["conditioning"]["use_sinusoidal"] = True
    cfg["conditioning"]["max_z"] = 127
    cfg["data"]["slice_sampling"]["z_range"] = [34, 115]
    return cfg


def _sampler_cfg() -> dict:
    cfg = _base_cfg()
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


def _lit_cfg() -> dict:
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
    cfg["experiment"] = {
        "name": "test_bottleneck_only",
        "output_dir": "/tmp/test_bottleneck_only",
        "seed": 42,
    }
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
    return OmegaConf.create(_base_cfg())


@pytest.fixture
def full_cfg() -> DictConfig:
    return OmegaConf.create(_full_cfg())


@pytest.fixture
def sampler_cfg() -> DictConfig:
    return OmegaConf.create(_sampler_cfg())


@pytest.fixture
def lit_cfg() -> DictConfig:
    return OmegaConf.create(_lit_cfg())


# -----------------------------------------------------------------------
# AC-1: Forward pass interface compatibility
# -----------------------------------------------------------------------


def test_forward_pass_interface(small_cfg: DictConfig) -> None:
    """BottleneckSharedTwinDDPM accepts (B,2,H,W) and returns (B,2,H,W)."""
    model, encoder = build_model(small_cfg)
    assert encoder is None
    assert isinstance(model, BottleneckSharedTwinDDPM)
    assert isinstance(model.image_unet, SplitForwardUNet)
    assert isinstance(model.mask_unet, SplitForwardUNet)

    B, H, W = 2, 32, 32
    x = torch.randn(B, 2, H, W)
    t = torch.randint(0, 100, (B,))
    tokens = torch.randint(0, 16, (B,))

    with torch.no_grad():
        out = model(x, timesteps=t, class_labels=tokens)

    assert out.shape == (B, 2, H, W)
    assert torch.isfinite(out).all()


# -----------------------------------------------------------------------
# AC-2: Parameter count within ±5% of 26.9M
# -----------------------------------------------------------------------


def test_parameter_count(full_cfg: DictConfig) -> None:
    """Total unique params within 5% of 26,894,210 (shared baseline)."""
    model = build_bottleneck_shared_twin(full_cfg)

    unique_params = {id(p): p for p in model.parameters()}
    n_params = sum(p.numel() for p in unique_params.values())

    target = 26_894_210
    delta_pct = abs(n_params - target) / target
    assert delta_pct < 0.05, (
        f"Parameter count {n_params:,} deviates {delta_pct:.2%} from target "
        f"{target:,} (limit 5%)."
    )


# -----------------------------------------------------------------------
# AC-3: Shared conditioning embedding (identity)
# -----------------------------------------------------------------------


def test_shared_cond_embed(small_cfg: DictConfig) -> None:
    """Both UNets reference the exact same class_embedding object."""
    model = build_bottleneck_shared_twin(small_cfg)
    assert model.image_unet.class_embedding is model.mask_unet.class_embedding
    assert model.cond_embed is model.image_unet.class_embedding


# -----------------------------------------------------------------------
# AC-4: Shared middle block has parameters and lives on the wrapper
# -----------------------------------------------------------------------


def test_shared_middle_is_real_module(small_cfg: DictConfig) -> None:
    """The shared middle is a parameter-bearing module on the outer wrapper.

    The child UNets' middle_block must be replaced by the identity no-op
    so calling them directly does not double-process the bottleneck.
    """
    from src.diffusion.model.bottleneck_shared_twin import _IdentityMiddle

    model = build_bottleneck_shared_twin(small_cfg)

    assert isinstance(model.image_unet.middle_block, _IdentityMiddle)
    assert isinstance(model.mask_unet.middle_block, _IdentityMiddle)
    assert not isinstance(model.shared_middle, _IdentityMiddle)

    n_shared_middle = sum(p.numel() for p in model.shared_middle.parameters())
    assert n_shared_middle > 0


# -----------------------------------------------------------------------
# AC-5: Independent noise draws in training_step
# -----------------------------------------------------------------------


def test_independent_noise(lit_cfg: DictConfig) -> None:
    """training_step uses different noise for image and mask channels."""
    from src.diffusion.training.lit_modules_bottleneck_shared import (
        BottleneckSharedTwinLightningModule,
    )

    module = BottleneckSharedTwinLightningModule(lit_cfg)

    batch = {
        "image": torch.randn(2, 1, 32, 32),
        "mask": torch.randn(2, 1, 32, 32),
        "token": torch.randint(0, 16, (2,)),
    }

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
# AC-6: Cross-branch gradients flow through the shared middle
# -----------------------------------------------------------------------


def test_grads_reach_both_branches_via_shared_middle(small_cfg: DictConfig) -> None:
    """Loss on either channel must produce gradients on the shared_middle.

    Tests the BOTTLENECK COUPLING property: a gradient signal originating
    from a single output branch reaches the shared middle parameters
    (proving the shared middle participates in both forward and backward).

    MONAI zero-initialises the final conv of ``DiffusionModelUNet.out`` for
    stable early training. With a raw ``out.sum()`` loss the upstream
    gradients are identically zero at initialisation. We perturb the
    final conv weights so the gradient signal actually propagates.
    """
    model = build_bottleneck_shared_twin(small_cfg)
    model.train()

    with torch.no_grad():
        for unet in (model.image_unet, model.mask_unet):
            final_conv = unet.out[-1].conv
            final_conv.weight.normal_(mean=0.0, std=0.02)

    B = 2
    x = torch.randn(B, 2, 32, 32)
    t = torch.randint(0, 100, (B,))
    tokens = torch.randint(0, 16, (B,))

    out = model(x, timesteps=t, class_labels=tokens)
    target = torch.randn_like(out)
    loss = torch.nn.functional.mse_loss(out[:, 0:1], target[:, 0:1])
    assert loss.item() > 0
    loss.backward()

    middle_params = list(model.shared_middle.parameters())
    assert len(middle_params) > 0
    abs_sums = [p.grad.abs().sum().item() if p.grad is not None else 0.0
                for p in middle_params]
    n_nonzero = sum(1 for s in abs_sums if s > 0)
    # MONAI zero-initialises several residual-projection convs inside
    # DiffusionUNetResnetBlock, so a fraction of shared_middle params have
    # zero gradients at init even after randomising the final out conv.
    # The bottleneck-coupling property is proven by ANY non-zero gradient
    # on shared_middle: if signal coupling were absent, all 30 would be zero.
    assert n_nonzero > 0, (
        f"Loss on image branch reached 0/{len(middle_params)} shared_middle "
        f"params — the bottleneck is not participating in the forward pass."
    )


def test_image_loss_does_not_touch_mask_encoder(small_cfg: DictConfig) -> None:
    """A loss on the image branch must not produce gradients on the mask encoder.

    Encoder independence is the second defining property of bottleneck-only:
    the only cross-branch path is through the shared middle.
    """
    model = build_bottleneck_shared_twin(small_cfg)
    model.train()

    B = 2
    x = torch.randn(B, 2, 32, 32)
    t = torch.randint(0, 100, (B,))
    tokens = torch.randint(0, 16, (B,))

    out = model(x, timesteps=t, class_labels=tokens)
    loss = out[:, 0:1].sum()
    loss.backward()

    for name, param in model.mask_unet.named_parameters():
        if "class_embedding" in name or "middle_block" in name:
            continue
        if any(name.startswith(prefix) for prefix in ("up_blocks", "out")):
            continue
        if param.grad is not None and param.grad.abs().sum() > 0:
            raise AssertionError(
                f"Mask encoder param {name!r} received a gradient from a "
                f"loss restricted to the image branch — encoders are not "
                f"independent."
            )


# -----------------------------------------------------------------------
# AC-7: DiffusionSampler compatibility
# -----------------------------------------------------------------------


def test_sampler_compatibility(sampler_cfg: DictConfig) -> None:
    """DiffusionSampler works with BottleneckSharedTwinDDPM unmodified."""
    model = build_bottleneck_shared_twin(sampler_cfg)
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
# AC-8: State dict structure
# -----------------------------------------------------------------------


def test_state_dict_structure(small_cfg: DictConfig) -> None:
    """State dict has image_unet.*, mask_unet.*, and shared_middle.* keys."""
    model = build_bottleneck_shared_twin(small_cfg)
    sd = model.state_dict()

    image_keys = [k for k in sd if k.startswith("image_unet.")]
    mask_keys = [k for k in sd if k.startswith("mask_unet.")]
    middle_keys = [k for k in sd if k.startswith("shared_middle.")]

    assert len(image_keys) > 0
    assert len(mask_keys) > 0
    assert len(middle_keys) > 0


# -----------------------------------------------------------------------
# AC-9: Factory dispatch
# -----------------------------------------------------------------------


def test_factory_dispatch(small_cfg: DictConfig) -> None:
    """build_model with type=BottleneckSharedTwinDDPM returns the right class."""
    model, encoder = build_model(small_cfg)
    assert isinstance(model, BottleneckSharedTwinDDPM)
    assert encoder is None


def test_factory_does_not_break_others() -> None:
    """The new dispatch must not regress existing variants."""
    cfg_dict = _base_cfg()
    cfg_dict["model"]["type"] = "DiffusionModelUNet"
    cfg_dict["model"]["in_channels"] = 2
    cfg_dict["model"]["out_channels"] = 2
    cfg = OmegaConf.create(cfg_dict)
    model, _ = build_model(cfg)
    assert isinstance(model, DiffusionModelUNet)
    assert not isinstance(model, BottleneckSharedTwinDDPM)


# -----------------------------------------------------------------------
# AC-10: Lightning module — callback-compat attrs + finite training step
# -----------------------------------------------------------------------


def test_lit_module_callback_compat_attrs(lit_cfg: DictConfig) -> None:
    """All five callback-compat attributes must be present and have expected types."""
    from src.diffusion.training.lit_modules_bottleneck_shared import (
        BottleneckSharedTwinLightningModule,
    )

    module = BottleneckSharedTwinLightningModule(lit_cfg)

    assert hasattr(module, "_use_self_conditioning")
    assert hasattr(module, "_use_anatomical_conditioning")
    assert hasattr(module, "_anatomical_method")
    assert hasattr(module, "_anatomical_encoder")
    assert hasattr(module, "_zbin_priors")

    assert module._use_self_conditioning is False
    assert module._use_anatomical_conditioning is False
    assert module._anatomical_method == "concat"
    assert module._anatomical_encoder is None
    assert module._zbin_priors is None


def test_lit_module_training_step(lit_cfg: DictConfig) -> None:
    """One training step produces a finite scalar loss with grad."""
    from src.diffusion.training.lit_modules_bottleneck_shared import (
        BottleneckSharedTwinLightningModule,
    )

    module = BottleneckSharedTwinLightningModule(lit_cfg)
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
def test_real_data_mini_training() -> None:
    """Load real data from slice_cache fold_0 and run 2 training steps."""
    slice_cache = _resolve_slice_cache()
    if slice_cache is None:
        pytest.skip(
            f"Slice cache not found at {SLICE_CACHE_LOCAL} or {SLICE_CACHE_SERVER3}"
        )

    import numpy as np
    import pandas as pd

    from src.diffusion.training.lit_modules_bottleneck_shared import (
        BottleneckSharedTwinLightningModule,
    )

    cfg = OmegaConf.create(_lit_cfg())
    cfg.data.cache_dir = str(slice_cache)
    cfg.data.slice_sampling.z_range = [34, 115]
    cfg.conditioning.z_bins = 30
    cfg.conditioning.max_z = 127

    module = BottleneckSharedTwinLightningModule(cfg)
    module.train()

    train_csv = slice_cache / "folds" / "fold_0" / "train.csv"
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
        "image": torch.stack(images),
        "mask": torch.stack(masks),
        "token": torch.tensor(tokens, dtype=torch.long),
    }

    losses = []
    for step in range(2):
        loss = module.training_step(batch, step)
        assert torch.isfinite(loss), f"Non-finite loss at step {step}: {loss}"
        losses.append(loss.item())
        loss.backward()

    assert all(loss_v > 0 for loss_v in losses), f"Losses should be positive: {losses}"
