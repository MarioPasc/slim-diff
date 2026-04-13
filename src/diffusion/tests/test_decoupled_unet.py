"""Acceptance tests for the decoupled-bottleneck variant (TASK-01).

The tests below map 1:1 to the 6 acceptance criteria in
``docs/icip2026/rebuttal_plans/TASK_01_ARCH_decoupled_bottleneck.md`` and
add two bonus tests (cross-attention path, numerical sanity). They are
CPU-only and avoid the heavy data / logging stack: they only exercise
``build_model`` and the model's ``forward``.
"""

from __future__ import annotations

import copy
import warnings

import pytest
import torch
from monai.networks.nets.diffusion_model_unet import (
    AttnMidBlock,
    CrossAttnMidBlock,
    DiffusionModelUNet,
)
from omegaconf import OmegaConf

from src.diffusion.model.decoupled_unet import DecoupledMiddleBlock
from src.diffusion.model.factory import build_model


# -----------------------------------------------------------------------
# Config fixtures
# -----------------------------------------------------------------------


def _base_cfg() -> dict:
    """Minimal config supporting ``build_model`` (no cfg/anatomical)."""
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
        "training": {"self_conditioning": {"enabled": False, "probability": 0.5}},
        "model": {
            "type": "DiffusionModelUNet",
            "spatial_dims": 2,
            "in_channels": 2,
            "out_channels": 2,
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


@pytest.fixture
def small_config_shared():
    cfg = OmegaConf.create(_base_cfg())
    cfg.model.bottleneck_mode = "shared"
    return cfg


@pytest.fixture
def small_config_decoupled():
    cfg = OmegaConf.create(_base_cfg())
    cfg.model.bottleneck_mode = "decoupled"
    # Default channels_per_path = 32 (half of channels[-1]=64); 32 % 8 == 0 OK.
    return cfg


def _full_cfg() -> dict:
    base = _base_cfg()
    base["data"]["transforms"]["roi_size"] = [160, 160, 160]
    base["data"]["slice_sampling"]["z_range"] = [34, 115]
    base["conditioning"]["z_bins"] = 30
    base["conditioning"]["max_z"] = 159
    base["model"]["channels"] = [64, 128, 256, 256]
    base["model"]["attention_levels"] = [False, False, True, True]
    base["model"]["num_res_blocks"] = 2
    base["model"]["num_head_channels"] = 32
    base["model"]["norm_num_groups"] = 32
    return base


@pytest.fixture
def full_config_shared():
    cfg = OmegaConf.create(_full_cfg())
    cfg.model.bottleneck_mode = "shared"
    return cfg


@pytest.fixture
def full_config_decoupled():
    cfg = OmegaConf.create(_full_cfg())
    cfg.model.bottleneck_mode = "decoupled"
    # Camera-ready knobs live here; default channels_per_path=128, no extras.
    return cfg


# -----------------------------------------------------------------------
# AC-1 — forward pass shapes
# -----------------------------------------------------------------------


def test_forward_pass_shapes_shared_and_decoupled(
    small_config_shared, small_config_decoupled
):
    """Both variants accept the same inputs and return the same shape."""
    torch.manual_seed(0)
    model_s, _ = build_model(small_config_shared)
    model_d, _ = build_model(small_config_decoupled)

    B, H, W = 2, 32, 32
    x = torch.randn(B, 2, H, W)
    t = torch.randint(0, 1000, (B,), dtype=torch.long)
    labels = torch.randint(0, 2 * small_config_shared.conditioning.z_bins, (B,))

    model_s.eval()
    model_d.eval()
    with torch.no_grad():
        out_s = model_s(x, timesteps=t, class_labels=labels)
        out_d = model_d(x, timesteps=t, class_labels=labels)

    assert out_s.shape == (B, 2, H, W)
    assert out_d.shape == (B, 2, H, W)


# -----------------------------------------------------------------------
# AC-2 — parameter count within tolerance
# -----------------------------------------------------------------------


def test_parameter_count_within_tolerance(
    full_config_shared, full_config_decoupled
):
    """Full paper config: |Δ|/N_shared must stay below 5 % (hard cap) and
    we warn if it exceeds the 1 % target."""
    model_s, _ = build_model(full_config_shared)
    model_d, _ = build_model(full_config_decoupled)

    n_s = sum(p.numel() for p in model_s.parameters())
    n_d = sum(p.numel() for p in model_d.parameters())
    ratio = abs(n_s - n_d) / max(n_s, 1)

    print(
        f"[param-count] shared={n_s:,} decoupled={n_d:,} "
        f"Δ={n_d - n_s:+,} ({ratio:.4%})"
    )
    if ratio > 0.01:
        warnings.warn(
            f"Decoupled params deviate by {ratio:.2%} from shared "
            f"(>1% target). Tune model.decoupled_bottleneck.extra_resnet_blocks"
            f" or channels_per_path before the camera-ready run.",
            UserWarning,
            stacklevel=2,
        )
    assert ratio < 0.05, (
        f"Decoupled variant exceeds the 5% hard cap "
        f"(shared={n_s:,}, decoupled={n_d:,}, Δ={ratio:.2%})."
    )


# -----------------------------------------------------------------------
# AC-3 — gradient flow through both independent paths
# -----------------------------------------------------------------------


def _defuse_zero_inits(model: torch.nn.Module) -> None:
    """MONAI initializes the final output conv and every ResBlock's conv2 /
    attention proj_out with zero weights (see ``zero_module`` in
    ``diffusion_model_unet.py``). At initialisation this blocks ALL
    gradient flow upstream of the terminal output conv, so we perturb
    those weights slightly before running the gradient-flow test. This
    mirrors what a single optimisation step would do during real
    training and gives the test a meaningful signal."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if (
                name.endswith("conv2.conv.weight")
                or name.endswith("proj_out.conv.weight")
                or name.startswith("out.")
            ):
                if p.dim() >= 2:
                    p.add_(torch.randn_like(p) * 1e-2)


def test_gradient_flow_decoupled(small_config_decoupled):
    """Every decoupled-middle-block parameter receives a non-zero gradient.

    We first defuse MONAI's zero-init on terminal output layers so
    gradient can actually reach the middle block (at strict init, the
    final output conv is zero-weighted and blocks every upstream grad).
    """
    torch.manual_seed(1)
    model, _ = build_model(small_config_decoupled)
    _defuse_zero_inits(model)
    model.train()

    B, H, W = 1, 32, 32
    x = torch.randn(B, 2, H, W, requires_grad=False)
    t = torch.randint(0, 1000, (B,), dtype=torch.long)
    labels = torch.randint(0, 2 * small_config_decoupled.conditioning.z_bins, (B,))

    out = model(x, timesteps=t, class_labels=labels)
    loss = out.sum()
    loss.backward()

    middle_named = {
        name: p
        for name, p in model.named_parameters()
        if name.startswith("middle_block.") and p.requires_grad
    }
    assert middle_named, "No middle_block params found."
    zero_grad = [
        name for name, p in middle_named.items()
        if p.grad is None or p.grad.abs().sum().item() == 0.0
    ]
    assert not zero_grad, f"Params with zero/None grad: {zero_grad[:5]}"

    # Both paths plus the split/merge projections must have received gradient.
    required_prefixes = (
        "middle_block.path_a.",
        "middle_block.path_b.",
        "middle_block.proj_split.",
        "middle_block.proj_merge.",
    )
    for pfx in required_prefixes:
        hits = [n for n in middle_named if n.startswith(pfx)]
        assert hits, f"No params found under {pfx}"


# -----------------------------------------------------------------------
# AC-4 — path independence (no weight sharing)
# -----------------------------------------------------------------------


def test_path_parameter_independence(small_config_decoupled):
    """path_a and path_b must hold disjoint parameter objects."""
    model, _ = build_model(small_config_decoupled)

    path_a_ids: set[int] = set()
    path_b_ids: set[int] = set()
    for name, p in model.named_parameters():
        if name.startswith("middle_block.path_a."):
            path_a_ids.add(id(p))
        elif name.startswith("middle_block.path_b."):
            path_b_ids.add(id(p))

    assert path_a_ids, "No path_a parameters found."
    assert path_b_ids, "No path_b parameters found."
    assert path_a_ids.isdisjoint(path_b_ids), (
        "Decoupled paths share parameters — they must be independent."
    )


# -----------------------------------------------------------------------
# AC-5 — default config remains shared (backward compatibility)
# -----------------------------------------------------------------------


def test_default_config_shared():
    """Omitting bottleneck_mode yields the stock MONAI middle block."""
    cfg = OmegaConf.create(_base_cfg())
    assert "bottleneck_mode" not in cfg.model
    model, _ = build_model(cfg)

    assert isinstance(model, DiffusionModelUNet)
    # Stock mid block is AttnMidBlock (with_conditioning=False in _base_cfg).
    assert isinstance(model.middle_block, AttnMidBlock)
    assert not isinstance(model.middle_block, DecoupledMiddleBlock)


# -----------------------------------------------------------------------
# AC-6 — shared-mode state_dict keys are unchanged
# -----------------------------------------------------------------------


def test_state_dict_keys_unchanged_in_shared_mode():
    """Explicit bottleneck_mode='shared' must not alter state_dict keys."""
    cfg_plain = OmegaConf.create(_base_cfg())
    cfg_shared = OmegaConf.create(_base_cfg())
    cfg_shared.model.bottleneck_mode = "shared"

    model_plain, _ = build_model(cfg_plain)
    model_shared, _ = build_model(cfg_shared)

    keys_plain = set(model_plain.state_dict().keys())
    keys_shared = set(model_shared.state_dict().keys())
    assert keys_plain == keys_shared, (
        "Explicitly setting bottleneck_mode='shared' changed state_dict keys."
    )

    # And none of those keys reference decoupled-only submodules.
    for k in keys_shared:
        assert "path_a" not in k and "path_b" not in k, (
            f"Shared mode should not expose decoupled keys: {k}"
        )


# -----------------------------------------------------------------------
# Bonus — cross-attention path with decoupled mode
# -----------------------------------------------------------------------


def test_decoupled_cross_attention_path():
    """Unit-level: DecoupledMiddleBlock forwards ``context`` to both paths
    when ``with_conditioning=True``. (Covering this at the factory level
    would require wiring the anatomical encoder, which is TASK-scope
    orthogonal; the unit test below exercises the exact code path a
    ``with_conditioning=True`` UNet would hit on its middle block.)"""
    torch.manual_seed(3)
    in_channels = 64
    temb_channels = 64
    cross_attention_dim = 32

    block = DecoupledMiddleBlock(
        spatial_dims=2,
        in_channels=in_channels,
        temb_channels=temb_channels,
        norm_num_groups=8,
        norm_eps=1e-6,
        with_conditioning=True,
        num_head_channels=16,  # must divide channels_per_path=32
        transformer_num_layers=1,
        cross_attention_dim=cross_attention_dim,
    )

    B, H, W = 2, 16, 16
    x = torch.randn(B, in_channels, H, W)
    temb = torch.randn(B, temb_channels)
    ctx = torch.randn(B, 4, cross_attention_dim)

    block.eval()
    with torch.no_grad():
        out = block(x, temb=temb, context=ctx)
    assert out.shape == (B, in_channels, H, W)
    assert torch.isfinite(out).all()


# -----------------------------------------------------------------------
# Bonus — numerical sanity
# -----------------------------------------------------------------------


def test_decoupled_numerical_sanity(small_config_decoupled):
    """Decoupled model produces finite outputs (no NaN/Inf)."""
    torch.manual_seed(2)
    model, _ = build_model(small_config_decoupled)
    model.eval()
    B, H, W = 2, 32, 32
    x = torch.randn(B, 2, H, W)
    t = torch.randint(0, 1000, (B,), dtype=torch.long)
    labels = torch.randint(0, 2 * small_config_decoupled.conditioning.z_bins, (B,))

    with torch.no_grad():
        out = model(x, timesteps=t, class_labels=labels)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf."


# -----------------------------------------------------------------------
# Bonus — unknown bottleneck_mode is rejected
# -----------------------------------------------------------------------


def test_unknown_bottleneck_mode_rejected():
    cfg = OmegaConf.create(_base_cfg())
    cfg.model.bottleneck_mode = "frobnicate"
    with pytest.raises(ValueError, match="bottleneck_mode"):
        build_model(cfg)
