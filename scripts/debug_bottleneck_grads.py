"""Diagnostic: trace gradient flow through the shared middle block."""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.diffusion.model.bottleneck_shared_twin import build_bottleneck_shared_twin


def main() -> None:
    cfg = OmegaConf.create({
        "data": {
            "cache_dir": "/tmp",
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
            "type": "BottleneckSharedTwinDDPM",
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": [16, 32, 64, 64],
            "attention_levels": [False, False, True, True],
            "num_res_blocks": 1,
            "num_head_channels": 16,
            "norm_num_groups": 8,
            "use_class_embedding": True,
            "dropout": 0.0,
            "resblock_updown": False,
            "with_conditioning": False,
            "bottleneck_shared": {
                "extra_resnet_blocks": 0,
                "norm_num_groups_joint": None,
                "num_head_channels_joint": None,
            },
        },
    })

    model = build_bottleneck_shared_twin(cfg)
    model.train()
    print("shared_middle children:", list(model.shared_middle._modules.keys()))
    for n, p in model.shared_middle.named_parameters():
        print(" ", n, tuple(p.shape), "requires_grad=", p.requires_grad)

    x = torch.randn(2, 2, 32, 32)
    t = torch.randint(0, 100, (2,))
    tok = torch.randint(0, 16, (2,))

    orig_fwd = model.shared_middle.forward
    captured: dict = {}

    def captured_fwd(hidden_states, temb, context=None):
        captured["in"] = hidden_states
        out = orig_fwd(hidden_states, temb, context)
        captured["out"] = out
        return out

    model.shared_middle.forward = captured_fwd

    out = model(x, timesteps=t, class_labels=tok)
    print("out shape:", tuple(out.shape), "requires_grad:", out.requires_grad)
    print("shared_middle.in  requires_grad:", captured["in"].requires_grad)
    print("shared_middle.out requires_grad:", captured["out"].requires_grad)
    print("shared_middle.in is same object as out (identity?)", captured["in"] is captured["out"])

    loss = out[:, 0:1].sum()
    print("loss:", loss.item(), "requires_grad:", loss.requires_grad)
    loss.backward()

    print("=== shared_middle grads ===")
    for n, p in model.shared_middle.named_parameters():
        g = p.grad
        if g is None:
            print(f"  {n}: NONE")
        else:
            print(
                f"  {n}: norm={g.norm().item():.3e} abs_sum={g.abs().sum().item():.3e}"
            )

    # Sanity: image_unet.out param should have grads
    print("=== image_unet.out.0 (last conv) grad sanity ===")
    for n, p in model.image_unet.out.named_parameters():
        g = p.grad
        if g is None:
            print(f"  {n}: NONE")
        else:
            print(f"  {n}: norm={g.norm().item():.3e}")


if __name__ == "__main__":
    main()
