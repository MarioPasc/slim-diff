"""One-off param-count probe for BottleneckSharedTwinDDPM channel tuning.

Run on server3 (where the slimdiff conda env lives) to pick a
``model.channels`` tuple that lands within ±5% of 26.9M params.
"""
from __future__ import annotations

from omegaconf import OmegaConf

from src.diffusion.model.bottleneck_shared_twin import build_bottleneck_shared_twin

TARGET = 26_894_210


def build_cfg(channels, num_res_blocks=2, num_head_channels=20, norm_num_groups=12, extra=0):
    return OmegaConf.create({
        "model": {
            "type": "BottleneckSharedTwinDDPM",
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": list(channels),
            "attention_levels": [False, False, True, True],
            "num_res_blocks": num_res_blocks,
            "num_head_channels": num_head_channels,
            "norm_num_groups": norm_num_groups,
            "use_class_embedding": True,
            "dropout": 0.0,
            "resblock_updown": False,
            "with_conditioning": False,
            "bottleneck_shared": {"extra_resnet_blocks": extra},
        },
        "conditioning": {
            "z_bins": 30,
            "use_sinusoidal": True,
            "max_z": 127,
            "cfg": {"enabled": False, "null_token": 60, "dropout_prob": 0.1},
        },
        "data": {"slice_sampling": {"z_range": [34, 115]}},
    })


CANDIDATES = [
    ([42, 84, 168, 168], 2, 14, 14, 0),
    ([42, 84, 168, 168], 2, 14, 14, 1),
    ([42, 84, 168, 168], 2, 14, 14, 2),
    ([48, 96, 168, 168], 2, 12, 12, 0),
    ([48, 96, 168, 168], 2, 14, 12, 0),
    ([42, 84, 180, 180], 2, 12, 6, 0),
    ([45, 90, 180, 180], 2, 18, 9, 0),
    ([45, 90, 180, 180], 2, 15, 9, 0),
    ([44, 88, 176, 176], 2, 16, 4, 0),
    ([44, 88, 176, 176], 2, 22, 4, 0),
    ([48, 96, 176, 176], 2, 16, 16, 0),
    ([48, 96, 176, 176], 2, 16, 8, 0),
]


def main() -> int:
    for ch, rb, nh, ng, ex in CANDIDATES:
        try:
            cfg = build_cfg(ch, rb, nh, ng, ex)
            m = build_bottleneck_shared_twin(cfg)
            unique = {id(p): p for p in m.parameters()}
            n = sum(p.numel() for p in unique.values())
            sm = sum(p.numel() for p in m.shared_middle.parameters())
            delta = (n - TARGET) / TARGET * 100
            print(
                f"channels={ch} rb={rb} nh={nh} ng={ng} ex={ex}: "
                f"total={n:,} ({delta:+.2f}%) | shared_middle={sm:,}"
            )
        except Exception as e:
            print(
                f"channels={ch} rb={rb} nh={nh} ng={ng} ex={ex}: "
                f"ERROR {type(e).__name__}: {e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
