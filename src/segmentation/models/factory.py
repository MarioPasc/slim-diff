"""Model factory for MONAI segmentation models."""

from __future__ import annotations

import logging

from monai.networks.nets import BasicUNetPlusPlus, DynUNet, SwinUNETR, UNet
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_model(cfg: DictConfig):
    """Build segmentation model from config.

    Args:
        cfg: Configuration with cfg.model section

    Returns:
        MONAI segmentation model

    Raises:
        ValueError: If model name is unknown
    """
    model_name = cfg.model.name.lower()
    model_cfg = cfg.model

    logger.info(f"Building model: {model_name}")

    if model_name == "unet":
        model = UNet(
            spatial_dims=model_cfg.spatial_dims,
            in_channels=model_cfg.in_channels,
            out_channels=model_cfg.out_channels,
            channels=tuple(model_cfg.channels),
            strides=tuple(model_cfg.strides),
            num_res_units=model_cfg.get("num_res_units", 0),
            norm=model_cfg.get("norm", "batch"),
        )

    elif model_name == "dynunet":
        model = DynUNet(
            spatial_dims=model_cfg.spatial_dims,
            in_channels=model_cfg.in_channels,
            out_channels=model_cfg.out_channels,
            kernel_size=model_cfg.kernel_size,
            strides=model_cfg.strides,
            upsample_kernel_size=model_cfg.upsample_kernel_size,
            filters=model_cfg.get("filters", None),
            dropout=model_cfg.get("dropout", 0.0),
            norm_name=model_cfg.get("norm_name", "instance"),
            act_name=model_cfg.get("act_name", "leakyrelu"),
            deep_supervision=model_cfg.get("deep_supervision", False),
            deep_supr_num=model_cfg.get("deep_supr_num", 1),
            res_block=model_cfg.get("res_block", True),
            trans_bias=model_cfg.get("trans_bias", False),
        )

    elif model_name in ("unetplusplus", "basicunetplusplus"):
        model = BasicUNetPlusPlus(
            spatial_dims=model_cfg.spatial_dims,
            in_channels=model_cfg.in_channels,
            out_channels=model_cfg.out_channels,
            features=tuple(model_cfg.features),
            deep_supervision=model_cfg.get("deep_supervision", False),
            act=model_cfg.get("act", "relu"),
            norm=model_cfg.get("norm", "batch"),
            bias=model_cfg.get("bias", True),
            dropout=model_cfg.get("dropout", 0.0),
            upsample=model_cfg.get("upsample", "deconv"),
        )

    elif model_name == "swinunetr":
        model = SwinUNETR(
            img_size=tuple(model_cfg.img_size),
            in_channels=model_cfg.in_channels,
            out_channels=model_cfg.out_channels,
            depths=tuple(model_cfg.depths),
            num_heads=tuple(model_cfg.num_heads),
            feature_size=model_cfg.feature_size,
            norm_name=model_cfg.get("norm_name", "instance"),
            drop_rate=model_cfg.get("drop_rate", 0.0),
            attn_drop_rate=model_cfg.get("attn_drop_rate", 0.0),
            dropout_path_rate=model_cfg.get("dropout_path_rate", 0.0),
            normalize=model_cfg.get("normalize", True),
            use_checkpoint=model_cfg.get("use_checkpoint", False),
            spatial_dims=model_cfg.spatial_dims,
            downsample=model_cfg.get("downsample", "merging"),
            use_v2=model_cfg.get("use_v2", False),
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Log model info
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"Built {model_name}: {n_params:,} params ({n_trainable:,} trainable)"
    )

    return model
