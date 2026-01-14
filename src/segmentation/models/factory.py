"""Model factory for MONAI segmentation models."""

from __future__ import annotations

import logging

from omegaconf import DictConfig
from torch import nn

logger = logging.getLogger(__name__)


def build_model(cfg: DictConfig) -> nn.Module:
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

    model: nn.Module

    if model_name == "unet":
        from monai.networks.nets.unet import UNet
        
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
        from monai.networks.nets.dynunet import DynUNet
        
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
        from monai.networks.nets.basic_unetplusplus import BasicUNetPlusPlus

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
        from monai.networks.nets.swin_unetr import SwinUNETR

        model = SwinUNETR(
            #img_size=tuple(model_cfg.img_size),
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
    elif model_name == "segresnet":
        from monai.networks.nets.segresnet import SegResNet
        model = SegResNet(
            spatial_dims=model_cfg.spatial_dims,
            in_channels=model_cfg.in_channels,
            out_channels=model_cfg.out_channels,
            init_filters=model_cfg.get("init_filters", 8),
            dropout_prob=model_cfg.get("dropout_prob", None),
            blocks_down=tuple(model_cfg.get("blocks_down", [1, 2, 2, 4])),
            blocks_up=tuple(model_cfg.get("blocks_up", [1, 1, 1])),
        )

    elif model_name == "attentionunet":
        from monai.networks.nets.attentionunet import AttentionUnet
        model = AttentionUnet(
            spatial_dims=model_cfg.spatial_dims,
            in_channels=model_cfg.in_channels,
            out_channels=model_cfg.out_channels,
            channels=tuple(model_cfg.channels),
            strides=tuple(model_cfg.strides),
            kernel_size=model_cfg.get("kernel_size", 3),
            up_kernel_size=model_cfg.get("up_kernel_size", 3),
            dropout=model_cfg.get("dropout", 0.0),
        )

    elif model_name == "unetr":
        from monai.networks.nets.unetr import UNETR
        model = UNETR(
            in_channels=model_cfg.in_channels,
            out_channels=model_cfg.out_channels,
            img_size=tuple(model_cfg.img_size),
            feature_size=model_cfg.get("feature_size", 16),
            hidden_size=model_cfg.get("hidden_size", 768),
            mlp_dim=model_cfg.get("mlp_dim", 3072),
            num_heads=model_cfg.get("num_heads", 12),
            proj_type=model_cfg.get("proj_type", "perceptron"),
            norm_name=model_cfg.get("norm_name", "instance"),
            res_block=model_cfg.get("res_block", True),
            dropout_rate=model_cfg.get("dropout_rate", 0.0),
            spatial_dims=model_cfg.spatial_dims,
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
