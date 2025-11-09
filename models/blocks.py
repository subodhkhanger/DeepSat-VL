from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            in_d = input_dim if i == 0 else hidden_dim
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_dim if num_layers > 1 else input_dim, output_dim)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PositionEmbeddingSine(nn.Module):
    """2D sine-cosine positional encoding, like DETR."""

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale or 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def build_backbone(name: str, hidden_dim: int):
    """Build a simple vision backbone returning a feature map and a projection to hidden_dim.
    name: open_clip_vit_b_16 | timm_vit_base_patch16_224 | torchvision_resnet50
    """
    name = (name or "").lower()
    if name.startswith("open_clip_vit"):
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained=None)
            # Use visual stem only
            vision = model.visual
            # Construct a wrapper to get a 2D feature map via patch embeddings
            class CLIPViTWrapper(nn.Module):
                def __init__(self, v, proj_dim):
                    super().__init__()
                    self.v = v
                    self.proj = nn.Conv2d(v.conv1.out_channels, proj_dim, kernel_size=1)

                def forward(self, x):
                    # x: [B,3,H,W]
                    x = self.v.conv1(x)  # [B, C, H/patch, W/patch]
                    return self.proj(x)

            return CLIPViTWrapper(vision, hidden_dim)
        except Exception:
            pass

    if name.startswith("timm_vit"):
        try:
            import timm
            vit = timm.create_model("vit_base_patch16_224", pretrained=False, features_only=True, out_indices=[-1])
            proj = nn.Conv2d(vit.feature_info[-1]['num_chs'], hidden_dim, kernel_size=1)

            class TimmViTWrapper(nn.Module):
                def __init__(self, v, p):
                    super().__init__()
                    self.v = v
                    self.p = p

                def forward(self, x):
                    feats = self.v(x)[-1]
                    return self.p(feats)

            return TimmViTWrapper(vit, proj)
        except Exception:
            pass

    # Fallback: torchvision resnet50
    import torchvision.models as tvm
    backbone = tvm.resnet50(weights=None)
    modules = list(backbone.children())[:-2]
    body = nn.Sequential(*modules)
    proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

    class ResNetWrapper(nn.Module):
        def __init__(self, b, p):
            super().__init__()
            self.b = b
            self.p = p

        def forward(self, x):
            x = self.b(x)
            x = self.p(x)
            return x

    return ResNetWrapper(body, proj)
