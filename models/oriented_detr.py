from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import PositionEmbeddingSine, MLP, build_backbone


class OrientedDETR(nn.Module):
    """
    Minimal DETR-style model predicting rotated boxes (cx, cy, w, h, theta) and class logits.
    - Backbone -> feature map -> positional encoding -> Transformer encoder-decoder
    - N object queries decoded into predictions
    """

    def __init__(self, num_classes: int, backbone: str = "open_clip_vit_b_16", hidden_dim: int = 256,
                 num_queries: int = 100, nheads: int = 8, enc_layers: int = 4, dec_layers: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = build_backbone(backbone, hidden_dim)
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Identity()  # backbone already projects to hidden_dim

        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_head = MLP(hidden_dim, hidden_dim, 5, 3)  # (cx, cy, w, h, theta)

    def forward(self, images: torch.Tensor):
        # images: [B,3,H,W]
        feats = self.backbone(images)  # [B, C, H', W'] with C=hidden_dim
        pos = self.pos_embed(feats)    # [B, C, H', W']

        B, C, Hp, Wp = feats.shape
        src = feats.flatten(2).permute(0, 2, 1)  # [B, L, C]
        pos = pos.flatten(2).permute(0, 2, 1)    # [B, L, C]

        memory = self.encoder(src + pos)         # [B, L, C]
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, C]
        hs = self.decoder(queries, memory)       # [B, Q, C]

        logits = self.class_head(hs)             # [B, Q, K+1]
        boxes = self.bbox_head(hs)               # [B, Q, 5]

        # Normalize center and size to [0,1] relative to input image size; theta in radians unconstrained
        boxes_sig = boxes.clone()
        boxes_sig[..., :4] = boxes_sig[..., :4].sigmoid()
        return {"pred_logits": logits, "pred_boxes": boxes_sig}

