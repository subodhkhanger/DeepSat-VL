from __future__ import annotations

import math
from typing import Tuple

import torch


def rbox_to_polygon(cx: torch.Tensor, cy: torch.Tensor, w: torch.Tensor, h: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Convert rotated boxes (center, size, angle in radians) to polygons.
    Returns tensor of shape [N, 4, 2] (x,y points).
    """
    # local corners before rotation
    dx = w / 2.0
    dy = h / 2.0
    corners = torch.stack([
        torch.stack([-dx, -dy], dim=-1),
        torch.stack([ dx, -dy], dim=-1),
        torch.stack([ dx,  dy], dim=-1),
        torch.stack([-dx,  dy], dim=-1),
    ], dim=1)  # [N, 4, 2]

    cos_t = torch.cos(theta).unsqueeze(-1)
    sin_t = torch.sin(theta).unsqueeze(-1)
    rot = torch.stack([
        torch.cat([cos_t, -sin_t], dim=-1),
        torch.cat([sin_t,  cos_t], dim=-1)
    ], dim=1)  # [N, 2, 2]

    rotated = torch.einsum('nij,nkj->nki', rot, corners)
    center = torch.stack([cx, cy], dim=-1).unsqueeze(1)
    poly = rotated + center
    return poly


def box_cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    """box: [..., 4] (cx, cy, w, h) -> [..., 4] (x1,y1,x2,y2)"""
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Minimal angular difference in radians, result in [-pi, pi]."""
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi


def angle_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cos(angle_error)."""
    return 1.0 - torch.cos(angle_diff(pred, target))

