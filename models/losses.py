from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.boxes import angle_cosine_loss


def _compute_cost(pred_boxes: torch.Tensor, tgt_boxes: torch.Tensor, cls_logits: torch.Tensor, tgt_labels: torch.Tensor,
                  cls_weight: float, bbox_l1_weight: float, angle_weight: float) -> torch.Tensor:
    """Compute simple pairwise cost matrix for greedy matching.
    pred_boxes: [Q, 5] in [0,1] for first 4 dims
    tgt_boxes:  [M, 5] normalized to [0,1]
    cls_logits: [Q, K+1]
    tgt_labels: [M]
    returns cost: [Q, M]
    """
    Q, M = pred_boxes.size(0), tgt_boxes.size(0)
    if M == 0:
        return torch.zeros((Q, 0), device=pred_boxes.device)

    # Classification cost: negative probability of target class
    prob = F.softmax(cls_logits, dim=-1)  # [Q, K+1]
    tgt_prob = prob[:, tgt_labels]        # broadcast [Q, M]
    cls_cost = -tgt_prob * cls_weight

    # BBox L1 cost (cx,cy,w,h)
    l1 = torch.cdist(pred_boxes[:, :4], tgt_boxes[:, :4], p=1)
    bbox_cost = l1 * bbox_l1_weight

    # Angle cost (1 - cos delta)
    pred_theta = pred_boxes[:, 4:5]
    tgt_theta = tgt_boxes[:, 4:5].T  # [1, M]
    ang = angle_cosine_loss(pred_theta, tgt_theta)
    angle_cost = ang * angle_weight

    return cls_cost + bbox_cost + angle_cost


def greedy_match(cost: torch.Tensor) -> List[Tuple[int, int]]:
    """Greedy bipartite matching based on minimal cost. Returns list of (q_idx, m_idx)."""
    if cost.numel() == 0 or cost.shape[1] == 0:
        return []
    Q, M = cost.shape
    matched: List[Tuple[int, int]] = []
    used_q = set()
    used_m = set()
    # Flatten all pairs sorted by cost
    vals, idxs = torch.sort(cost.flatten())
    for idx in idxs.tolist():
        q = idx // M
        m = idx % M
        if q in used_q or m in used_m:
            continue
        matched.append((q, m))
        used_q.add(q)
        used_m.add(m)
        if len(matched) == min(Q, M):
            break
    return matched


class OrientedSetCriterion(nn.Module):
    def __init__(self, num_classes: int, cls_weight: float = 2.0, bbox_l1_weight: float = 5.0,
                 angle_weight: float = 2.0, no_object_weight: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.bbox_l1_weight = bbox_l1_weight
        self.angle_weight = angle_weight
        self.no_object_weight = no_object_weight

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        logits = outputs["pred_logits"]  # [B,Q,K+1]
        boxes = outputs["pred_boxes"]    # [B,Q,5]

        B, Q, _ = boxes.shape
        device = boxes.device

        loss_cls_sum = torch.tensor(0., device=device)
        loss_bbox_sum = torch.tensor(0., device=device)
        loss_ang_sum = torch.tensor(0., device=device)

        for b in range(B):
            tgt = targets[b]
            tgt_boxes = tgt["boxes"]  # [M,5] absolute pixels
            tgt_labels = tgt["labels"] - 1  # convert 1..K -> 0..K-1
            H, W = tgt["size"].tolist()

            if tgt_boxes.numel() > 0:
                # Normalize to [0,1]
                norm = tgt_boxes.clone()
                norm[:, 0] /= W
                norm[:, 1] /= H
                norm[:, 2] /= W
                norm[:, 3] /= H
            else:
                norm = tgt_boxes

            cost = _compute_cost(
                boxes[b], norm, logits[b], tgt_labels,
                self.cls_weight, self.bbox_l1_weight, self.angle_weight
            )
            matches = greedy_match(cost)

            # Classification targets
            tgt_cls = torch.full((Q,), self.num_classes, dtype=torch.long, device=device)  # no-object index
            if len(matches) > 0:
                for qi, mi in matches:
                    tgt_cls[qi] = tgt_labels[mi]

            ce_weights = torch.ones(self.num_classes + 1, device=device)
            ce_weights[self.num_classes] = self.no_object_weight
            loss_cls = F.cross_entropy(logits[b].transpose(1, 2), tgt_cls.unsqueeze(0).expand_as(logits[b][..., 0]),
                                       weight=ce_weights)
            loss_cls_sum += loss_cls

            if len(matches) > 0:
                q_idx = torch.tensor([m[0] for m in matches], dtype=torch.long, device=device)
                m_idx = torch.tensor([m[1] for m in matches], dtype=torch.long, device=device)

                pb = boxes[b][q_idx, :4]
                tb = norm[m_idx, :4]
                loss_bbox = F.l1_loss(pb, tb, reduction='mean') * self.bbox_l1_weight

                pth = boxes[b][q_idx, 4]
                tth = norm[m_idx, 4]
                loss_ang = angle_cosine_loss(pth, tth).mean() * self.angle_weight
            else:
                loss_bbox = torch.tensor(0., device=device)
                loss_ang = torch.tensor(0., device=device)

            loss_bbox_sum += loss_bbox
            loss_ang_sum += loss_ang

        losses = {
            "loss_cls": loss_cls_sum / B,
            "loss_bbox": loss_bbox_sum / B,
            "loss_angle": loss_ang_sum / B,
        }
        losses["loss_total"] = losses["loss_cls"] + losses["loss_bbox"] + losses["loss_angle"]
        return losses

