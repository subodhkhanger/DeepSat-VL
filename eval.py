#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from datasets import DotaRboxDataset
from models import OrientedDETR
from utils.boxes import angle_diff


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = [b["target"] for b in batch]
    paths = [b["path"] for b in batch]
    return images, targets, paths


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Device selection: prioritize CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    ds_cfg = cfg["dataset"]
    val_ds = DotaRboxDataset(ds_cfg["root"], ds_cfg["val_ann"], ds_cfg.get("img_dir", "images"), ds_cfg.get("img_size", None))
    ld = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    m_cfg = cfg["model"]
    model = OrientedDETR(
        num_classes=ds_cfg["num_classes"],
        backbone=m_cfg.get("backbone", "open_clip_vit_b_16"),
        hidden_dim=m_cfg.get("hidden_dim", 256),
        num_queries=m_cfg.get("num_queries", 100),
        nheads=m_cfg.get("nheads", 8),
        enc_layers=m_cfg.get("enc_layers", 4),
        dec_layers=m_cfg.get("dec_layers", 4),
        dim_feedforward=m_cfg.get("dim_feedforward", 1024),
        dropout=m_cfg.get("dropout", 0.1),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    total = 0
    angle_err_sum = 0.0
    for images, targets, paths in ld:
        images = images.to(device)
        out = model(images)
        logits = out["pred_logits"][0]   # [Q,K+1]
        boxes = out["pred_boxes"][0]     # [Q,5]
        probs = torch.softmax(logits, dim=-1)
        scores, labels = probs.max(dim=-1)

        # naive selection: top-scoring predictions over threshold
        conf_thr = float(cfg.get("eval", {}).get("conf_threshold", 0.3))
        keep = (labels < ds_cfg["num_classes"]) & (scores > conf_thr)
        pred = boxes[keep]
        plab = labels[keep]
        psc = scores[keep]

        tgt = targets[0]
        tboxes = tgt["boxes"]
        tlab = tgt["labels"] - 1

        # Compute angle MAE on a naive 1-1 greedy match by label
        matched = 0
        for c in range(ds_cfg["num_classes"]):
            p_mask = plab == c
            t_mask = tlab == c
            if p_mask.any() and t_mask.any():
                p_theta = pred[p_mask][:, 4]
                t_theta = tboxes[t_mask][:, 4]
                # compare min count
                m = min(p_theta.numel(), t_theta.numel())
                if m > 0:
                    d = angle_diff(p_theta[:m], t_theta[:m]).abs().mean().item()
                    angle_err_sum += d
                    matched += 1
        total += 1 if matched == 0 else matched

    angle_mae = angle_err_sum / max(1, total)
    print({"angle_mae_rad": angle_mae, "angle_mae_deg": angle_mae * 180.0 / 3.14159265})


if __name__ == "__main__":
    main()

