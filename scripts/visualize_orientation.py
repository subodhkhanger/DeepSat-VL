#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.config import load_config
from datasets import DotaRboxDataset
from models import OrientedDETR


def draw_rbox(img: np.ndarray, rbox: np.ndarray, color=(0, 255, 0), label: str | None = None):
    cx, cy, w, h, th = rbox
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]], dtype=np.float32)
    pts = (corners @ R.T) + np.array([cx, cy], dtype=np.float32)
    pts = pts.astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    # heading arrow
    tip = (np.array([w/2, 0.0]) @ R.T) + np.array([cx, cy])
    cv2.arrowedLine(img, (int(cx), int(cy)), (int(tip[0]), int(tip[1])), color, 2)
    if label:
        cv2.putText(img, label, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"]) 
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default="outputs/vis")
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
    ann = ds_cfg["val_ann"] if args.split == "val" else ds_cfg["train_ann"]
    ds = DotaRboxDataset(ds_cfg["root"], ann, ds_cfg.get("img_dir", "images"), ds_cfg.get("img_size", None))

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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(len(ds), args.limit)):
        item = ds[i]
        img = (item["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        inp = item["image"].unsqueeze(0).to(device)
        out = model(inp)
        logits = out["pred_logits"][0]
        boxes = out["pred_boxes"][0]
        prob = torch.softmax(logits, dim=-1)
        scores, labels = prob.max(dim=-1)
        H, W = item["target"]["size"].tolist()
        boxes = boxes.clone()
        boxes[:, 0] *= W
        boxes[:, 1] *= H
        boxes[:, 2] *= W
        boxes[:, 3] *= H

        conf_thr = float(cfg.get("eval", {}).get("conf_threshold", 0.3))
        keep = (labels < ds_cfg["num_classes"]) & (scores > conf_thr)
        for b, c, s in zip(boxes[keep], labels[keep], scores[keep]):
            rbox = b.cpu().numpy()
            draw_rbox(img, rbox, color=(0, 255, 0), label=f"c{int(c)}:{s:.2f}")

        cv2.imwrite(str(out_dir / f"vis_{i:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()

