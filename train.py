#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.config import load_config
from datasets import DotaRboxDataset
from models import OrientedDETR
from models.losses import OrientedSetCriterion


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = [b["target"] for b in batch]
    return images, targets


def save_checkpoint(state: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def train_one_epoch(model, criterion, loader, optimizer, scaler, device, epoch, amp: bool):
    model.train()
    total = 0
    logs = {"loss": 0.0, "loss_cls": 0.0, "loss_bbox": 0.0, "loss_angle": 0.0}
    for images, targets in loader:
        images = images.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                out = model(images)
                losses = criterion(out, targets)
                loss = losses["loss_total"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        elif amp and device.type == "mps":
            # MPS doesn't support autocast yet, use regular forward pass
            out = model(images)
            losses = criterion(out, targets)
            loss = losses["loss_total"]
            loss.backward()
            optimizer.step()
        else:
            out = model(images)
            losses = criterion(out, targets)
            loss = losses["loss_total"]
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total += bs
        logs["loss"] += losses["loss_total"].item() * bs
        logs["loss_cls"] += losses["loss_cls"].item() * bs
        logs["loss_bbox"] += losses["loss_bbox"].item() * bs
        logs["loss_angle"] += losses["loss_angle"].item() * bs

    for k in logs:
        logs[k] /= total
    print(f"[epoch {epoch}] train: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))
    return logs


@torch.no_grad()
def validate(model, criterion, loader, device, epoch):
    model.eval()
    total = 0
    logs = {"loss": 0.0, "loss_cls": 0.0, "loss_bbox": 0.0, "loss_angle": 0.0}
    for images, targets in loader:
        images = images.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        out = model(images)
        losses = criterion(out, targets)
        bs = images.size(0)
        total += bs
        logs["loss"] += losses["loss_total"].item() * bs
        logs["loss_cls"] += losses["loss_cls"].item() * bs
        logs["loss_bbox"] += losses["loss_bbox"].item() * bs
        logs["loss_angle"] += losses["loss_angle"].item() * bs

    for k in logs:
        logs[k] /= total
    print(f"[epoch {epoch}] valid: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
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

    torch.manual_seed(cfg.get("seed", 42))

    out_dir = Path(cfg.get("output_dir", "outputs/run"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    ds_cfg = cfg["dataset"]
    train_ds = DotaRboxDataset(ds_cfg["root"], ds_cfg["train_ann"], ds_cfg.get("img_dir", "images"), ds_cfg.get("img_size", None))
    val_ds = DotaRboxDataset(ds_cfg["root"], ds_cfg["val_ann"], ds_cfg.get("img_dir", "images"), ds_cfg.get("img_size", None))

    batch_size = cfg["train"]["batch_size"]
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Model
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

    # Criterion and Optimizer
    l_cfg = cfg["loss"]
    criterion = OrientedSetCriterion(
        num_classes=ds_cfg["num_classes"],
        cls_weight=l_cfg.get("cls_weight", 2.0),
        bbox_l1_weight=l_cfg.get("bbox_l1_weight", 5.0),
        angle_weight=l_cfg.get("angle_weight", 2.0),
        no_object_weight=l_cfg.get("no_object_weight", 0.1)
    )

    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 1e-4))

    # AMP: CUDA supports full autocast, MPS runs without autocast (not yet supported)
    amp = bool(cfg["train"].get("amp", True) and (device.type == "cuda" or device.type == "mps"))
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    if amp and device.type == "mps":
        print("Note: MPS doesn't support autocast yet, running in standard precision mode")

    best_val = float('inf')

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_logs = train_one_epoch(model, criterion, train_ld, optimizer, scaler, device, epoch, amp)
        val_logs = validate(model, criterion, val_ld, device, epoch)

        # Save latest
        save_checkpoint({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_logs["loss"],
            "config": cfg,
        }, out_dir / "latest.pt")

        if val_logs["loss"] < best_val:
            best_val = val_logs["loss"]
            save_checkpoint({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_logs["loss"],
                "config": cfg,
            }, out_dir / "best.pt")
            print(f"Saved best checkpoint at epoch {epoch} (val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()

