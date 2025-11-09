#!/usr/bin/env python3
from __future__ import annotations

"""
Prepare DOTA dataset: tile large images, parse OBB txt annotations, and export COCO-like rbox JSON.

Requirements: shapely, opencv-python, numpy

Usage example:
  python scripts/prepare_dota.py \
      --src /path/to/DOTA \
      --dst data/DOTA \
      --split train \
      --tile-size 1024 --overlap 200
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from shapely.geometry import Polygon


def obb_to_poly(coords: List[float]) -> Polygon:
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    return Polygon(pts)


def poly_to_rbox(poly: Polygon) -> Tuple[float, float, float, float, float]:
    # minimum rotated rectangle
    mrr = poly.minimum_rotated_rectangle
    pts = np.array(mrr.exterior.coords)[:-1]
    # compute center, size, and angle via OpenCV
    rect = cv2.minAreaRect(pts.astype(np.float32))  # (cx,cy),(w,h),angle(-90..0)
    (cx, cy), (w, h), angle = rect
    theta = np.deg2rad(angle)
    return float(cx), float(cy), float(w), float(h), float(theta)


def tile_image(img: np.ndarray, tile_size: int, overlap: int) -> List[Tuple[np.ndarray, int, int]]:
    H, W = img.shape[:2]
    tiles = []
    step = tile_size - overlap
    for y in range(0, max(1, H - tile_size + 1), step):
        for x in range(0, max(1, W - tile_size + 1), step):
            crop = img[y:y + tile_size, x:x + tile_size]
            if crop.shape[0] != tile_size or crop.shape[1] != tile_size:
                pad = np.zeros((tile_size, tile_size, 3), dtype=img.dtype)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad
            tiles.append((crop, x, y))
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to raw DOTA split folder (e.g., DOTA/train)")
    ap.add_argument("--dst", required=True, help="Output dataset root (e.g., data/DOTA)")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=200)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    img_dir = dst / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ann_out = dst / ("annotations_" + args.split + ".json")
    images = []
    annotations = []
    categories = []  # Fill from DOTA label list encountered
    cat2id = {}

    img_id_counter = 1
    ann_id_counter = 1

    raw_imgs = list((src / "images").glob("*.png")) + list((src / "images").glob("*.jpg"))
    for raw_img_path in raw_imgs:
        base = raw_img_path.stem
        raw = cv2.imread(str(raw_img_path))
        H, W = raw.shape[:2]
        # Read OBBs from label txt: each line x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
        label_path = src / "labelTxt" / f"{base}.txt"
        obbs = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        coords = list(map(float, parts[:8]))
                        cls = parts[8]
                        obbs.append((coords, cls))

        tiles = tile_image(raw, args.tile_size, args.overlap)
        for ti, (tile, ox, oy) in enumerate(tiles):
            tile_name = f"{base}_x{ox}_y{oy}.jpg"
            cv2.imwrite(str(img_dir / tile_name), tile)
            images.append({"id": img_id_counter, "file_name": tile_name, "width": args.tile_size, "height": args.tile_size})

            tile_poly = Polygon([(ox, oy), (ox + args.tile_size, oy), (ox + args.tile_size, oy + args.tile_size), (ox, oy + args.tile_size)])
            for coords, cls in obbs:
                poly = obb_to_poly(coords)
                inter = poly.intersection(tile_poly)
                if inter.is_empty or not inter.is_valid:
                    continue
                cx, cy, w, h, th = poly_to_rbox(inter)
                # shift to tile coords
                cx -= ox
                cy -= oy
                # skip tiny
                if w < 2 or h < 2:
                    continue
                if cls not in cat2id:
                    cid = len(cat2id) + 1
                    cat2id[cls] = cid
                    categories.append({"id": cid, "name": cls})
                annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_id_counter,
                    "category_id": cat2id[cls],
                    "rbox": [float(cx), float(cy), float(w), float(h), float(th)],
                    "iscrowd": 0
                })
                ann_id_counter += 1

            img_id_counter += 1

    with open(ann_out, 'w') as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    print(f"Wrote: {ann_out}")


if __name__ == "__main__":
    main()

