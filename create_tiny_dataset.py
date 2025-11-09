#!/usr/bin/env python3
"""
Create a tiny synthetic DOTA dataset for testing.
This creates a small dataset with random images and annotations
so you can test the training/evaluation pipeline without downloading
the full DOTA dataset.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import random
import math

# DOTA 15 classes
CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool"
]


def create_synthetic_image(width=1024, height=1024, num_objects=5):
    """Create a synthetic aerial image with random colored rectangles as objects."""
    # Create base image with sky/ground color
    img = Image.new('RGB', (width, height), color=(120, 140, 160))
    draw = ImageDraw.Draw(img)

    # Add some texture (random patches)
    for _ in range(50):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = x1 + random.randint(50, 200), y1 + random.randint(50, 200)
        color = (
            random.randint(80, 180),
            random.randint(100, 180),
            random.randint(100, 180)
        )
        draw.rectangle([x1, y1, x2, y2], fill=color)

    annotations = []

    # Add objects (colored rectangles)
    for i in range(num_objects):
        # Random position and size
        cx = random.uniform(100, width - 100)
        cy = random.uniform(100, height - 100)
        w = random.uniform(30, 150)
        h = random.uniform(30, 150)
        theta = random.uniform(-math.pi, math.pi)  # Random orientation

        # Draw rotated rectangle
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Calculate corners of rotated rectangle
        corners = [
            (-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)
        ]

        rotated_corners = []
        for corner_x, corner_y in corners:
            rot_x = corner_x * cos_t - corner_y * sin_t + cx
            rot_y = corner_x * sin_t + corner_y * cos_t + cy
            rotated_corners.append((rot_x, rot_y))

        # Draw the object
        color = (
            random.randint(150, 255),
            random.randint(150, 255),
            random.randint(0, 100)
        )
        draw.polygon(rotated_corners, fill=color, outline=(0, 0, 0))

        # Store annotation
        category_id = random.randint(1, len(CLASSES))
        annotations.append({
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "theta": theta,
            "category_id": category_id
        })

    return img, annotations


def create_tiny_dataset(output_dir, num_train=20, num_val=5, img_size=1024):
    """Create a tiny DOTA-format dataset."""
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating tiny dataset in {output_dir}")
    print(f"  Training images: {num_train}")
    print(f"  Validation images: {num_val}")
    print(f"  Image size: {img_size}x{img_size}")
    print()

    # Create categories list
    categories = [{"id": i+1, "name": name} for i, name in enumerate(CLASSES)]

    # Create training set
    print("Generating training images...")
    train_data = {
        "images": [],
        "categories": categories,
        "annotations": []
    }

    ann_id = 1
    for img_id in range(1, num_train + 1):
        filename = f"train_{img_id:04d}.jpg"

        # Generate image and annotations
        num_objects = random.randint(3, 10)
        img, annotations = create_synthetic_image(img_size, img_size, num_objects)

        # Save image
        img.save(images_dir / filename, quality=95)

        # Add to dataset
        train_data["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": img_size,
            "height": img_size
        })

        # Add annotations
        for ann in annotations:
            train_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"],
                "rbox": [ann["cx"], ann["cy"], ann["w"], ann["h"], ann["theta"]],
                "iscrowd": 0
            })
            ann_id += 1

        if img_id % 5 == 0:
            print(f"  Created {img_id}/{num_train} training images")

    # Save training annotations
    with open(output_dir / "annotations_train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    print(f"✓ Saved annotations_train.json ({len(train_data['annotations'])} annotations)")
    print()

    # Create validation set
    print("Generating validation images...")
    val_data = {
        "images": [],
        "categories": categories,
        "annotations": []
    }

    ann_id = 1
    for img_id in range(1, num_val + 1):
        filename = f"val_{img_id:04d}.jpg"

        # Generate image and annotations
        num_objects = random.randint(3, 10)
        img, annotations = create_synthetic_image(img_size, img_size, num_objects)

        # Save image
        img.save(images_dir / filename, quality=95)

        # Add to dataset
        val_data["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": img_size,
            "height": img_size
        })

        # Add annotations
        for ann in annotations:
            val_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"],
                "rbox": [ann["cx"], ann["cy"], ann["w"], ann["h"], ann["theta"]],
                "iscrowd": 0
            })
            ann_id += 1

    # Save validation annotations
    with open(output_dir / "annotations_val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    print(f"✓ Saved annotations_val.json ({len(val_data['annotations'])} annotations)")
    print()

    # Print summary
    print("=" * 60)
    print("Dataset created successfully!")
    print("=" * 60)
    print(f"Location: {output_dir}")
    print(f"Images: {images_dir}")
    print(f"  Training: {num_train} images, {len(train_data['annotations'])} objects")
    print(f"  Validation: {num_val} images, {len(val_data['annotations'])} objects")
    print(f"Classes: {len(CLASSES)}")
    print()
    print("Next steps:")
    print(f"  1. Update config to use this dataset:")
    print(f"     dataset:")
    print(f"       root: {output_dir}")
    print(f"       train_ann: annotations_train.json")
    print(f"       val_ann: annotations_val.json")
    print()
    print(f"  2. Train: python train.py --config configs/dota_vitb_m1.yaml")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a tiny synthetic DOTA dataset for testing")
    parser.add_argument("--output", default="data/tiny_DOTA", help="Output directory")
    parser.add_argument("--train", type=int, default=20, help="Number of training images")
    parser.add_argument("--val", type=int, default=5, help="Number of validation images")
    parser.add_argument("--size", type=int, default=1024, help="Image size (width and height)")

    args = parser.parse_args()

    create_tiny_dataset(args.output, args.train, args.val, args.size)
