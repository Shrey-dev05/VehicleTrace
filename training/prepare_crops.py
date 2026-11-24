"""
Dataset Crop Preparation Script
Ingests raw scene images and annotations, validates vehicle bounding boxes,
extracts crops with margin context, and writes metadata CSV with session grouping.
"""
import os
import csv
import cv2
import argparse
from pathlib import Path
from backend.app.attributes.crop_extractor import compute_image_quality


def prepare_crops(input_csv: str, images_dir: str, output_dir: str, output_csv: str):
    os.makedirs(output_dir, exist_ok=True)
    crops_meta = []

    print(f"Reading annotations from {input_csv}...")
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            img_name = row["filename"]
            img_path = os.path.join(images_dir, img_name)
            if not os.path.isfile(img_path):
                continue

            frame = cv2.imread(img_path)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            try:
                x1, y1, x2, y2 = [int(float(row[k])) for k in ("x1", "y1", "x2", "y2")]
            except KeyError:
                continue

            # Bounds checking and adaptive expansion
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            ax1 = max(0, x1 - pad_x)
            ay1 = max(0, y1 - pad_y)
            ax2 = min(w, x2 + pad_x)
            ay2 = min(h, y2 + pad_y)

            crop = frame[ay1:ay2, ax1:ax2]
            if crop.size == 0:
                continue

            q = compute_image_quality(crop)
            if q.score < 10.0:
                continue

            crop_filename = f"crop_{idx:06d}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, crop)

            crops_meta.append({
                "crop_path": crop_filename,
                "vehicle_type": row.get("vehicle_type", "car"),
                "colour": row.get("colour", "unknown"),
                "make": row.get("make", "Unknown"),
                "model": row.get("model", "Unknown"),
                "view": row.get("view", "front"),
                "lighting": row.get("lighting", "day"),
                "quality_score": q.score,
                "vehicle_group_id": row.get("vehicle_group_id", f"session_{idx}"),
                "source": row.get("source", "authorised")
            })

    # Write output CSV
    fieldnames = [
        "crop_path", "vehicle_type", "colour", "make", "model",
        "view", "lighting", "quality_score", "vehicle_group_id", "source"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(crops_meta)

    print(f"Extracted and saved {len(crops_meta)} vehicle crops to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="training/datasets/raw_annotations.csv")
    parser.add_argument("--images-dir", default="training/datasets/images")
    parser.add_argument("--output-dir", default="storage/crops")
    parser.add_argument("--output-csv", default="training/datasets/vehicle_crops.csv")
    args = parser.parse_args()

    if os.path.isfile(args.input_csv):
        prepare_crops(args.input_csv, args.images_dir, args.output_dir, args.output_csv)
    else:
        print("Dataset CSV not found. Provide --input-csv to run crop extraction.")
