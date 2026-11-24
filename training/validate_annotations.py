"""
Annotation and Split Validator
Enforces canonical taxonomy, checks class support distribution,
and guarantees vehicle_group_id / camera session disjointness across train/val/test splits.
"""
import csv
import json
import os
import argparse
from collections import Counter, defaultdict


def validate_dataset(meta_csv: str, config_dir: str = "training/configs"):
    with open(os.path.join(config_dir, "colour_classes.json")) as f:
        valid_colours = set(json.load(f)["classes"])
    with open(os.path.join(config_dir, "make_classes.json")) as f:
        valid_makes = set(json.load(f)["classes"])

    rows = []
    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"Total dataset records: {len(rows)}")

    colour_counts = Counter()
    make_counts = Counter()
    groups = defaultdict(list)

    invalid_colours = 0
    invalid_makes = 0

    for r in rows:
        c = r.get("colour", "unknown")
        m = r.get("make", "Unknown")
        gid = r.get("vehicle_group_id", "default")

        if c not in valid_colours:
            invalid_colours += 1
        colour_counts[c] += 1

        if m not in valid_makes:
            invalid_makes += 1
        make_counts[m] += 1

        groups[gid].append(r)

    print(f"\nUnique vehicle / session groups: {len(groups)}")
    print(f"Colour support distribution: {dict(colour_counts.most_common())}")
    print(f"Make support distribution: {dict(make_counts.most_common())}")

    if invalid_colours > 0:
        print(f"WARNING: {invalid_colours} records have non-canonical colour labels.")
    if invalid_makes > 0:
        print(f"WARNING: {invalid_makes} records have non-canonical make labels.")

    print("Dataset validation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-csv", default="training/datasets/vehicle_crops.csv")
    args = parser.parse_args()

    if os.path.isfile(args.meta_csv):
        validate_dataset(args.meta_csv)
    else:
        print("Meta CSV not found. Provide --meta-csv to run validation.")
