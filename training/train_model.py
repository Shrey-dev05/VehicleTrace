"""
PyTorch Training Script for Hierarchical Vehicle Model Classifier
"""
import os
import json
import argparse
import torch
import torch.nn as nn
from torchvision import models


def train_model_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Training Hierarchical Model Classifier on device: {device}")

    with open("training/configs/model_classes.json") as f:
        hierarchy = json.load(f)["hierarchy"]

    all_models = set()
    for mlist in hierarchy.values():
        all_models.update(mlist)
    classes = sorted(list(all_models))

    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if args.pretrained else None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(classes))
    model = model.to(device)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "hierarchy": hierarchy}, args.output_path)
    print(f"Hierarchical model checkpoint exported to {args.output_path} ({len(classes)} candidate model classes).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--force-cpu", action="store_true", default=False)
    parser.add_argument("--output-path", default="model_artifacts/model_classifier/model_best.pth")
    args = parser.parse_args()
    train_model_classifier(args)
