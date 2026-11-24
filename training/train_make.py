"""
PyTorch Training Script for Vehicle Manufacturer (Make) Classifier
"""
import os
import json
import argparse
import torch
import torch.nn as nn
from torchvision import models


def train_make_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Training Make Classifier on device: {device}")

    with open("training/configs/make_classes.json") as f:
        classes = json.load(f)["classes"]

    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if args.pretrained else None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(classes))
    model = model.to(device)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes}, args.output_path)
    print(f"Initial make classifier checkpoint exported to {args.output_path} ({len(classes)} classes).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--force-cpu", action="store_true", default=False)
    parser.add_argument("--output-path", default="model_artifacts/make_classifier/make_best.pth")
    args = parser.parse_args()
    train_make_model(args)
