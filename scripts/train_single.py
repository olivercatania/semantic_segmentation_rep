import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dsad_rep.config import IMAGE_SIZE, ORGANS, SEED
from dsad_rep.data import DSADSegmentationDataset
from dsad_rep.models import build_single_model
from dsad_rep.engine import (
    train_one_epoch,
    validate_one_epoch,
    calculate_class_weights,
)


def main():
    # Read experiment settings from the command line
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--organ",
        choices=ORGANS,
        default="liver",
    )

    parser.add_argument(
        "--architecture",
        default="deeplabv3",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--width", type=int, default=IMAGE_SIZE[1])

    args = parser.parse_args()

    # Make random operations more reproducible
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Select CPU or CUDA
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Organ:", args.organ)
    print("Architecture:", args.architecture)

    image_size = (args.height, args.width)

    # Load training data with augmentation
    train_dataset = DSADSegmentationDataset(
        split="train",
        organ=args.organ,
        augment=True,
        image_size=image_size,
    )

    # Load validation data without augmentation
    val_dataset = DSADSegmentationDataset(
        split="validation",
        organ=args.organ,
        augment=False,
        image_size=image_size,
    )

    # Use deterministic transforms for class frequencies
    weights_dataset = DSADSegmentationDataset(
        split="train",
        organ=args.organ,
        augment=False,
        image_size=image_size,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    weights_loader = DataLoader(
        weights_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Calculate class weights using training data only
    weights = calculate_class_weights(weights_loader)
    weights = weights.to(device)

    print("Class weights:", weights)

    # Build the selected segmentation model
    model = build_single_model(
        args.architecture,
        num_classes=2,
    ).to(device)

    # Weighted cross-entropy
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.1,
    )

    # Reduce the learning rate every 10 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.9,
    )

    # Enable gradient scaling when using CUDA
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    # Directory for this experiment's checkpoints
    output_dir = (
        Path("checkpoints")
        / args.architecture
        / args.organ
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -float("inf")

    # Training and validation
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
        )

        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Select the best checkpoint using validation F1
        if (
            math.isfinite(val_metrics["f1"])
            and val_metrics["f1"] > best_f1
        ):

            best_f1 = val_metrics["f1"]

            torch.save({
                "epoch": epoch + 1,
                "architecture": args.architecture,
                "organ": args.organ,
                "image_size": image_size,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_weights": weights.cpu(),
                "val_metrics": val_metrics,
            }, output_dir / "best.pt")

            print("Saved new best checkpoint")

    print("Training complete")
    print("Best validation F1:", best_f1)


if __name__ == "__main__":
    main()
