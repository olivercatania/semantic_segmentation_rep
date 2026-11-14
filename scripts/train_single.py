"""Train one organ model and immediately evaluate its best checkpoint."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dsad_rep.config import IMAGE_SIZE, ORGANS, SEED
from dsad_rep.data import DSADSegmentationDataset
from dsad_rep.devices import make_scaler, pin_memory, select_device
from dsad_rep.engine import (
    calculate_class_weights,
    train_one_epoch,
    validate_one_epoch,
)
from dsad_rep.models import build_single_model
from dsad_rep.reporting import (
    clear_evaluation,
    evaluation_splits,
    print_summary,
    save_evaluation,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organ", choices=ORGANS, default="liver")
    parser.add_argument(
        "--architecture", choices=("deeplabv3", "segformer"), default="deeplabv3"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--width", type=int, default=IMAGE_SIZE[1])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="auto",
        help="auto, cpu, cuda, mps or xpu; accelerator indices such as cuda:0 are supported.",
    )
    parser.add_argument(
        "--eval-split", choices=("train", "test", "both"), default="train",
        help="Evaluate the best validation checkpoint on these unaugmented splits.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fresh", action="store_true", help="Restart this model from scratch.")
    mode.add_argument(
        "--evaluate-only", action="store_true", help="Evaluate an existing best.pt without training."
    )
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.height, args.width) < 1:
        parser.error("Epochs, batch size, height and width must be positive.")
    if args.num_workers < 0:
        parser.error("Number of workers cannot be negative.")
    return args


def save_checkpoint(checkpoint, path):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def check_identity(checkpoint, args, path):
    if (
        checkpoint.get("architecture") != args.architecture
        or checkpoint.get("organ") != args.organ
        or tuple(checkpoint.get("image_size", ())) != (args.height, args.width)
    ):
        raise ValueError(
            f"Checkpoint architecture, organ or image size differs from this run: {path}. "
            "Use its original settings, or --fresh to restart."
        )


def completed_epochs(output_dir, checkpoint, args):
    """Recover the run length, including checkpoints made by the older trainer."""
    latest_path = output_dir / "latest.pt"
    if latest_path.is_file():
        latest = torch.load(latest_path, map_location="cpu", weights_only=True)
        check_identity(latest, args, latest_path)
        return max(int(latest["epoch"]), int(checkpoint["epoch"]))

    try:
        marker = json.loads((output_dir / "complete.json").read_text())
    except (OSError, json.JSONDecodeError):
        marker = {}
    if (
        isinstance(marker, dict)
        and marker.get("architecture") == args.architecture
        and marker.get("organ") == args.organ
        and marker.get("height") == args.height
        and marker.get("width") == args.width
        and isinstance(marker.get("epochs"), int)
        and marker["epochs"] >= checkpoint["epoch"]
    ):
        return marker["epochs"]
    return int(checkpoint["epoch"])


def evaluate_best(args, device, output_dir, *, model=None, train_loader=None):
    best_path = output_dir / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"No best checkpoint to evaluate: {best_path}")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    check_identity(checkpoint, args, best_path)
    if not math.isfinite(checkpoint["val_metrics"]["f1"]):
        raise ValueError(f"Best checkpoint has no finite validation F1: {best_path}")

    if model is None:
        model = build_single_model(args.architecture, num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss(weight=checkpoint["class_weights"].to(device))
    evaluations = {}
    for split in evaluation_splits(args.eval_split):
        loader = train_loader if split == "train" else None
        if loader is None:
            dataset = DSADSegmentationDataset(
                split=split,
                organ=args.organ,
                augment=False,
                image_size=(args.height, args.width),
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory(device),
            )
        evaluations[split] = validate_one_epoch(
            model, loader, criterion, device, desc=f"Final {split} evaluation"
        )

    result = {
        "model_type": "single",
        "architecture": args.architecture,
        "organ": args.organ,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "completed_epochs": completed_epochs(output_dir, checkpoint, args),
        "image_size": [args.height, args.width],
        "evaluations": evaluations,
    }
    save_evaluation(output_dir, result)
    print_summary([result], title=f"Completed {args.architecture} / {args.organ}")
    return result


def main(argv=None):
    args = parse_args(argv)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = select_device(args.device)
    print("Device:", device, flush=True)
    print("Organ:", args.organ, flush=True)
    print("Architecture:", args.architecture, flush=True)

    output_dir = args.output_dir / args.architecture / args.organ
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"
    if args.evaluate_only:
        return evaluate_best(args, device, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        best_path.unlink(missing_ok=True)
        latest_path.unlink(missing_ok=True)
        (output_dir / "complete.json").unlink(missing_ok=True)
        clear_evaluation(output_dir)

    image_size = (args.height, args.width)
    settings = {
        "architecture": args.architecture,
        "organ": args.organ,
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "learning_rate": 1e-4,
        "weight_decay": 0.1,
        "seed": SEED,
    }
    checkpoint = None
    start_epoch = 0
    best_f1 = -float("inf")
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
        check_identity(checkpoint, args, latest_path)
        if checkpoint.get("settings") != settings:
            raise ValueError(
                f"Existing checkpoint has different training settings: {latest_path}. "
                "Use --fresh to restart."
            )
        start_epoch = int(checkpoint["epoch"])
        best_f1 = checkpoint["best_f1"]
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint already completed {start_epoch} epochs; --epochs must be "
                "at least that large, or use --evaluate-only or --fresh."
            )
        if math.isfinite(best_f1) and not best_path.is_file():
            raise FileNotFoundError(
                f"Resumable checkpoint is missing its best model: {best_path}. "
                "Restore it or use --fresh."
            )
        print(f"Resuming from completed epoch {start_epoch}", flush=True)

    if start_epoch == args.epochs:
        if not math.isfinite(best_f1):
            raise RuntimeError("Training produced no finite validation F1; there is no best model to evaluate.")
        return evaluate_best(args, device, output_dir)

    clear_evaluation(output_dir)
    (output_dir / "complete.json").unlink(missing_ok=True)
    if checkpoint is None:
        # A best checkpoint without resumable state belongs to an older run.
        best_path.unlink(missing_ok=True)
    train_dataset = DSADSegmentationDataset(
        split="train", organ=args.organ, augment=True, image_size=image_size
    )
    if len(train_dataset) < args.batch_size:
        raise ValueError(
            f"Training has {len(train_dataset)} images but batch size is {args.batch_size}; "
            "reduce --batch-size so at least one complete training batch exists."
        )
    val_dataset = DSADSegmentationDataset(
        split="validation", organ=args.organ, augment=False, image_size=image_size
    )
    # This full, unaugmented loader serves both class weighting and final train evaluation.
    weights_dataset = DSADSegmentationDataset(
        split="train", organ=args.organ, augment=False, image_size=image_size
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory(device),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    weights_loader = DataLoader(weights_dataset, shuffle=False, drop_last=False, **loader_options)
    weights = (
        checkpoint["class_weights"] if checkpoint is not None
        else calculate_class_weights(weights_loader)
    ).to(device)
    print("Class weights:", weights, flush=True)

    model = build_single_model(args.architecture, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    scaler = make_scaler(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
    del checkpoint

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{args.epochs} | Train loss: {train_loss:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | Val IoU: {val_metrics['iou']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}",
            flush=True,
        )
        state = {
            "epoch": epoch + 1,
            "architecture": args.architecture,
            "organ": args.organ,
            "image_size": image_size,
            "settings": settings,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_weights": weights.cpu(),
            "val_metrics": val_metrics,
        }
        if math.isfinite(val_metrics["f1"]) and val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(state, best_path)
            print("Saved new best checkpoint", flush=True)
        save_checkpoint({
            **state,
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else {},
            "best_f1": best_f1,
        }, latest_path)

    if not math.isfinite(best_f1):
        raise RuntimeError("Training produced no finite validation F1; there is no best model to evaluate.")
    print(f"Training complete. Best validation F1: {best_f1:.4f}", flush=True)
    return evaluate_best(args, device, output_dir, model=model, train_loader=weights_loader)


if __name__ == "__main__":
    main()
