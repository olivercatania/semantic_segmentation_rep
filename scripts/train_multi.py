
import argparse
import csv
import gc
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dsad_rep.config import IMAGE_SIZE, SEED
from dsad_rep.data import DSADSegmentationDataset
from dsad_rep.devices import (
    make_scaler,
    pin_memory,
    release_device_cache,
    select_device,
)
from dsad_rep.models_multi import build_multi_model
from dsad_rep.engine_multi import (
    calculate_multi_class_weights,
    train_multi_one_epoch,
    validate_multi_one_epoch,
)
from dsad_rep.plotting import save_training_plot
from dsad_rep.reporting import (
    clear_evaluation,
    evaluation_matches,
    evaluation_splits,
    load_evaluation,
    print_summary,
    save_evaluation,
    save_summary,
)


def save_history(history, output_dir):

    path = output_dir / "history.csv"

    with path.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_iou",
            "val_f1",
        ])

        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["val_loss"][i],
                history["val_iou"][i],
                history["val_f1"][i],
            ])

    save_training_plot(history, output_dir)


def save_checkpoint(checkpoint, path):

    temporary = path.with_suffix(".tmp")

    torch.save(checkpoint, temporary)

    temporary.replace(path)


def training_settings(architecture, args):
    return {
        "architecture": architecture,
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "learning_rate": 1e-4,
        "weight_decay": 0.1,
        "seed": SEED,
    }


def train_architecture(
    architecture,
    args,
    device,
    train_loader,
    val_loader,
    class_weights,
    evaluation_loaders,
):

    output_dir = (
        Path(args.output_dir)
        / "combined"
        / architecture
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    if args.fresh and args.evaluate_only:
        raise ValueError("--fresh cannot be combined with --evaluate-only")

    settings = training_settings(architecture, args)

    if args.fresh:
        latest_path.unlink(missing_ok=True)
        best_path.unlink(missing_ok=True)
        clear_evaluation(output_dir)

    if args.evaluate_only and not best_path.is_file():
        raise FileNotFoundError(f"No checkpoint to evaluate: {best_path}")

    print(f"\nArchitecture: {architecture}", flush=True)
    print(f"Device: {device}", flush=True)

    # Start with independently initialized combined model
    model = build_multi_model(architecture).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.1,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.9,
    )

    scaler = make_scaler(device)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_f1": [],
    }

    best_f1 = -float("inf")
    start_epoch = 0

    # Restore a previously interrupted experiment
    if latest_path.exists() and not args.fresh:

        checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=True,
        )

        if not args.evaluate_only and checkpoint["settings"] != settings:
            raise ValueError(
                f"Existing checkpoint has different settings: "
                f"{latest_path}. Use --fresh to restart."
            )

        start_epoch = checkpoint["epoch"]
        if not args.evaluate_only:
            if start_epoch > args.epochs:
                raise ValueError(
                    f"Checkpoint already completed {start_epoch} epochs, more than "
                    f"the requested {args.epochs}. Use --evaluate-only or --fresh."
                )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler_state = checkpoint.get("scaler_state_dict")
            if scaler is not None and scaler_state:
                scaler.load_state_dict(scaler_state)
            history = checkpoint["history"]
            best_f1 = checkpoint["best_f1"]
            if math.isfinite(best_f1) and not best_path.is_file():
                raise FileNotFoundError(
                    f"Resumable checkpoint is missing its best model: {best_path}. "
                    "Restore it or use --fresh."
                )
            print(f"Resuming from completed epoch {start_epoch}", flush=True)
        del checkpoint

    stop_epoch = start_epoch if args.evaluate_only else args.epochs
    if start_epoch < stop_epoch:
        # Metrics from an earlier training run must not survive a resumed run.
        clear_evaluation(output_dir)
        if start_epoch == 0:
            # Without a resumable latest checkpoint, this is a new experiment.
            best_path.unlink(missing_ok=True)

    completed_epochs = start_epoch
    for epoch in range(start_epoch, stop_epoch):

        print(
            f"\n{architecture} — Epoch {epoch + 1}/{args.epochs}",
            flush=True,
        )

        train_loss = train_multi_one_epoch(
            model,
            train_loader,
            optimizer,
            class_weights,
            device,
            scaler,
        )

        val_metrics = validate_multi_one_epoch(
            model,
            val_loader,
            class_weights,
            device,
        )

        scheduler.step()

        # Record epoch results
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_f1"].append(val_metrics["f1"])

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}",
            flush=True,
        )

        save_history(history, output_dir)

        # Select the best model using macro validation F1
        if (
            math.isfinite(val_metrics["f1"])
            and val_metrics["f1"] > best_f1
        ):

            best_f1 = val_metrics["f1"]

            save_checkpoint({
                "epoch": epoch + 1,
                "settings": settings,
                "model_state_dict": model.state_dict(),
                "class_weights": class_weights.cpu(),
                "val_metrics": val_metrics,
            }, best_path)

            print(
                "Saved new best checkpoint",
                flush=True,
            )

        # Save the most recently completed epoch
        save_checkpoint({
            "epoch": epoch + 1,
            "settings": settings,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_f1": best_f1,
            "history": history,
        }, latest_path)
        completed_epochs = epoch + 1

    if not best_path.is_file():
        raise RuntimeError(f"Training did not produce a best checkpoint: {best_path}")

    # Evaluate this model before starting the next architecture. Validation alone
    # selects the checkpoint; final training/test results never influence it.
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    if not math.isfinite(checkpoint["val_metrics"]["f1"]):
        raise ValueError(f"Best checkpoint has no finite validation F1: {best_path}")
    checkpoint_settings = checkpoint["settings"]
    for key in ("architecture", "height", "width"):
        if checkpoint_settings[key] != settings[key]:
            raise ValueError(f"Checkpoint {key} does not match the requested evaluation: {best_path}")
    checkpoint_epoch = checkpoint["epoch"]
    completed_epochs = max(completed_epochs, checkpoint_epoch)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluation_weights = checkpoint["class_weights"].to(device)
    del checkpoint

    result = {
        "model_type": "combined",
        "architecture": architecture,
        "organ": None,
        "checkpoint_epoch": checkpoint_epoch,
        "completed_epochs": completed_epochs,
        "image_size": [args.height, args.width],
        "evaluations": {},
    }
    previous = load_evaluation(output_dir)
    if (
        not args.evaluate_only
        and previous is not None
        and previous.get("checkpoint_epoch") == checkpoint_epoch
    ):
        for split, metrics in previous["evaluations"].items():
            if evaluation_matches(
                previous,
                model_type="combined",
                architecture=architecture,
                organ=None,
                completed_epochs=completed_epochs,
                image_size=result["image_size"],
                splits=(split,),
            ):
                result["evaluations"][split] = metrics

    for split, loader in evaluation_loaders.items():
        if split not in result["evaluations"]:
            result["evaluations"][split] = validate_multi_one_epoch(
                model,
                loader,
                evaluation_weights,
                device,
                desc=f"{architecture}: final {split} evaluation",
            )
    save_evaluation(output_dir, result)
    print_summary([result], title=f"{architecture}: final evaluation")

    # Release accelerator allocations before the next architecture.
    del model, optimizer, scheduler, scaler, evaluation_weights

    gc.collect()
    release_device_cache(device)
    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("deeplabv3", "segformer"),
        default=["deeplabv3", "segformer"],
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--width", type=int, default=IMAGE_SIZE[1])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or xpu")
    parser.add_argument("--eval-split", choices=("train", "test", "both"), default="train")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.height < 1
        or args.width < 1
        or args.num_workers < 0
    ):
        parser.error("Invalid training arguments")
    if args.fresh and args.evaluate_only:
        parser.error("--fresh cannot be combined with --evaluate-only")

    try:
        device = select_device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    image_size = (args.height, args.width)
    architectures = list(dict.fromkeys(args.architectures))
    splits = evaluation_splits(args.eval_split)
    results_by_architecture = {}
    pending = []
    needs_training = False
    # Check completion before loading data. A finished suite should only read
    # its saved scores; a missing evaluation needs no class-frequency scan.
    for architecture in architectures:
        output_dir = args.output_dir / "combined" / architecture
        best_path = output_dir / "best.pt"
        latest_path = output_dir / "latest.pt"
        if args.evaluate_only:
            if not best_path.is_file():
                parser.error(f"No checkpoint to evaluate: {best_path}")
            pending.append(architecture)
            continue
        if args.fresh or not latest_path.is_file():
            needs_training = True
            pending.append(architecture)
            continue
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
        if checkpoint["settings"] != training_settings(architecture, args):
            parser.error(f"Existing checkpoint has different settings: {latest_path}. Use --fresh.")
        epoch = checkpoint["epoch"]
        best_f1 = checkpoint["best_f1"]
        del checkpoint
        if epoch > args.epochs:
            parser.error(f"Checkpoint already completed {epoch} epochs. Use --evaluate-only or --fresh.")
        if math.isfinite(best_f1) and not best_path.is_file():
            parser.error(f"Resumable checkpoint is missing its best model: {best_path}. Restore it or use --fresh.")
        if epoch < args.epochs:
            needs_training = True
            pending.append(architecture)
            continue
        if not math.isfinite(best_f1):
            parser.error(f"Training produced no finite validation F1: {latest_path}. Use --fresh.")
        result = load_evaluation(output_dir)
        if evaluation_matches(
            result, model_type="combined", architecture=architecture, organ=None,
            completed_epochs=args.epochs, image_size=image_size, splits=splits,
        ):
            results_by_architecture[architecture] = {
                **result,
                "evaluations": {split: result["evaluations"][split] for split in splits},
            }
            print(f"{architecture}: already trained and evaluated; using saved results.", flush=True)
        else:
            pending.append(architecture)

    if not pending:
        results = [results_by_architecture[name] for name in architectures]
        save_summary(args.output_dir / "combined", results)
        print_summary(results, title="All combined model results")
        return

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory(device),
    }

    # Evaluation visits every image without augmentation, shuffling or dropped
    # remainder batches. Reuse the training evaluation data for class weights.
    evaluation_datasets = {
        split: DSADSegmentationDataset(
            split=split,
            organ=None,
            augment=False,
            image_size=image_size,
        )
        for split in splits
    }
    evaluation_loaders = {
        split: DataLoader(dataset, shuffle=False, **loader_options)
        for split, dataset in evaluation_datasets.items()
    }

    train_loader = val_loader = class_weights = None
    if needs_training:
        print("Loading combined training dataset", flush=True)
        train_dataset = DSADSegmentationDataset(
            split="train", organ=None, augment=True, image_size=image_size,
        )
        val_dataset = DSADSegmentationDataset(
            split="validation", organ=None, augment=False, image_size=image_size,
        )
        weights_dataset = evaluation_datasets.get("train")
        if weights_dataset is None:
            weights_dataset = DSADSegmentationDataset(
                split="train", organ=None, augment=False, image_size=image_size,
            )

        train_loader = DataLoader(
            train_dataset, shuffle=True, drop_last=True, **loader_options,
        )
        if len(train_loader) == 0:
            parser.error("Training loader contains no batches; reduce --batch-size.")
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
        weights_loader = DataLoader(
            weights_dataset, batch_size=8, shuffle=False, num_workers=args.num_workers,
        )
        class_weights = calculate_multi_class_weights(weights_loader).to(device)
        print("Combined class weights:", class_weights, flush=True)
        print("Training images:", len(train_dataset), flush=True)
        print("Validation images:", len(val_dataset), flush=True)

    for architecture in pending:
        result = train_architecture(
            architecture,
            args,
            device,
            train_loader,
            val_loader,
            class_weights,
            evaluation_loaders,
        )
        results_by_architecture[architecture] = {
            **result,
            "evaluations": {split: result["evaluations"][split] for split in splits},
        }

    results = [results_by_architecture[name] for name in architectures]
    save_summary(args.output_dir / "combined", results)
    print_summary(results, title="All combined model results")
    print("\nAll combined experiments completed.", flush=True)


if __name__ == "__main__":
    main()
