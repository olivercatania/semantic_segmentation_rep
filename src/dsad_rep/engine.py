import torch
from tqdm.auto import tqdm
from torchvision.models.segmentation.deeplabv3 import ASPPPooling

from dsad_rep.devices import autocast_context

from dsad_rep.metrics import (
    binary_segmentation_metrics,
    summarize_metrics,
)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()

    if loader.batch_size == 1:
        for module in model.modules():
            if isinstance(module, ASPPPooling):
                module.eval()

    total_loss = 0
    total_images = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        with autocast_context(device):
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_images += batch_size

        progress.set_postfix(loss=f"{loss.item():.4f}")

    if total_images == 0:
        raise ValueError("Training loader has no batches; reduce --batch-size.")
    return total_loss / total_images

def validate_one_epoch(model, loader, criterion, device, *, desc="Validation"):
    # switch to eval mode
    model.eval()

    total_loss = 0
    total_images = 0
    all_metrics = []

    # Disable gradient computation

    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc, leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with autocast_context(device):
                logits = model(images)
                loss = criterion(logits, masks)

            predictions = logits.argmax(dim=1)

            # count pixels for each image
            metrics = binary_segmentation_metrics(predictions, masks)

            all_metrics.extend(metrics)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_images += batch_size

        if total_images == 0:
            raise ValueError(f"{desc} loader has no images.")
        summary = summarize_metrics(all_metrics)

        summary["loss"] = total_loss / total_images
        summary["samples"] = total_images

        return summary

def calculate_class_weights(loader):
    # Pixel counts for background and foreground
    counts = torch.zeros(2, dtype=torch.float64)

    for batch in loader:
        masks = batch["mask"]

        # Count pixels belonging to each class
        counts += torch.bincount(
            masks.reshape(-1),
            minlength=2,
        )

    if torch.any(counts == 0):
        raise ValueError("A class has no training pixels")

    # Inverse-frequency class weighting
    weights = counts.sum() / (2 * counts)

    return weights.float()
