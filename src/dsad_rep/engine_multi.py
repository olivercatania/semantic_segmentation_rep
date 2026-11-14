import math

import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from torchvision.models.segmentation.deeplabv3 import ASPPPooling

from dsad_rep.config import ORGANS
from dsad_rep.devices import autocast_context
from dsad_rep.metrics import (
    binary_segmentation_metrics,
    summarize_metrics,
)

ORGAN_INDEX = {
    organ: index
    for index, organ in enumerate(ORGANS)
}

# batch masks and organ_names are passed
def build_multi_targets(masks, organ_names):
    # builds tensor list of organ indeces
    indices = torch.tensor(
        [ORGAN_INDEX[name] for name in organ_names],
        device=masks.device,
    )

    # array of length organs(11) numbers 0-10
    head_ids = torch.arange(
        len(ORGANS),
        device=masks.device,
    )

    # which head is annotated organ for each image
    is_annotated = (
        indices[:, None, None, None]
        == head_ids[None, :, None, None]
    )

    # torch where works: if condition (is_annotated) is true,
    # use (masks[:, None]), else use masks[:, None] - 1

    # for the annotated head, use the mask unchanged
    # for the non annotated heads, subtract mask values by 1 to signify
    # 0 is not this heads organ, and -1 is unknown
    # in cross entorpy loss calculation we have to ignore -1
    targets = torch.where(
        is_annotated,
        masks[:, None],
        masks[:, None] - 1,
    )

    # [batch_size, 11, height, width] at values -1, 0, or 1
    return targets.long()


"""
function that handles class inbalance
makes incorrect predictions on the more frequent class less penalized
so the background class is taken into account
"""
def calculate_multi_class_weights(loader):
    # will store # of background and foreground pixels per organ
    counts = torch.zeros(
        len(ORGANS), 2,
        dtype=torch.float64,
    )


    # tqdm displays progress bar for each batch in loader
    for batch in tqdm(
        loader,
        desc="Counting class frequencies",
        leave=False,
    ):

        masks = batch["mask"]

        # reshape flattens mask into tensor buncount counts repetition of each value
        # minlength always returns 2D tensor
        for mask, organ in zip(masks, batch["organ"]):
            counts[ORGAN_INDEX[organ]] += torch.bincount(
                mask.reshape(-1),
                minlength=2,
            ).double()

    # flag because of division by zero later
    if torch.any(counts == 0):
        raise ValueError(
            "At least one organ has no foreground or "
            "background training pixels."
        )

    # calculates N / (2 * n) where N = total pixels for organ
    # n = number of pixels
    # 2 classes
    per_organ_weights = (
        counts.sum(dim=1, keepdim=True) / (2 * counts)
    )

    # averages weights
    return per_organ_weights.mean(dim=0).float()


def train_multi_one_epoch(model, loader, optimizer, class_weights, device, scaler=None):
    model.train()

    # A batch of 1 causes the global-pooling BatchNorm
    # branch in DeepLabv3 to fail. Use running statistics
    # in those particular branches when batch size is 1.
    if loader.batch_size == 1:
        for module in model.modules():
            if isinstance(module, ASPPPooling):
                module.eval()

    total_loss = 0
    total_images = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        targets = build_multi_targets(
            masks,
            batch["organ"],
        )

        optimizer.zero_grad(set_to_none=True)

        # to use mixed floats (32, 16)
        with autocast_context(device):

            # [batch_size, 2(bckgrnd and frgrnd), 11(organs), H, W]
            logits = model(images)

            loss = F.cross_entropy(
                logits,
                targets,
                weight=class_weights,
                ignore_index=-1,
            )

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

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    # return average epoch loss
    if total_images == 0:
        raise ValueError("Training loader contains no batches; reduce --batch-size.")
    return total_loss / total_images


def validate_multi_one_epoch(
    model, loader, class_weights, device, desc="Validation"
):

    model.eval()

    total_loss = 0
    total_images = 0

    results_by_organ = {
        organ: []
        for organ in ORGANS
    }

    with torch.inference_mode():
        progress = tqdm(
            loader,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
        )

        for batch in progress:

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            indices = torch.tensor(
                [
                    ORGAN_INDEX[name]
                    for name in batch["organ"]
                ],
                device=device,
            )

            with autocast_context(device):

                logits = model(images)

                # Rearrange to [B, 11, 2, H, W]
                logits = logits.permute(0, 2, 1, 3, 4)

                # Select each image's annotated organ head
                selected = logits[
                    torch.arange(images.size(0), device=device),
                    indices,
                ]

                loss = F.cross_entropy(
                    selected,
                    masks,
                    weight=class_weights,
                )

            predictions = selected.argmax(dim=1)

            batch_results = binary_segmentation_metrics(
                predictions,
                masks,
            )

            for organ, result in zip(
                batch["organ"], batch_results
            ):
                results_by_organ[organ].append(result)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_images += batch_size

    per_organ = {
        organ: summarize_metrics(results)
        for organ, results in results_by_organ.items()
        if results
    }

    # First average over images within each organ,
    # then average across the 11 organs
    summary = {}

    for metric in (
        "iou", "f1", "precision", "recall", "specificity"
    ):

        values = [
            result[metric]
            for result in per_organ.values()
            if math.isfinite(result[metric])
        ]

        summary[metric] = (
            sum(values) / len(values)
            if values else float("nan")
        )

    if total_images == 0:
        raise ValueError("Evaluation loader contains no samples.")
    summary["loss"] = total_loss / total_images
    summary["samples"] = total_images
    summary["per_organ"] = per_organ

    return summary
