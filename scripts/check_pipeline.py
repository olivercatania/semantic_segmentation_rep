
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dsad_rep.data import DSADSegmentationDataset
from dsad_rep.models import build_single_model


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load liver images at a reduced test resolution
    dataset = DSADSegmentationDataset(
        split="train",
        organ="liver",
        augment=False,
        image_size=(128, 160),
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Construct the model and move it to our device
    model = build_single_model("deeplabv3").to(device)

    # Evaluation mode: no training yet
    model.eval()

    batch = next(iter(loader))

    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    # Forward pass without computing gradients
    with torch.inference_mode():
        logits = model(images)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, masks)

        predictions = logits.argmax(dim=1)

    print("Images:", images.shape)
    print("Masks:", masks.shape)
    print("Logits:", logits.shape)
    print("Predictions:", predictions.shape)
    print("Loss:", loss.item())


if __name__ == "__main__":
    main()
