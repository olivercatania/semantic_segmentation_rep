
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from dsad_rep.data import DSADSegmentationDataset
from dsad_rep.models import build_single_model
from dsad_rep.engine import train_one_epoch


def main():

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load liver images at a reduced resolution
    dataset = DSADSegmentationDataset(
        split="train",
        organ="liver",
        augment=False,
        image_size=(128, 160),
    )

    # Use only the first four images
    small_dataset = Subset(dataset, range(4))

    loader = DataLoader(
        small_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    model = build_single_model("deeplabv3").to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.1,
    )

    for epoch in range(3):

        avg_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device,
        )

        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")


if __name__ == "__main__":
    main()
