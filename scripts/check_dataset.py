import torch
from torch.utils.data import DataLoader

from dsad_rep.data import DSADSegmentationDataset


def main():

    dataset = DSADSegmentationDataset(
        split="train",
        organ="liver",
        augment=False,
    )

    print("Number of images:", len(dataset))

    sample = dataset[0]

    print("Single image:", sample["image"].shape)
    print("Single mask:", sample["mask"].shape)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    images = batch["image"]
    masks = batch["mask"]

    print("Batched images:", images.shape)
    print("Batched masks:", masks.shape)

    print("Image dtype:", images.dtype)
    print("Mask dtype:", masks.dtype)
    print("Mask classes:", torch.unique(masks))


if __name__ == "__main__":
    main()