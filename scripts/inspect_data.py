from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

def main():
    dataset = load_dataset(
        "nct-tso/dsad",
        "single_organ",
        split="train",
        filters=[("organ", "==", "liver")],
        verification_mode = "no_checks",
    )

    print("Number of images:", len(dataset))

    sample = dataset[0]

    print("Available fields:", sample.keys())
    print("Organ:", sample["organ"])
    print("Surgery:", sample["surgery_id"])
    print("Frame:", sample["frame_id"])

    image = np.asarray(
        sample["image"].convert("RGB")
    )

    mask = np.asarray(
        sample["mask"].convert("L")
    )

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)
    print("Mask values:", np.unique(mask))

    # Convert mask pixel values to class indices
    assert set(np.unique(mask)).issubset({0, 255})

    mask = (mask > 0).astype(np.uint8)

    print("Converted mask values:", np.unique(mask))

    # Visualize the image and its correct segmentation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original image")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Liver mask")

    for ax in axes:
        ax.axis("off")

    Path("results").mkdir(exist_ok=True)

    fig.savefig("results/liver_inspection.png")
    plt.close(fig)



if __name__ == "__main__":
    main()