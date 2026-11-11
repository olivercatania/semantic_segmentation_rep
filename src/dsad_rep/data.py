import numpy as np
import cv2
import albumentations as A

from datasets import load_dataset
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import BasicTransform

from dsad_rep.config import IMAGE_SIZE, ORGANS, SPLIT_IDS

def build_transforms(image_size=IMAGE_SIZE, augment=False):
    height, width = image_size

    transforms = [
        A.Resize(
            height=height,
            width=width,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST
        )
    ]

    if augment:
        transforms.extend([
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-5, 5),
                p=0.5,
            ),
            A.RGBShift(
                r_shift_limit=(-20, 20),
                g_shift_limit=(-20, 20),
                b_shift_limit=(-20, 20),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
        ])

    transforms.extend([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])

    return A.Compose(transforms)


class DSADSegmentationDataset(Dataset):

    def __init__(
        self,
        split,
        organ=None,
        augment=False,
        image_size=IMAGE_SIZE,
    ):
        super().__init__()

        if split not in SPLIT_IDS:
            raise ValueError(f"Unknown split: {split}")

        if organ is not None and organ not in ORGANS:
            raise ValueError(f"Unknown organ: {organ}")

        filters = None

        if organ is not None:
            filters = [("organ", "==", organ)]

        self.data = load_dataset(
            "nct-tso/dsad",
            "single_organ",
            split=split,
            filters=filters,
            verification_mode="no_checks" if filters else "basic_checks",
        )

        if len(self.data) == 0:
            raise ValueError("Dataset contains no samples")

        if organ is not None:
            if set(self.data.unique("organ")) != {organ}:
                raise ValueError("Unexpected organ in dataset")

        surgery_ids = set(self.data.unique("surgery_id"))

        if not surgery_ids.issubset(SPLIT_IDS[split]):
            raise ValueError("Unexpected surgery ID in split")

        self.transform = build_transforms(
            image_size=image_size,
            augment=augment,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        image = np.asarray(
            sample["image"].convert("RGB")
        )

        mask = np.asarray(
            sample["mask"].convert("L")
        )

        # Dataset masks use 0 and 255.
        # CrossEntropyLoss needs class indices 0 and 1.
        mask = (mask > 0).astype(np.uint8)

        # Apply coordinated image/mask transformations
        transformed = self.transform(
            image=image,
            mask=mask,
        )

        return {
            "image": transformed["image"],
            "mask": transformed["mask"].long(),
            "organ": sample["organ"],
            "surgery_id": sample["surgery_id"],
            "frame_id": sample["frame_id"],
        }