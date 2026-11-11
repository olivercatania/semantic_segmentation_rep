import torch.nn as nn

from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SegFormerSingle(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.network = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):

        outputs = self.network(pixel_values=x)

        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits


class DeepLabSingle(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Load existing pretrained segmentation network
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 #first version of coco dataset trained on 21 classes voc

        self.network = deeplabv3_resnet50(
            weights=weights
        )

        # Replace the origional segmentation prediction head
        self.network.classifier = DeepLabHead(
            2048,
            num_classes,
        )

    def forward(self, x):
        return self.network(x)["out"]

def build_single_model(architecture, num_classes=2):
    if architecture == "deeplabv3":
        return DeepLabSingle(num_classes)

    if architecture == "segformer":
        return SegFormerSingle(num_classes)

    raise ValueError(f"Unknown architecture: {architecture}")
    