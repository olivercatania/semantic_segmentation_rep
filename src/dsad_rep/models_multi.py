import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from transformers import SegformerForSemanticSegmentation

from dsad_rep.config import ORGANS

class MultiHeadDeepLab(nn.Module):
    def __init__(self):
        super().__init__()

        pretrained = deeplabv3_resnet50(
            weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        )

        # copies all but the output classifiers of the model
        # (encoder)
        self.backbone = pretrained.backbone

        # 11 independentl deeplab heads 
        self.heads = nn.ModuleList([
            DeepLabHead(2048, 2)
            for _ in ORGANS
        ])

    def forward(self, x):
        # x.shape gives tuple (batch_size, channels, height, width)
        image_size = x.shape[-2:] # grabs last 2 items of tuple

        # grabs feature map at end of the encoder
        features = self.backbone(x)["out"]

        predictions = []

        # run features through each decoder
        for head in self.heads:
            logits = head(features)

            # upscale each map back to image_size
            logits = F.interpolate(
                logits,
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )

            predictions.append(logits)

        # combine the 11 appended logits
        # group by channel first and leave height and width at the end
        return torch.stack(predictions, dim=2)


class MultiHeadSegFormer(nn.Module):
    def __init__(self):
        super().__init__()

        pretrained = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
            num_labels=2,
            ignore_mismatched_sizes=True,
        )

        self.backbone = pretrained.segformer

        self.heads = nn.ModuleList([
            copy.deepcopy(pretrained.decode_head)
            for _ in ORGANS
        ])

    def forward(self, x):
        image_size = x.shape[-2:]

        features = self.backbone(
            x,
            output_hidden_states=True,
            return_dict=False,
        )

        hidden_states = features[1]
        predictions = []

        for head in self.heads:
            logits = head(hidden_states)

            logits = F.interpolate(
                logits,
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )

            predictions.append(logits)

        return torch.stack(predictions, dim=2)

def build_multi_model(architecture):
    if architecture == "deeplabv3":
        return MultiHeadDeepLab()

    if architecture == "segformer":
        return MultiHeadSegFormer()

    raise ValueError(f"Unknown architecture: {architecture}")
