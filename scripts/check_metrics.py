import torch

from dsad_rep.metrics import (
    binary_segmentation_metrics,
    summarize_metrics,
)


predictions = torch.tensor([
    [[1, 1],
     [0, 0]]
])

masks = torch.tensor([
    [[1, 0],
     [1, 0]]
])

results = binary_segmentation_metrics(predictions, masks)

print(summarize_metrics(results))