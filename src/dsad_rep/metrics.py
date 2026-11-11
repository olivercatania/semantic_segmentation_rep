import math

def binary_segmentation_metrics(predictions, mask):

    # compare predictions against ground truth

    # predictions == 1 makes boolean tensor indentifying ouxels predicted as liver
    # masks == 1 does same for correct liver pixels
    # .sum counts as many pixels are both == 1
    tp = ((predictions == 1) & (mask == 1)).sum(dim=(1, 2))
    fp = ((predictions == 1) & (mask == 0)).sum(dim=(1, 2))
    fn = ((predictions == 0) & (mask == 1)).sum(dim=(1, 2))
    tn = ((predictions == 0) & (mask == 0)).sum(dim=(1, 2))

    # convert to python lists
    tp = tp.tolist()
    fp = fp.tolist()
    fn = fn.tolist()
    tn = tn.tolist()

    results = []

    for t, f, n, z in zip(tp, fp, fn, tn):
        def divide(a, b):
            return a / b if b > 0 else float("nan")

        results.append({
            "iou": divide(t, t + f + n),          # overlap with ground truth
            "f1": divide(2 * t, 2 * t + f + n),   # same, more weight to the overlapping region
            "precision": divide(t, t + n),        # how much predicted liver is actually liver
            "recall": divide(t, t + n),           # how much actual liver was detected
            "specificity": divide(z, z + f),      # how accurately non-liver pixels were rejected
        })

    return results

def summarize_metrics(results):
    summary = {}

    for metric in (
        "iou", "f1", "precision", "recall", "specificity"
    ):
        values = [
            result[metric]
            for result in results
            if math.isfinite(result[metric])
        ]

        summary[metric] = (
            sum(values) / len(values)
            if values else float("nan")
        )

    return summary