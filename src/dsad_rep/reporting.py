"""Persist per-model evaluations and report suites without rerunning inference."""

import csv
import json
import math
from pathlib import Path


METRICS = ("loss", "iou", "f1", "precision", "recall", "specificity")
COLUMNS = (
    "model_type", "architecture", "organ", "split",
    "checkpoint_epoch", "completed_epochs", *METRICS,
)


def evaluation_splits(value):
    if value == "both":
        return ("train", "test")
    if value in ("train", "test"):
        return (value,)
    raise ValueError(f"Unknown evaluation split: {value}")


def _json_values(value):
    """JSON has no NaN/Infinity; retain undefined metrics as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    return value


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_values(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _result_rows(results):
    for result in results:
        for split, metrics in result["evaluations"].items():
            identity = {
                "model_type": result["model_type"],
                "architecture": result["architecture"],
                "organ": result.get("organ") or "macro",
                "split": split,
                "checkpoint_epoch": result["checkpoint_epoch"],
                "completed_epochs": result["completed_epochs"],
            }
            yield {**identity, **{name: metrics.get(name) for name in METRICS}}
            for organ, organ_metrics in metrics.get("per_organ", {}).items():
                yield {
                    **identity,
                    "organ": organ,
                    **{name: organ_metrics.get(name) for name in METRICS},
                }


def _write_csv(path, results):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(_json_values(list(_result_rows(results))))
    temporary.replace(path)


def save_evaluation(output_dir, result):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "evaluation.csv", [result])
    # Publish JSON last: it is the completion record used by the launchers.
    _write_json(output_dir / "evaluation.json", result)


def load_evaluation(output_dir):
    path = Path(output_dir) / "evaluation.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        best_path = Path(output_dir) / "best.pt"
        if best_path.exists() and best_path.stat().st_mtime_ns > path.stat().st_mtime_ns:
            return None
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict):
        return None
    if not isinstance(result.get("evaluations"), dict):
        return None
    if not all(_valid_metrics(metrics) for metrics in result["evaluations"].values()):
        return None
    if not {"model_type", "architecture", "checkpoint_epoch", "completed_epochs", "image_size"} <= result.keys():
        return None
    return result


def _valid_metrics(metrics, names=METRICS):
    if not isinstance(metrics, dict):
        return False
    if not all(
        name in metrics and (
            metrics[name] is None
            or (isinstance(metrics[name], (int, float)) and not isinstance(metrics[name], bool))
        )
        for name in names
    ):
        return False
    per_organ = metrics.get("per_organ", {})
    return isinstance(per_organ, dict) and all(
        isinstance(organ, str) and _valid_metrics(values, METRICS[1:])
        for organ, values in per_organ.items()
    )


def clear_evaluation(output_dir):
    for name in ("evaluation.json", "evaluation.csv"):
        (Path(output_dir) / name).unlink(missing_ok=True)


def evaluation_matches(
    result, *, model_type, architecture, organ, completed_epochs, image_size, splits
):
    """Check that saved scores belong to the requested completed experiment."""
    if not isinstance(result, dict):
        return False
    return (
        result.get("model_type") == model_type
        and result.get("architecture") == architecture
        and result.get("organ") == organ
        and result.get("completed_epochs") == completed_epochs
        and result.get("image_size") == list(image_size)
        and isinstance(result.get("checkpoint_epoch"), int)
        and 0 < result["checkpoint_epoch"] <= completed_epochs
        and isinstance(result.get("evaluations"), dict)
        and all(_valid_metrics(result["evaluations"].get(split)) for split in splits)
    )


def save_summary(output_dir, results, name="summary"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = list(results)
    _write_csv(output_dir / f"{name}.csv", results)
    _write_json(output_dir / f"{name}.json", results)


def print_summary(results, title="Evaluation results"):
    rows = list(_result_rows(results))
    print(f"\n{title}", flush=True)
    if not rows:
        print("No saved evaluation results.", flush=True)
        return
    headings = (
        "Model", "Architecture", "Organ", "Split", "Best epoch",
        "Loss", "IoU", "F1", "Precision", "Recall", "Specificity",
    )

    def display(value):
        if value is None or not math.isfinite(value):
            return "n/a"
        return f"{value:.4f}"

    values = [
        [
            row["model_type"], row["architecture"], row["organ"], row["split"],
            str(row["checkpoint_epoch"]),
            *(display(row.get(metric)) for metric in METRICS),
        ]
        for row in rows
    ]
    widths = [max(len(str(value)) for value in column) for column in zip(headings, *values)]
    for row in (headings, *values):
        print(" | ".join(str(value).ljust(width) for value, width in zip(row, widths)), flush=True)
