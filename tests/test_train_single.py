"""Single-trainer lifecycle checks using five synthetic images and a tiny model."""

from contextlib import ExitStack, redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset, SequentialSampler

from scripts import train_single


class SyntheticDataset(Dataset):
    def __init__(self, split, organ, augment, image_size):
        self.split = split
        self.organ = organ
        self.augment = augment
        self.image_size = image_size

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {
            "image": torch.zeros(3, 2, 2),
            "mask": torch.tensor([[0, 1], [0, 1]]),
            "organ": self.organ,
            "index": index,
        }


def metrics(f1):
    return {
        "loss": 0.5,
        "iou": 0.5,
        "f1": f1,
        "precision": 0.7,
        "recall": 0.8,
        "specificity": 0.9,
        "samples": 5,
    }


class SingleTrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "deeplabv3" / "liver"
        self.events = []
        self.epochs = 0
        self.fail_evaluation = False
        self.validation_f1 = [0.9, 0.5]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(train_single, "DSADSegmentationDataset", SyntheticDataset))
        self.stack.enter_context(patch.object(
            train_single, "build_single_model", side_effect=lambda *args, **kwargs: nn.Conv2d(3, 2, 1)
        ))
        self.training = self.stack.enter_context(patch.object(
            train_single, "train_one_epoch", side_effect=self.train_epoch
        ))
        self.stack.enter_context(patch.object(
            train_single, "validate_one_epoch", side_effect=self.validate
        ))

    def args(self, *extra):
        return [
            "--output-dir", str(self.root), "--device", "cpu", "--epochs", "2",
            "--height", "2", "--width", "2", "--batch-size", "2", *extra,
        ]

    def train_epoch(self, model, loader, optimizer, criterion, device, scaler):
        self.events.append("train")
        self.epochs += 1
        self.assertTrue(loader.dataset.augment)
        optimizer.step()
        with torch.no_grad():
            model.weight.fill_(self.epochs)
        return 0.5

    def validate(self, model, loader, criterion, device, *, desc="Validation"):
        self.assertFalse(loader.dataset.augment)
        self.assertFalse(loader.drop_last)
        self.assertIsInstance(loader.sampler, SequentialSampler)
        self.assertEqual([index for batch in loader for index in batch["index"].tolist()], list(range(5)))
        if desc == "Validation":
            self.events.append("validation")
            return metrics(self.validation_f1[self.epochs - 1])
        self.events.append(f"evaluate:{loader.dataset.split}")
        if self.fail_evaluation:
            raise RuntimeError("Synthetic interrupted evaluation")
        # Epoch one was the best; evaluation must not silently use epoch two's weights.
        self.assertTrue(torch.all(model.weight == 1).item())
        return metrics(0.8)

    def test_best_checkpoint_evaluated_immediately_on_full_splits(self):
        result = train_single.main(self.args("--eval-split", "both"))
        self.assertEqual(self.events, [
            "train", "validation", "train", "validation", "evaluate:train", "evaluate:test"
        ])
        self.assertEqual(result["checkpoint_epoch"], 1)
        self.assertEqual(result["completed_epochs"], 2)
        self.assertEqual(set(result["evaluations"]), {"train", "test"})
        self.assertEqual(json.loads((self.output / "evaluation.json").read_text()), result)
        self.assertTrue((self.output / "evaluation.csv").is_file())

    def test_interrupted_evaluation_recovers_without_retraining(self):
        self.fail_evaluation = True
        with self.assertRaisesRegex(RuntimeError, "interrupted evaluation"):
            train_single.main(self.args())
        self.assertTrue((self.output / "latest.pt").is_file())
        self.assertFalse((self.output / "evaluation.json").exists())
        self.fail_evaluation = False
        self.events.clear()
        self.training.side_effect = AssertionError("Completed training must not rerun")
        result = train_single.main(self.args())
        self.assertEqual(self.events, ["evaluate:train"])
        self.assertEqual(result["completed_epochs"], 2)

    def test_evaluate_only_legacy_checkpoint_preserves_completed_epochs(self):
        train_single.main(self.args())
        (self.output / "latest.pt").unlink()
        (self.output / "complete.json").write_text(json.dumps({
            "architecture": "deeplabv3", "organ": "liver", "epochs": 2,
            "height": 2, "width": 2,
        }))
        self.events.clear()
        self.training.side_effect = AssertionError("Evaluation must not train")
        result = train_single.main(self.args("--evaluate-only", "--eval-split", "test"))
        self.assertEqual(self.events, ["evaluate:test"])
        self.assertEqual(result["checkpoint_epoch"], 1)
        self.assertEqual(result["completed_epochs"], 2)
        self.assertEqual(set(result["evaluations"]), {"test"})

    def test_partial_run_resumes_from_latest_epoch(self):
        def interrupted_train(*args):
            if self.epochs == 1:
                raise RuntimeError("Synthetic training interruption")
            return self.train_epoch(*args)

        self.training.side_effect = interrupted_train
        with self.assertRaisesRegex(RuntimeError, "training interruption"):
            train_single.main(self.args())
        self.assertEqual(self.events, ["train", "validation"])
        self.events.clear()
        self.training.side_effect = self.train_epoch
        result = train_single.main(self.args())
        self.assertEqual(self.events, ["train", "validation", "evaluate:train"])
        self.assertEqual(result["checkpoint_epoch"], 1)
        self.assertEqual(result["completed_epochs"], 2)

    def test_nonfinite_run_cannot_evaluate_stale_best_checkpoint(self):
        train_single.main(self.args())
        (self.output / "latest.pt").unlink()
        self.events.clear()
        self.epochs = 0
        self.validation_f1 = [float("nan"), float("nan")]
        with self.assertRaisesRegex(RuntimeError, "no finite validation F1"):
            train_single.main(self.args())
        self.assertNotIn("evaluate:train", self.events)
        self.assertFalse((self.output / "evaluation.json").exists())
        self.assertFalse((self.output / "best.pt").exists())

    def test_best_checkpoint_newer_than_latest_has_consistent_epoch_metadata(self):
        train_single.main(self.args())
        best_path = self.output / "best.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        best["epoch"] = 3  # Simulate interruption between best.pt and latest.pt writes.
        torch.save(best, best_path)
        self.events.clear()
        result = train_single.main(self.args("--evaluate-only"))
        self.assertEqual(self.events, ["evaluate:train"])
        self.assertEqual(result["checkpoint_epoch"], 3)
        self.assertEqual(result["completed_epochs"], 3)

    def test_smaller_epoch_target_is_rejected(self):
        train_single.main(self.args())
        with self.assertRaisesRegex(ValueError, "already completed 2 epochs"):
            train_single.main(self.args("--epochs", "1"))

    def test_no_complete_training_batch_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "reduce --batch-size"):
            train_single.main(self.args("--batch-size", "6"))
        self.training.assert_not_called()

    def test_indexed_device_is_accepted_by_parser(self):
        args = train_single.parse_args(self.args("--device", "cuda:0"))
        self.assertEqual(args.device, "cuda:0")


if __name__ == "__main__":
    unittest.main()
