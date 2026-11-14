import io
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from dsad_rep.config import ORGANS
from dsad_rep.engine_multi import (
    build_multi_targets,
    train_multi_one_epoch,
    validate_multi_one_epoch,
)
from dsad_rep.models_multi import MultiHeadSegFormer
from dsad_rep.reporting import clear_evaluation, load_evaluation, save_evaluation
from scripts import train_multi


class TinyBackbone(nn.Module):
    def forward(self, images, **kwargs):
        return images, (images,)


class TinyDecodeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1)

    def forward(self, hidden_states):
        return self.conv(hidden_states[0])


class TinyMultiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2 * len(ORGANS), 1)

    def forward(self, images):
        logits = self.conv(images)
        return logits.reshape(images.shape[0], 2, len(ORGANS), *images.shape[-2:])


class MultiModelTests(unittest.TestCase):
    def test_segformer_initialization_and_spatial_output(self):
        pretrained = SimpleNamespace(segformer=TinyBackbone(), decode_head=TinyDecodeHead())
        with patch(
            "dsad_rep.models_multi.SegformerForSemanticSegmentation.from_pretrained",
            return_value=pretrained,
        ):
            model = MultiHeadSegFormer()
        logits = model(torch.randn(2, 3, 5, 7))
        self.assertEqual(logits.shape, (2, 2, len(ORGANS), 5, 7))
        self.assertEqual(len({id(head) for head in model.heads}), len(ORGANS))

    def test_partial_annotation_targets(self):
        masks = torch.tensor([[[0, 1], [1, 0]]])
        targets = build_multi_targets(masks, ["liver"])
        self.assertEqual(targets.shape, (1, len(ORGANS), 2, 2))
        self.assertTrue(torch.equal(targets[:, ORGANS.index("liver")], masks))
        self.assertTrue(torch.equal(targets[:, ORGANS.index("colon")], masks - 1))

    def test_cpu_training_and_evaluation_keep_remainder(self):
        samples = [
            {
                "image": torch.randn(3, 4, 4),
                "mask": torch.tensor([[0, 1, 0, 1]] * 4),
                "organ": "liver",
            }
            for _ in range(3)
        ]
        loader = DataLoader(samples, batch_size=2)
        model = TinyMultiModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        weights = torch.ones(2)
        before = model.conv.weight.detach().clone()
        loss = train_multi_one_epoch(model, loader, optimizer, weights, torch.device("cpu"))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model.conv.weight))
        result = validate_multi_one_epoch(model, loader, weights, torch.device("cpu"))
        self.assertEqual(result["samples"], 3)
        self.assertIn("liver", result["per_organ"])
        self.assertFalse(model.training)


class MultiTrainingTests(unittest.TestCase):
    def test_immediate_best_checkpoint_evaluation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args = Namespace(
                output_dir=Path(directory), batch_size=2, height=4, width=4,
                epochs=2, fresh=False, evaluate_only=False,
            )
            events = []
            train_calls = 0
            validation_calls = 0

            def train(model, loader, optimizer, *rest):
                nonlocal train_calls
                train_calls += 1
                optimizer.zero_grad()
                model.weight.sum().backward()
                optimizer.step()
                with torch.no_grad():
                    model.weight.fill_(train_calls)
                events.append("train")
                return 0.5

            def validate(model, loader, weights, device, desc="Validation"):
                nonlocal validation_calls
                metrics = {name: 0.5 for name in (
                    "loss", "iou", "f1", "precision", "recall", "specificity",
                )}
                if desc == "Validation":
                    validation_calls += 1
                    metrics["f1"] = 1 / validation_calls
                    events.append("validation")
                else:
                    self.assertEqual(model.weight.item(), 1.0)
                    self.assertTrue(torch.equal(weights, torch.ones(2)))
                    events.append(loader)
                metrics["per_organ"] = {"liver": {
                    name: metrics[name] for name in ("iou", "f1", "precision", "recall", "specificity")
                }}
                return metrics

            def run():
                return train_multi.train_architecture(
                    "segformer", args, torch.device("cpu"), None, None,
                    torch.ones(2), {"train": "evaluate_train", "test": "evaluate_test"},
                )

            with patch.object(train_multi, "build_multi_model", side_effect=lambda _: nn.Linear(1, 1, bias=False)), patch.object(
                train_multi, "train_multi_one_epoch", side_effect=train
            ), patch.object(train_multi, "validate_multi_one_epoch", side_effect=validate), patch.object(
                train_multi, "save_history"
            ):
                result = run()
                self.assertEqual(events, ["train", "validation", "train", "validation", "evaluate_train", "evaluate_test"])
                self.assertEqual(result["checkpoint_epoch"], 1)
                self.assertEqual(result["completed_epochs"], 2)
                output_dir = args.output_dir / "combined" / "segformer"
                self.assertEqual(load_evaluation(output_dir), result)

                events.clear()
                run()
                self.assertEqual(events, [])

                clear_evaluation(output_dir)
                run()
                self.assertEqual(events, ["evaluate_train", "evaluate_test"])

                events.clear()
                args.epochs = 3
                result = run()
                self.assertEqual(events, ["train", "validation", "evaluate_train", "evaluate_test"])
                self.assertEqual(result["completed_epochs"], 3)

                events.clear()
                args.evaluate_only = True
                run()
                self.assertEqual(events, ["evaluate_train", "evaluate_test"])

                args.evaluate_only = False
                args.epochs = 2
                with self.assertRaisesRegex(ValueError, "already completed 3 epochs"):
                    run()

    def test_new_run_does_not_reuse_stale_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args = Namespace(
                output_dir=Path(directory), batch_size=2, height=4, width=4,
                epochs=1, fresh=False, evaluate_only=False,
            )
            output_dir = args.output_dir / "combined" / "segformer"
            output_dir.mkdir(parents=True)
            best_path = output_dir / "best.pt"
            best_path.write_bytes(b"stale checkpoint without latest.pt")
            metrics = {"loss": 0.5, "iou": 0.5, "f1": float("nan")}

            def train(model, loader, optimizer, *rest):
                optimizer.zero_grad()
                model.weight.sum().backward()
                optimizer.step()
                return 0.5

            with patch.object(train_multi, "build_multi_model", return_value=nn.Linear(1, 1, bias=False)), patch.object(
                train_multi, "train_multi_one_epoch", side_effect=train
            ), patch.object(train_multi, "validate_multi_one_epoch", return_value=metrics), patch.object(
                train_multi, "save_history"
            ):
                with self.assertRaisesRegex(RuntimeError, "did not produce a best checkpoint"):
                    train_multi.train_architecture(
                        "segformer", args, torch.device("cpu"), None, None,
                        torch.ones(2), {"train": object()},
                    )
            self.assertFalse(best_path.exists())
            self.assertFalse((output_dir / "evaluation.json").exists())

    def test_evaluate_only_datasets_are_unaugmented_and_complete(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            checkpoint = Path(directory) / "combined" / "segformer" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            calls = []

            def dataset(**kwargs):
                calls.append(kwargs)
                return [{"image": torch.zeros(3, 4, 4)} for _ in range(3)]

            def evaluate(*args):
                self.assertIsNone(args[3])
                self.assertIsNone(args[4])
                self.assertIsNone(args[5])
                loaders = args[6]
                self.assertEqual(set(loaders), {"train", "test"})
                for loader in loaders.values():
                    self.assertFalse(loader.drop_last)
                    self.assertEqual(sum(len(batch["image"]) for batch in loader), 3)
                return {"evaluations": {"train": {}, "test": {}}}

            argv = [
                "train_multi", "--architectures", "segformer", "--evaluate-only",
                "--eval-split", "both", "--device", "cpu", "--batch-size", "2",
                "--output-dir", directory,
            ]
            with patch("sys.argv", argv), patch.object(
                train_multi, "DSADSegmentationDataset", side_effect=dataset
            ), patch.object(train_multi, "train_architecture", side_effect=evaluate), patch.object(
                train_multi, "calculate_multi_class_weights"
            ) as count, patch.object(train_multi, "save_summary"), patch.object(train_multi, "print_summary"):
                train_multi.main()
                count.assert_not_called()
            self.assertEqual([call["split"] for call in calls], ["train", "test"])
            self.assertTrue(all(call["augment"] is False for call in calls))

    def test_completed_main_uses_cache_or_only_missing_evaluation(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output_dir = Path(directory) / "combined" / "segformer"
            output_dir.mkdir(parents=True)
            (output_dir / "best.pt").touch()
            settings = train_multi.training_settings(
                "segformer", Namespace(batch_size=2, height=4, width=4)
            )
            torch.save({"settings": settings, "epoch": 2, "best_f1": 0.9}, output_dir / "latest.pt")
            metrics = {name: 0.5 for name in ("loss", "iou", "f1", "precision", "recall", "specificity")}
            result = {
                "model_type": "combined", "architecture": "segformer", "organ": None,
                "checkpoint_epoch": 1, "completed_epochs": 2, "image_size": [4, 4],
                "evaluations": {"train": metrics},
            }
            save_evaluation(output_dir, result)
            argv = [
                "train_multi", "--architectures", "segformer", "--epochs", "2",
                "--device", "cpu", "--height", "4", "--width", "4", "--output-dir", directory,
            ]
            with patch("sys.argv", argv), patch.object(
                train_multi, "DSADSegmentationDataset", return_value=[{}] * 3
            ) as dataset, patch.object(train_multi, "train_architecture", return_value=result) as run, patch.object(
                train_multi, "calculate_multi_class_weights"
            ) as weights:
                train_multi.main()
                dataset.assert_not_called()
                run.assert_not_called()
                weights.assert_not_called()

                clear_evaluation(output_dir)
                train_multi.main()
                dataset.assert_called_once_with(split="train", organ=None, augment=False, image_size=(4, 4))
                weights.assert_not_called()
                self.assertIsNone(run.call_args.args[3])  # No training loader.
                self.assertIsNone(run.call_args.args[4])  # No validation loader.
                self.assertIsNone(run.call_args.args[5])  # Saved checkpoint supplies weights.

                dataset.reset_mock()
                (output_dir / "best.pt").unlink()
                with self.assertRaises(SystemExit):
                    train_multi.main()
                dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
