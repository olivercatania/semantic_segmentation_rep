import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dsad_rep.reporting import METRICS, save_evaluation
from scripts import train_all, train_all_single


def result(model_type="single", splits=("train",)):
    return {
        "model_type": model_type,
        "architecture": "deeplabv3",
        "organ": "liver" if model_type == "single" else None,
        "checkpoint_epoch": 1, "completed_epochs": 2, "image_size": [32, 40],
        "evaluations": {split: {metric: 0.5 for metric in METRICS} for split in splits},
    }


SETTINGS = {
    "architecture": "deeplabv3", "organ": "liver", "epochs": 2,
    "batch_size": 2, "height": 32, "width": 40, "num_workers": 3,
}


def arguments(output_dir):
    return [
        "launcher", "--architectures", "deeplabv3", "--organs", "liver",
        "--epochs", "2", "--height", "32", "--width", "40",
        "--num-workers", "3", "--device", "cpu", "--output-dir", str(output_dir),
    ]


class LauncherTests(unittest.TestCase):
    def test_dry_runs_do_not_write_or_start_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "new"
            for module in (train_all, train_all_single):
                with patch("sys.argv", arguments(output_dir) + ["--dry-run"]), \
                     patch.object(module.subprocess, "run") as run, \
                     patch.object(module.subprocess, "Popen") as popen, \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    module.main()
                run.assert_not_called()
                popen.assert_not_called()
                self.assertIn("--num-workers 3", output.getvalue())
                self.assertIn("--device cpu", output.getvalue())
                self.assertIn("--eval-split train", output.getvalue())
                self.assertFalse(output_dir.exists())

    def test_legacy_completed_model_is_evaluated_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "deeplabv3" / "liver"
            model_dir.mkdir(parents=True)
            (model_dir / "best.pt").write_bytes(b"existing checkpoint")
            marker = model_dir / "complete.json"
            marker.write_text(json.dumps(SETTINGS))

            def evaluate(command, log_path):
                self.assertIn("--evaluate-only", command)
                self.assertNotIn("--fresh", command)
                self.assertTrue(marker.exists(), "Legacy completion evidence must survive evaluation")
                self.assertEqual(command[command.index("--num-workers") + 1], "3")
                save_evaluation(model_dir, result())
                return 0

            with patch("sys.argv", arguments(root)), \
                 patch.object(train_all_single, "run_logged", side_effect=evaluate) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                train_all_single.main()
            run.assert_called_once()
            self.assertEqual(json.loads(marker.read_text()), SETTINGS)
            self.assertEqual(len(json.loads((root / "single_summary.json").read_text())), 1)

    def test_cached_results_skip_training_and_filter_unrequested_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "deeplabv3" / "liver"
            model_dir.mkdir(parents=True)
            (model_dir / "best.pt").write_bytes(b"checkpoint")
            (model_dir / "complete.json").write_text(json.dumps(SETTINGS))
            save_evaluation(model_dir, result(splits=("train", "test")))
            with patch("sys.argv", arguments(root)), \
                 patch.object(train_all_single, "run_logged") as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                train_all_single.main()
            run.assert_not_called()
            summary = json.loads((root / "single_summary.json").read_text())
            self.assertEqual(list(summary[0]["evaluations"]), ["train"])

    def test_force_passes_fresh_and_waits_for_evaluation_before_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "deeplabv3" / "liver"
            model_dir.mkdir(parents=True)
            marker = model_dir / "complete.json"
            marker.write_text(json.dumps(SETTINGS))

            def train(command, log_path):
                self.assertIn("--fresh", command)
                self.assertNotIn("--evaluate-only", command)
                self.assertFalse(marker.exists())
                (model_dir / "best.pt").write_bytes(b"new checkpoint")
                save_evaluation(model_dir, result())
                return 0

            with patch("sys.argv", arguments(root) + ["--force"]), \
                 patch.object(train_all_single, "run_logged", side_effect=train), \
                 contextlib.redirect_stdout(io.StringIO()):
                train_all_single.main()
            self.assertTrue(marker.exists())

    def test_failed_job_does_not_mark_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("sys.argv", arguments(root)), \
                 patch.object(train_all_single, "run_logged", return_value=1), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaises(SystemExit):
                train_all_single.main()
            self.assertFalse((root / "deeplabv3" / "liver" / "complete.json").exists())
            self.assertFalse((root / "single_summary.json").exists())

    def test_top_launcher_summarizes_only_requested_models_after_both_suites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = []

            def run(command, check):
                commands.append(command)
                self.assertTrue(check)
                self.assertFalse((root / "summary.json").exists())
                self.assertEqual(command[command.index("--num-workers") + 1], "3")
                if "scripts.train_all_single" in command:
                    self.assertIn("--force", command)
                    save_evaluation(root / "deeplabv3" / "liver", result())
                    # An unrelated earlier experiment must not enter the report.
                    save_evaluation(root / "segformer" / "liver", result())
                else:
                    self.assertIn("--fresh", command)
                    self.assertNotIn("--organs", command)
                    self.assertEqual(command[command.index("--batch-size") + 1], "1")
                    save_evaluation(root / "combined" / "deeplabv3", result("combined"))

            with patch("sys.argv", arguments(root) + ["--force"]), \
                 patch.object(train_all.subprocess, "run", side_effect=run), \
                 contextlib.redirect_stdout(io.StringIO()):
                train_all.main()
            self.assertEqual(len(commands), 2)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual([entry["model_type"] for entry in summary], ["single", "combined"])


if __name__ == "__main__":
    unittest.main()
