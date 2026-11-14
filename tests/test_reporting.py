import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from dsad_rep.reporting import (
    METRICS,
    clear_evaluation,
    evaluation_matches,
    evaluation_splits,
    load_evaluation,
    print_summary,
    save_evaluation,
    save_summary,
)


def sample_result():
    return {
        "model_type": "combined", "architecture": "segformer", "organ": None,
        "checkpoint_epoch": 2, "completed_epochs": 3, "image_size": [32, 40],
        "evaluations": {
            "train": {
                **{metric: 0.5 for metric in METRICS},
                "precision": float("nan"),
                "per_organ": {"liver": {metric: 0.75 for metric in METRICS}},
            },
        },
    }


class ReportingTests(unittest.TestCase):
    def test_json_csv_and_terminal_preserve_undefined_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_evaluation(path, sample_result())
            # Reject nonstandard JSON constants to ensure portable output.
            result = json.loads(
                (path / "evaluation.json").read_text(),
                parse_constant=lambda value: self.fail(f"Invalid JSON constant {value}"),
            )
            self.assertIsNone(result["evaluations"]["train"]["precision"])
            self.assertEqual(load_evaluation(path), result)
            with (path / "evaluation.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["organ"] for row in rows], ["macro", "liver"])
            self.assertEqual(rows[0]["precision"], "")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                print_summary([result])
            self.assertIn("n/a", output.getvalue())
            self.assertIn("liver", output.getvalue())
            self.assertFalse(list(path.glob("*.tmp")))

    def test_corrupt_and_incomplete_results_are_not_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            self.assertIsNone(load_evaluation(directory))
            for value in ("{broken", "null", "[]", '{"evaluations": {}}'):
                path.write_text(value)
                self.assertIsNone(load_evaluation(directory))
            result = sample_result()
            result["evaluations"]["train"]["loss"] = "broken"
            path.write_text(json.dumps(result))
            self.assertIsNone(load_evaluation(directory))

    def test_replaced_checkpoint_invalidates_saved_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_evaluation(path, sample_result())
            best = path / "best.pt"
            best.write_bytes(b"replacement checkpoint")
            timestamp = (path / "evaluation.json").stat().st_mtime_ns + 1_000_000
            os.utime(best, ns=(timestamp, timestamp))
            self.assertIsNone(load_evaluation(path))

    def test_cache_must_match_training_identity_and_requested_splits(self):
        result = sample_result()
        settings = {
            "model_type": "combined", "architecture": "segformer", "organ": None,
            "completed_epochs": 3, "image_size": (32, 40), "splits": ("train",),
        }
        self.assertTrue(evaluation_matches(result, **settings))
        for changed in (
            {"completed_epochs": 4}, {"architecture": "deeplabv3"},
            {"image_size": (40, 32)}, {"splits": ("train", "test")},
        ):
            self.assertFalse(evaluation_matches(result, **{**settings, **changed}))
        result["checkpoint_epoch"] = 4
        self.assertFalse(evaluation_matches(result, **settings))

    def test_summary_and_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_evaluation(path, sample_result())
            save_summary(path, [sample_result()], name="single_summary")
            self.assertTrue((path / "single_summary.csv").is_file())
            self.assertEqual(len(json.loads((path / "single_summary.json").read_text())), 1)
            clear_evaluation(path)
            clear_evaluation(path)
            self.assertIsNone(load_evaluation(path))
            self.assertFalse((path / "evaluation.csv").exists())
            self.assertTrue((path / "single_summary.json").exists())

    def test_split_selection_is_explicit(self):
        self.assertEqual(evaluation_splits("train"), ("train",))
        self.assertEqual(evaluation_splits("test"), ("test",))
        self.assertEqual(evaluation_splits("both"), ("train", "test"))
        with self.assertRaises(ValueError):
            evaluation_splits("validation")


if __name__ == "__main__":
    unittest.main()
