import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from dsad_rep import devices
from dsad_rep.engine import train_one_epoch, validate_one_epoch
from dsad_rep.metrics import binary_segmentation_metrics


class DeviceTests(unittest.TestCase):
    def test_auto_uses_available_runtime(self):
        for name in ("cuda", "mps", "xpu"):
            with self.subTest(device=name), patch.object(
                torch.accelerator, "current_accelerator", return_value=torch.device(name)
            ):
                self.assertEqual(devices.select_device().type, name)

    def test_cpu_fallback(self):
        with patch.object(torch.accelerator, "current_accelerator", return_value=None), patch(
            "dsad_rep.devices._available", return_value=False
        ):
            self.assertEqual(devices.select_device().type, "cpu")

    def test_backend_fallback(self):
        with patch.object(
            torch.accelerator, "current_accelerator", side_effect=RuntimeError("no runtime")
        ), patch("dsad_rep.devices._available", side_effect=lambda name: name == "mps"):
            self.assertEqual(devices.select_device().type, "mps")

    def test_explicit_cpu_does_not_probe_accelerator(self):
        with patch.object(torch.accelerator, "current_accelerator") as probe:
            self.assertEqual(devices.select_device("cpu").type, "cpu")
            probe.assert_not_called()

    def test_unavailable_explicit_device_fails(self):
        with patch("dsad_rep.devices._available", return_value=False):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                devices.select_device("cuda")

    def test_invalid_device_index_fails(self):
        with patch("dsad_rep.devices._available", return_value=True), patch.object(
            torch.cuda, "device_count", return_value=1
        ):
            with self.assertRaisesRegex(ValueError, "index"):
                devices.select_device("cuda:2")

    def test_cpu_and_mps_never_construct_cuda_amp(self):
        with patch.object(torch, "autocast") as autocast, patch.object(
            torch.amp, "GradScaler"
        ) as scaler:
            for name in ("cpu", "mps"):
                device = torch.device(name)
                with devices.autocast_context(device):
                    pass
                self.assertIsNone(devices.make_scaler(device))
                self.assertFalse(devices.pin_memory(device))
            autocast.assert_not_called()
            scaler.assert_not_called()

    def test_xpu_uses_its_own_amp_backend(self):
        with patch.object(torch, "autocast") as autocast, patch.object(
            torch.amp, "GradScaler"
        ) as scaler:
            device = torch.device("xpu")
            devices.autocast_context(device)
            devices.make_scaler(device)
            autocast.assert_called_once_with(device_type="xpu", dtype=torch.float16)
            scaler.assert_called_once_with("xpu")
            self.assertTrue(devices.pin_memory(device))

    def test_cache_cleanup_uses_selected_backend(self):
        with patch.object(torch.cuda, "empty_cache") as cuda, patch.object(
            torch.mps, "empty_cache"
        ) as mps:
            devices.release_device_cache(torch.device("cpu"))
            cuda.assert_not_called()
            mps.assert_not_called()
            devices.release_device_cache(torch.device("mps"))
            mps.assert_called_once_with()
            cuda.assert_not_called()


class EngineTests(unittest.TestCase):
    def test_cpu_training_and_full_evaluation(self):
        torch.manual_seed(42)
        samples = [
            {"image": torch.randn(3, 4, 4), "mask": torch.randint(0, 2, (4, 4))}
            for _ in range(3)
        ]
        loader = DataLoader(samples, batch_size=2, drop_last=False)
        model = torch.nn.Conv2d(3, 2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        before = model.weight.detach().clone()
        loss = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model.weight))
        seen = []
        hook = model.register_forward_hook(lambda module, args, result: seen.append(len(args[0])))
        summary = validate_one_epoch(
            model, loader, criterion, torch.device("cpu"), desc="Train evaluation"
        )
        hook.remove()
        self.assertEqual(sum(seen), 3)
        self.assertFalse(model.training)
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(set(summary), {"loss", "samples", "iou", "f1", "precision", "recall", "specificity"})

    def test_empty_loader_has_clear_error(self):
        loader = DataLoader([], batch_size=2)
        model = torch.nn.Conv2d(3, 2, 1)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, "no batches"):
            train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "no images"):
            validate_one_epoch(model, loader, criterion, torch.device("cpu"))

    def test_precision_differs_from_recall_when_errors_differ(self):
        predictions = torch.tensor([[[1, 1, 1, 0]]])
        masks = torch.tensor([[[1, 0, 0, 0]]])
        result = binary_segmentation_metrics(predictions, masks)[0]
        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertEqual(result["recall"], 1)


if __name__ == "__main__":
    unittest.main()
