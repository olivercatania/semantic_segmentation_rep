"""Device selection and optional acceleration shared by all entry points."""

from contextlib import nullcontext

import torch


def _available(device_type):
    backend = (
        torch.backends.mps
        if device_type == "mps"
        else getattr(torch, device_type, None)
    )
    check = getattr(backend, "is_available", None)
    if check is None:
        return False
    try:
        return check()
    except (RuntimeError, AssertionError):
        return False


def select_device(requested="auto"):
    """Prefer the installed runtime's available accelerator, otherwise CPU.

    Explicit devices fail clearly if unavailable; auto never assumes CUDA is
    installed merely because the machine has a GPU.
    """
    if requested == "auto":
        try:
            device = torch.accelerator.current_accelerator(check_available=True)
        except (RuntimeError, AssertionError):
            device = None
        if device is not None:
            return device
        for device_type in ("cuda", "xpu", "mps"):
            if _available(device_type):
                return torch.device(device_type)
        return torch.device("cpu")

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid device: {requested!r}") from error
    if device.type != "cpu" and not _available(device.type):
        raise ValueError(
            f"Device {requested!r} is unavailable in this PyTorch installation. "
            "Use --device auto or --device cpu."
        )
    if device.index is not None and device.type != "cpu":
        backend = getattr(torch, device.type, None)
        count = getattr(backend, "device_count", lambda: 1)()
        if device.index >= count:
            raise ValueError(f"Device index is unavailable: {requested!r}")
    return device


def autocast_context(device):
    # Full precision is the portable default for CPU, MPS, and other backends.
    if device.type in ("cuda", "xpu"):
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def make_scaler(device):
    if device.type in ("cuda", "xpu"):
        return torch.amp.GradScaler(device.type)
    return None


def pin_memory(device):
    return device.type in ("cuda", "xpu")


def release_device_cache(device):
    if device.type == "cpu":
        return
    backend = getattr(torch, device.type, None)
    empty_cache = getattr(backend, "empty_cache", None)
    if empty_cache is not None:
        empty_cache()
