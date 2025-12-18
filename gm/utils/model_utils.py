import torch
import torch.nn as nn


def save_checkpoint(path, model, optimizer=None, extra: dict | None = None):
    """
    Сохраняет модель, оптимизатор и доп. информацию в один .pt файл.

    path: путь к файлу (str)
    model: nn.Module
    optimizer: torch.optim.Optimizer или None
    extra: словарь с доп. информацией (config, epoch, метрики и т.п.)
    """
    if extra is None:
        extra = {}

    checkpoint = {
        "model_state": model.state_dict(),
        "extra": extra,
    }

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    else:
        checkpoint["optimizer_state"] = None

    torch.save(checkpoint, path)


def load_checkpoint(
        path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
        weights_only=True,
):
    """
    Загружает веса в уже созданный model (и опционально optimizer).
    Возвращает (model, optimizer, extra).

    Важно: модель и оптимизатор должны быть уже созданы с тем же архитектурой/параметрами.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    extra = checkpoint.get("extra", {})
    return model, optimizer, extra
