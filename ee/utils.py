from typing import Dict

import torch.nn as nn
from transformers import PreTrainedModel


def freeze_base_model(model: PreTrainedModel) -> None:
    """Freeze all parameters and set to eval mode (disables dropout)."""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total, trainable, and frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
