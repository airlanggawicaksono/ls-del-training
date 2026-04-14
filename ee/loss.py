from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_multi_exit_loss(
    exit_logits: Dict[int, torch.Tensor],
    labels: torch.Tensor,
    exit_weights: Dict[int, float],
    base_logits: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Weighted sum of cross-entropy losses across all exit heads.

        L = sum(w_i * CE(exit_logits_i, labels))  for each exit i

    Args:
        exit_logits:  {layer_idx: (B, S, V)} from each exit head
        labels:       (B, S) token ids, -100 for ignored positions
        exit_weights: {layer_idx: float} loss weight per exit
        base_logits:  (B, S, V) from final layer (tracked, no gradient)

    Returns:
        total_loss: scalar for backprop
        metrics:    per-exit losses + base loss for logging
    """
    total_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)
    metrics: Dict[str, float] = {}

    for layer_idx, logits in exit_logits.items():
        # Standard causal LM shift: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        weight = exit_weights.get(layer_idx, 1.0)
        total_loss = total_loss + weight * loss
        metrics[f"loss_exit_{layer_idx}"] = loss.item()

    # Track base model (final layer) loss for comparison — no gradient
    if base_logits is not None:
        with torch.no_grad():
            shift_base = base_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            base_loss = F.cross_entropy(
                shift_base.view(-1, shift_base.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            metrics["loss_base_final"] = base_loss.item()

    metrics["loss_total"] = total_loss.item()
    return total_loss, metrics
