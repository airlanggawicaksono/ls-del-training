from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .exit_head import ExitHead


@dataclass
class EarlyExitOutput:
    """Container for all outputs from a forward pass with early exits."""

    base_logits: torch.Tensor  # (B, S, V) from final layer
    exit_logits: Dict[int, torch.Tensor]  # {layer_idx: (B, S, V)}
    base_loss: Optional[torch.Tensor] = None


class EarlyExitLlamaWrapper(nn.Module):
    """
    Wraps a frozen LlamaForCausalLM with trainable exit heads at
    specified transformer layers.

    Uses forward hooks to capture intermediate hidden states during the
    base model's forward pass.  Only the exit heads receive gradients;
    the base model is completely frozen and untouched.

    Hook approach (vs output_hidden_states=True):
      - Captures only the layers we need, not all 32
      - Saves ~464 MB VRAM for 3 exits on Llama-3-8B @ seq_len 2048
      - Base model code is never modified
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        exit_layer_indices: List[int],
        hidden_size: int,
        vocab_size: int,
        norm_eps: float = 1e-5,
        init_from_base: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.exit_layer_indices = sorted(exit_layer_indices)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Build exit heads
        if init_from_base:
            heads = {
                str(idx): ExitHead.from_base_model(
                    base_model, hidden_size, vocab_size, norm_eps
                )
                for idx in self.exit_layer_indices
            }
        else:
            heads = {
                str(idx): ExitHead(hidden_size, vocab_size, norm_eps)
                for idx in self.exit_layer_indices
            }
        self.exit_heads = nn.ModuleDict(heads)

        # Internal state for hook-captured hidden states
        self._intermediate_outputs: Dict[int, torch.Tensor] = {}
        self._hooks = []

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hooks(self) -> None:
        """Attach forward hooks to the transformer layers we want to tap."""
        self.remove_hooks()
        for idx in self.exit_layer_indices:
            layer = self.base_model.model.layers[idx]
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_hook(self, layer_idx: int):
        """Return a hook fn that stores the detached hidden state."""

        def hook_fn(module, input, output):
            # Llama layer output: (hidden_states, self_attn_weights, present_kv)
            # We detach so no autograd graph is built for the frozen backbone.
            hidden = output[0].detach()
            self._intermediate_outputs[layer_idx] = hidden

        return hook_fn

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> EarlyExitOutput:
        # Run full forward through frozen base model.
        # no_grad is safe because we detach hook outputs anyway, and the
        # base model params have requires_grad=False.
        with torch.no_grad():
            with torch.profiler.record_function("base_model_forward"):
                base_out = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

        exit_logits: Dict[int, torch.Tensor] = {}
        with torch.profiler.record_function("exit_heads_forward"):
            for idx in self.exit_layer_indices:
                hidden = self._intermediate_outputs[idx]
                hidden = hidden.requires_grad_(True)
                exit_logits[idx] = self.exit_heads[str(idx)](hidden)

        self._intermediate_outputs.clear()

        return EarlyExitOutput(
            base_logits=base_out.logits.detach(),
            exit_logits=exit_logits,
            base_loss=base_out.loss.detach() if base_out.loss is not None else None,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def config(self):
        """Expose base model config so HF Trainer doesn't break."""
        return self.base_model.config

    @property
    def device(self):
        return next(self.exit_heads.parameters()).device
