import os
from typing import Dict, Optional

from transformers import Trainer

from .hub import save_exit_heads
from .loss import compute_multi_exit_loss
from .model_wrapper import EarlyExitLlamaWrapper


class EarlyExitTrainer(Trainer):
    """
    HuggingFace Trainer subclass for early-exit training.

    Overrides:
      - compute_loss: multi-exit weighted cross-entropy
      - _save: only saves exit head weights, not the 8B backbone
    """

    def __init__(
        self,
        exit_weights: Dict[int, float],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.exit_weights = exit_weights
        self._last_ee_metrics: Dict[str, float] = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )

        loss, metrics = compute_multi_exit_loss(
            exit_logits=outputs.exit_logits,
            labels=labels,
            exit_weights=self.exit_weights,
            base_logits=outputs.base_logits,
        )

        # Store for logging in on_log
        self._last_ee_metrics = metrics

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        """Inject per-exit loss metrics into every log call."""
        logs.update(self._last_ee_metrics)
        super().log(logs, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        """Save only exit heads + tokenizer, not the full base model."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Save exit head weights
        save_exit_heads(self.model, output_dir)

        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
