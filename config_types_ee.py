from typing import List, Optional

from pydantic import Field, model_validator

from config_types import TrainConfig


class EETrainConfig(TrainConfig):
    """Extends TrainConfig with early-exit-specific parameters."""

    # Which transformer layers get exit heads (0-indexed)
    exit_layer_indices: List[int] = Field(default_factory=lambda: [8, 16, 24])

    # Per-exit loss weights (positionally matched to exit_layer_indices)
    exit_loss_weights: List[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])

    # Initialize exit heads from base model's final norm + lm_head
    init_exit_from_base: bool = True

    # Cap training samples (None = use all). For C4 300M tokens ≈ 1_200_000 docs
    max_train_samples: Optional[int] = None

    # Cap validation samples
    max_val_samples: Optional[int] = 5000

    # Confidence threshold for early exit during inference
    exit_confidence_threshold: float = 0.9

    # HuggingFace Hub repo for exit heads (separate from base model)
    hub_exit_heads_repo: Optional[str] = None

    # Override: higher LR since we train small heads from scratch
    learning_rate: float = 1e-4

    # Override: no gradient checkpointing (backbone frozen, no benefit)
    gradient_checkpointing: bool = False

    @model_validator(mode="after")
    def validate_exit_config(self) -> "EETrainConfig":
        if len(self.exit_loss_weights) != len(self.exit_layer_indices):
            raise ValueError(
                f"exit_loss_weights length ({len(self.exit_loss_weights)}) must "
                f"match exit_layer_indices length ({len(self.exit_layer_indices)})"
            )
        for idx in self.exit_layer_indices:
            if not (0 <= idx < 32):
                raise ValueError(f"exit_layer_index {idx} out of range [0, 32)")
        return self
