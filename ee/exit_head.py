import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm matching Llama-3 implementation."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


class ExitHead(nn.Module):
    """
    Early-exit classifier head: RMSNorm -> Linear(hidden_size, vocab_size).

    Mirrors the base model's final path (model.norm -> lm_head) so that
    each exit head learns to project an intermediate hidden state into
    vocabulary logits the same way the full model does at the last layer.
    """

    def __init__(self, hidden_size: int, vocab_size: int, norm_eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, hidden_size) -> (batch, seq_len, vocab_size)"""
        return self.linear(self.norm(hidden_states))

    @classmethod
    def from_base_model(
        cls,
        base_model,
        hidden_size: int,
        vocab_size: int,
        norm_eps: float = 1e-5,
    ) -> "ExitHead":
        """Initialize an ExitHead with weights copied from the base model's
        final norm + lm_head.  Gives exit heads a strong starting point."""
        head = cls(hidden_size, vocab_size, norm_eps)
        # Copy final RMSNorm weights
        head.norm.weight.data.copy_(base_model.model.norm.weight.data)
        # Copy lm_head weights
        head.linear.weight.data.copy_(base_model.lm_head.weight.data)
        return head
