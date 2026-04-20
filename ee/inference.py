import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .exit_head import ExitHead

_nvml_available = False
try:
    import pynvml

    pynvml.nvmlInit()
    _nvml_available = True
except Exception:
    pass


def _gpu_power_watts() -> float:
    """Current GPU power draw in watts. Returns 0 if unavailable."""
    if not _nvml_available:
        return 0.0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
    except Exception:
        return 0.0


class EarlyExitGenerator:
    """
    Generates text using confidence-based early exit.

    Tracks latency (TTFT, per-token, end-to-end) and energy (tokens/joule).
    No KV-cache (simple version for evaluation / comparison).
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        exit_heads: Dict[int, ExitHead],
        tokenizer: PreTrainedTokenizerBase,
        confidence_threshold: float = 0.9,
        compile_runtime: bool = False,
    ):
        self.base_model = base_model
        self.exit_heads = exit_heads
        self.tokenizer = tokenizer
        self.confidence_threshold = confidence_threshold
        self.exit_layer_indices = sorted(exit_heads.keys())
        self.num_layers = len(base_model.model.layers)

        # Stats
        self.exit_counts: Dict[int, int] = defaultdict(int)
        self.total_tokens = 0

        # Compile the actual EE step function used during generation.
        # This makes EE benchmarking more comparable to compiled baseline runs.
        self._step_fn = self._forward_with_early_exit
        if compile_runtime:
            self._step_fn = torch.compile(self._forward_with_early_exit, dynamic=True)

    def _forward_with_early_exit(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[int, int, float]:
        device = input_ids.device
        hidden_states = self.base_model.model.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cache_position = torch.arange(seq_len, device=device)

        position_embeddings = self.base_model.model.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.base_model.model.layers):
            layer_out = layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
            )
            hidden_states = layer_out[0]

            if layer_idx in self.exit_heads:
                head = self.exit_heads[layer_idx]
                logits = head(hidden_states[:, -1:, :])
                probs = F.softmax(logits.squeeze(1), dim=-1)
                max_prob, token_id = probs.max(dim=-1)

                if max_prob.item() >= self.confidence_threshold:
                    return token_id.item(), layer_idx, max_prob.item()

        hidden_states = self.base_model.model.norm(hidden_states)
        logits = self.base_model.lm_head(hidden_states[:, -1:, :])
        probs = F.softmax(logits.squeeze(1), dim=-1)
        max_prob, token_id = probs.max(dim=-1)
        return token_id.item(), self.num_layers - 1, max_prob.item()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> Dict:
        """
        Generate text with early exit, tracking latency and energy.

        Returns dict with:
            text, tokens, exit_layers, confidences, exit_stats,
            ttft_sec, per_token_latency_sec, end_to_end_sec,
            total_energy_j, tokens_per_joule
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.base_model.device)

        generated_tokens: List[int] = []
        exit_layers: List[int] = []
        confidences: List[float] = []
        per_token_times: List[float] = []

        torch.cuda.synchronize()
        e2e_start = time.perf_counter()
        energy_samples: List[Tuple[float, float]] = []  # (timestamp, watts)

        for step in range(max_new_tokens):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            power_before = _gpu_power_watts()

            token_id, exit_layer, conf = self._step_fn(input_ids)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            power_after = _gpu_power_watts()

            step_time = t1 - t0
            per_token_times.append(step_time)

            # Sample energy: avg power * time for this step
            avg_power = (power_before + power_after) / 2.0
            energy_samples.append((step_time, avg_power))

            generated_tokens.append(token_id)
            exit_layers.append(exit_layer)
            confidences.append(conf)
            self.exit_counts[exit_layer] += 1
            self.total_tokens += 1

            if token_id == self.tokenizer.eos_token_id:
                break

            next_token = torch.tensor([[token_id]], device=input_ids.device)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        torch.cuda.synchronize()
        e2e_end = time.perf_counter()

        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        n_tokens = len(generated_tokens)
        e2e_sec = e2e_end - e2e_start
        ttft = per_token_times[0] if per_token_times else 0.0
        avg_per_token = sum(per_token_times[1:]) / max(len(per_token_times) - 1, 1) if n_tokens > 1 else ttft

        # Energy: sum(power_w * dt) for each step
        total_energy_j = sum(dt * pw for dt, pw in energy_samples)
        tokens_per_joule = n_tokens / total_energy_j if total_energy_j > 0 else 0.0

        return {
            "text": text,
            "tokens": generated_tokens,
            "n_tokens": n_tokens,
            "exit_layers": exit_layers,
            "confidences": confidences,
            "exit_stats": dict(self.exit_counts),
            # Latency
            "ttft_sec": round(ttft, 6),
            "per_token_latency_sec": round(avg_per_token, 6),
            "end_to_end_sec": round(e2e_sec, 6),
            # Energy
            "total_energy_j": round(total_energy_j, 4),
            "tokens_per_joule": round(tokens_per_joule, 2),
        }

    def print_exit_statistics(self) -> None:
        if self.total_tokens == 0:
            print("No tokens generated yet.")
            return

        print(f"\nExit statistics ({self.total_tokens} tokens total):")
        print("  Layer | Count | Percent")
        print("  ------+-------+--------")
        for layer_idx in sorted(self.exit_counts):
            count = self.exit_counts[layer_idx]
            pct = 100.0 * count / self.total_tokens
            tag = "EE" if layer_idx in self.exit_layer_indices else "FL"
            print(f"  {tag} {layer_idx:3d} | {count:5d} | {pct:5.1f}%")

    def reset_statistics(self) -> None:
        self.exit_counts.clear()
        self.total_tokens = 0


class BaselineGenerator:
    """
    Standard full-model generation with the same latency/energy tracking.
    Uses model.generate() for fair comparison with torch.compile.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.model = model
        self.tokenizer = tokenizer
        # Resolve layer count — works on both raw and torch.compile'd models
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        self._num_layers = len(raw.model.layers) - 1 if hasattr(raw, "model") else 31
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> Dict:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        prompt_len = input_ids.shape[1]

        # Single generate call
        torch.cuda.synchronize()
        power_start = _gpu_power_watts()
        e2e_start = time.perf_counter()

        out = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        torch.cuda.synchronize()
        e2e_end = time.perf_counter()
        power_end = _gpu_power_watts()

        generated_ids = out[0, prompt_len:].tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        n_tokens = len(generated_ids)
        e2e_sec = e2e_end - e2e_start

        # Approximate TTFT and per-token latency from total time
        # (model.generate doesn't expose per-step timing)
        ttft = e2e_sec / max(n_tokens, 1)  # first token ≈ avg when no KV warmup split
        avg_per_token = (e2e_sec - ttft) / max(n_tokens - 1, 1) if n_tokens > 1 else ttft

        # Energy: sample power at start + end, trapezoidal
        avg_power = (power_start + power_end) / 2.0
        total_energy_j = avg_power * e2e_sec
        tokens_per_joule = n_tokens / total_energy_j if total_energy_j > 0 else 0.0

        return {
            "text": text,
            "tokens": generated_ids,
            "n_tokens": n_tokens,
            "exit_layers": [self._num_layers] * n_tokens,
            # Latency
            "ttft_sec": round(ttft, 6),
            "per_token_latency_sec": round(avg_per_token, 6),
            "end_to_end_sec": round(e2e_sec, 6),
            # Energy
            "total_energy_j": round(total_energy_j, 4),
            "tokens_per_joule": round(tokens_per_joule, 2),
        }
