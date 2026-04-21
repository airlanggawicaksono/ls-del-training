import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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

    use_kv_cache=True (default): prefill runs full model once, decode steps
    process one token at a time with KV cache. Layers past the exit point get
    stale cache entries — an acceptable approximation for most prompts.

    use_kv_cache=False: reprocesses full context every step (slow but exact).
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        exit_heads: Dict[int, ExitHead],
        tokenizer: PreTrainedTokenizerBase,
        confidence_threshold: float = 0.9,
        use_kv_cache: bool = True,
        force_exit_layer: Optional[int] = None,
    ):
        self.base_model = base_model
        self.exit_heads = exit_heads
        self.tokenizer = tokenizer
        self.confidence_threshold = confidence_threshold
        self.use_kv_cache = use_kv_cache
        self.force_exit_layer = force_exit_layer
        self.exit_layer_indices = sorted(exit_heads.keys())
        self.num_layers = len(base_model.model.layers)

        if force_exit_layer is not None and force_exit_layer not in exit_heads:
            raise ValueError(f"force_exit_layer={force_exit_layer} not in exit_heads {list(exit_heads.keys())}")

        self.exit_counts: Dict[int, int] = defaultdict(int)
        self.total_tokens = 0

        self._step_fn = self._forward_with_early_exit

    # ------------------------------------------------------------------
    # No-KV path (original, kept for reference / compile_runtime mode)
    # ------------------------------------------------------------------

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
            hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

            if layer_idx in self.exit_heads:
                head = self.exit_heads[layer_idx]
                logits = head(hidden_states[:, -1:, :])
                probs = F.softmax(logits.squeeze(1), dim=-1)
                max_prob, token_id = probs.max(dim=-1)

                forced = self.force_exit_layer is not None and layer_idx == self.force_exit_layer
                if forced or max_prob.item() >= self.confidence_threshold:
                    return token_id.item(), layer_idx, max_prob.item()

        hidden_states = self.base_model.model.norm(hidden_states)
        logits = self.base_model.lm_head(hidden_states[:, -1:, :])
        probs = F.softmax(logits.squeeze(1), dim=-1)
        max_prob, token_id = probs.max(dim=-1)
        return token_id.item(), self.num_layers - 1, max_prob.item()

    # ------------------------------------------------------------------
    # KV-cache path
    # ------------------------------------------------------------------

    def _prefill(
        self, input_ids: torch.Tensor
    ) -> Tuple[int, float, object]:
        """Full forward pass on prompt. Returns (first_token_id, conf, kv_cache)."""
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        device = input_ids.device
        hidden_states = self.base_model.model.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.base_model.model.rotary_emb(hidden_states, position_ids)

        for layer in self.base_model.model.layers:
            layer_out = layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                past_key_value=cache,
                use_cache=True,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

        hidden_states = self.base_model.model.norm(hidden_states)
        logits = self.base_model.lm_head(hidden_states[:, -1:, :])
        probs = F.softmax(logits.squeeze(1), dim=-1)
        max_prob, token_id = probs.max(dim=-1)
        return token_id.item(), max_prob.item(), cache

    def _decode_one_kv(
        self,
        token_id: int,
        past_key_values: object,
        past_len: int,
    ) -> Tuple[int, int, float]:
        """Single-token decode with KV cache + early exit.

        Layers past the exit point do not have their caches updated for this
        token (stale KV approximation). Quality impact is minor when exit
        distribution is stable across steps.
        """
        device = self.base_model.device
        hidden_states = self.base_model.model.embed_tokens(
            torch.tensor([[token_id]], device=device)
        )
        cache_position = torch.tensor([past_len], device=device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.base_model.model.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.base_model.model.layers):
            layer_out = layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                past_key_value=past_key_values,
                use_cache=True,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

            if layer_idx in self.exit_heads:
                head = self.exit_heads[layer_idx]
                logits = head(hidden_states[:, -1:, :])
                probs = F.softmax(logits.squeeze(1), dim=-1)
                max_prob, next_id = probs.max(dim=-1)
                forced = self.force_exit_layer is not None and layer_idx == self.force_exit_layer
                if forced or max_prob.item() >= self.confidence_threshold:
                    return next_id.item(), layer_idx, max_prob.item()

        hidden_states = self.base_model.model.norm(hidden_states)
        logits = self.base_model.lm_head(hidden_states[:, -1:, :])
        probs = F.softmax(logits.squeeze(1), dim=-1)
        max_prob, next_id = probs.max(dim=-1)
        return next_id.item(), self.num_layers - 1, max_prob.item()

    # ------------------------------------------------------------------
    # Main generate
    # ------------------------------------------------------------------

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
        call_exit_counts: Dict[int, int] = defaultdict(int)
        energy_samples: List[Tuple[float, float]] = []

        torch.cuda.synchronize()
        e2e_start = time.perf_counter()

        if self.use_kv_cache:
            # ---- Prefill ----
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            power_before = _gpu_power_watts()

            first_token_id, first_conf, past_kv = self._prefill(input_ids)
            past_len = input_ids.shape[1]

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            power_after = _gpu_power_watts()

            step_time = t1 - t0
            per_token_times.append(step_time)
            energy_samples.append((step_time, (power_before + power_after) / 2.0))

            # Prefill always uses full model
            exit_layer = self.num_layers - 1
            generated_tokens.append(first_token_id)
            exit_layers.append(exit_layer)
            confidences.append(first_conf)
            call_exit_counts[exit_layer] += 1
            self.exit_counts[exit_layer] += 1
            self.total_tokens += 1
            past_len += 1

            # ---- Decode ----
            if first_token_id != self.tokenizer.eos_token_id:
                for _ in range(1, max_new_tokens):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    power_before = _gpu_power_watts()

                    token_id, exit_layer, conf = self._decode_one_kv(
                        generated_tokens[-1], past_kv, past_len
                    )

                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    power_after = _gpu_power_watts()

                    step_time = t1 - t0
                    per_token_times.append(step_time)
                    energy_samples.append((step_time, (power_before + power_after) / 2.0))

                    generated_tokens.append(token_id)
                    exit_layers.append(exit_layer)
                    confidences.append(conf)
                    call_exit_counts[exit_layer] += 1
                    self.exit_counts[exit_layer] += 1
                    self.total_tokens += 1
                    past_len += 1

                    if token_id == self.tokenizer.eos_token_id:
                        break

        else:
            # ---- No-KV path: reprocess full context every step ----
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
                energy_samples.append((step_time, (power_before + power_after) / 2.0))

                generated_tokens.append(token_id)
                exit_layers.append(exit_layer)
                confidences.append(conf)
                call_exit_counts[exit_layer] += 1
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

        total_energy_j = sum(dt * pw for dt, pw in energy_samples)
        tokens_per_joule = n_tokens / total_energy_j if total_energy_j > 0 else 0.0

        return {
            "text": text,
            "tokens": generated_tokens,
            "n_tokens": n_tokens,
            "exit_layers": exit_layers,
            "confidences": confidences,
            "exit_stats": dict(call_exit_counts),
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


class MultiExitGenerator:
    """
    One forward pass per token — collects predictions from ALL exit heads simultaneously.
    KV cache is shared: layers 0-8 computed once, not repeated for exit_16 or exit_24.

    Base model (final layer) drives next-token selection so all exits see a
    consistent context. Per-exit texts show what each head would have predicted
    at each position given the same input sequence.

    Use this for quality comparison (ROUGE per exit). For per-exit latency
    measurement use EarlyExitGenerator with force_exit_layer instead.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        exit_heads: Dict[int, ExitHead],
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.base_model = base_model
        self.exit_heads = exit_heads
        self.tokenizer = tokenizer
        self.exit_layer_indices = sorted(exit_heads.keys())
        self.num_layers = len(base_model.model.layers)

    def _run_all_layers(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings,
        cache_position: torch.Tensor,
        past_key_values,
        t_start: float,
    ) -> Tuple[Dict[int, Tuple[int, float]], Dict[int, float], int, float]:
        """
        Run all layers in one pass.
        Returns:
            exit_preds:   {layer_idx: (token_id, conf)}
            exit_elapsed: {layer_idx: seconds_from_t_start_to_this_exit}
            base_token_id, base_elapsed
        """
        exit_preds: Dict[int, Tuple[int, float]] = {}
        exit_elapsed: Dict[int, float] = {}

        for layer_idx, layer in enumerate(self.base_model.model.layers):
            layer_out = layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                past_key_value=past_key_values,
                use_cache=True,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)

            if layer_idx in self.exit_heads:
                torch.cuda.synchronize()
                exit_elapsed[layer_idx] = time.perf_counter() - t_start
                head = self.exit_heads[layer_idx]
                logits = head(hidden_states[:, -1:, :])
                probs = F.softmax(logits.squeeze(1), dim=-1)
                max_prob, token_id = probs.max(dim=-1)
                exit_preds[layer_idx] = (token_id.item(), max_prob.item())

        hidden_states = self.base_model.model.norm(hidden_states)
        logits = self.base_model.lm_head(hidden_states[:, -1:, :])
        probs = F.softmax(logits.squeeze(1), dim=-1)
        max_prob, token_id = probs.max(dim=-1)
        torch.cuda.synchronize()
        base_elapsed = time.perf_counter() - t_start
        return exit_preds, exit_elapsed, token_id.item(), base_elapsed

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> Dict:
        """
        Returns per-exit generated texts + latency/energy.

        {
          "exits": {
            "exit_8":  {"text": ..., "tokens": [...]},
            "exit_16": {"text": ..., "tokens": [...]},
            "exit_24": {"text": ..., "tokens": [...]},
            "base":    {"text": ..., "tokens": [...]},
          },
          "n_tokens": ...,
          "ttft_sec": ..., "per_token_latency_sec": ..., "end_to_end_sec": ...,
          "total_energy_j": ..., "tokens_per_joule": ...,
        }
        """
        from transformers.cache_utils import DynamicCache

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.base_model.device)
        device = input_ids.device

        exit_token_lists: Dict[int, List[int]] = {idx: [] for idx in self.exit_layer_indices}
        base_token_list: List[int] = []
        # Per-exit step times and energy: {layer_idx: [(elapsed, power_w)]}
        exit_step_samples: Dict[int, List[Tuple[float, float]]] = {idx: [] for idx in self.exit_layer_indices}
        base_step_samples: List[Tuple[float, float]] = []
        cache = DynamicCache()

        torch.cuda.synchronize()
        e2e_start = time.perf_counter()

        def _step(hs, pos_ids, pos_emb, cache_pos):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            power_before = _gpu_power_watts()
            ep, ee, base_tid, base_el = self._run_all_layers(hs, pos_ids, pos_emb, cache_pos, cache, t0)
            power_after = _gpu_power_watts()
            avg_pw = (power_before + power_after) / 2.0
            return ep, ee, base_tid, base_el, avg_pw

        # ---- Prefill ----
        seq_len = input_ids.shape[1]
        hidden_states = self.base_model.model.embed_tokens(input_ids)
        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.base_model.model.rotary_emb(hidden_states, position_ids)

        exit_preds, exit_elapsed, base_token_id, base_el, avg_pw = _step(
            hidden_states, position_ids, position_embeddings, cache_position
        )
        past_len = seq_len + 1

        for idx, (tid, _) in exit_preds.items():
            exit_token_lists[idx].append(tid)
            exit_step_samples[idx].append((exit_elapsed[idx], avg_pw))
        base_token_list.append(base_token_id)
        base_step_samples.append((base_el, avg_pw))
        next_token_id = base_token_id

        # ---- Decode ----
        if base_token_id != self.tokenizer.eos_token_id:
            for _ in range(1, max_new_tokens):
                hidden_states = self.base_model.model.embed_tokens(
                    torch.tensor([[next_token_id]], device=device)
                )
                cache_position = torch.tensor([past_len - 1], device=device)
                position_ids = cache_position.unsqueeze(0)
                position_embeddings = self.base_model.model.rotary_emb(hidden_states, position_ids)

                exit_preds, exit_elapsed, base_token_id, base_el, avg_pw = _step(
                    hidden_states, position_ids, position_embeddings, cache_position
                )
                past_len += 1

                for idx, (tid, _) in exit_preds.items():
                    exit_token_lists[idx].append(tid)
                    exit_step_samples[idx].append((exit_elapsed[idx], avg_pw))
                base_token_list.append(base_token_id)
                base_step_samples.append((base_el, avg_pw))
                next_token_id = base_token_id

                if base_token_id == self.tokenizer.eos_token_id:
                    break

        torch.cuda.synchronize()
        e2e_end = time.perf_counter()

        n_tokens = len(base_token_list)

        def _agg(samples: List[Tuple[float, float]]) -> Dict:
            times = [t for t, _ in samples]
            energy_j = sum(t * pw for t, pw in samples)
            ttft = times[0] if times else 0.0
            avg_pt = sum(times[1:]) / max(len(times) - 1, 1) if len(times) > 1 else ttft
            e2e = sum(times)  # total compute attributed to this exit across all steps
            return {
                "ttft_sec": round(ttft, 6),
                "per_token_latency_sec": round(avg_pt, 6),
                "end_to_end_sec": round(e2e, 6),
                "total_energy_j": round(energy_j, 4),
                "tokens_per_joule": round(len(times) / energy_j if energy_j > 0 else 0.0, 2),
            }

        exits_out: Dict[str, Dict] = {}
        for idx, tokens in exit_token_lists.items():
            exits_out[f"exit_{idx}"] = {
                "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
                "tokens": tokens,
                **_agg(exit_step_samples[idx]),
            }
        exits_out["base"] = {
            "text": self.tokenizer.decode(base_token_list, skip_special_tokens=True),
            "tokens": base_token_list,
            **_agg(base_step_samples),
        }

        return {
            "exits": exits_out,
            "n_tokens": n_tokens,
            "end_to_end_sec": round(e2e_end - e2e_start, 6),
        }


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
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        self._num_layers = len(raw.model.layers) - 1 if hasattr(raw, "model") else 31
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        if hasattr(raw, "generation_config"):
            raw.generation_config.temperature = None
            raw.generation_config.top_p = None

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> Dict:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]

        torch.cuda.synchronize()
        power_start = _gpu_power_watts()
        e2e_start = time.perf_counter()

        out = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        torch.cuda.synchronize()
        e2e_end = time.perf_counter()
        power_end = _gpu_power_watts()

        generated_ids = out[0, prompt_len:].tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        n_tokens = len(generated_ids)
        e2e_sec = e2e_end - e2e_start

        ttft = e2e_sec / max(n_tokens, 1)
        avg_per_token = (e2e_sec - ttft) / max(n_tokens - 1, 1) if n_tokens > 1 else ttft

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
