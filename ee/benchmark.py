"""
Full benchmark suite: quality (perplexity/accuracy) + latency + energy.

Compares early-exit model vs baseline (original LLaMA) on CNN/DailyMail.
Both models are torch.compiled before benchmarking.
"""

import json
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hub import load_exit_heads, push_benchmark_results_to_hub
from .inference import BaselineGenerator, EarlyExitGenerator
from .model_wrapper import EarlyExitLlamaWrapper
from .utils import freeze_base_model


def load_cnn_dailymail(n_samples: int = 100, max_length: int = 512) -> List[str]:
    """Load CNN/DailyMail articles as prompts for benchmarking."""
    ds = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{n_samples}]")
    prompts = []
    for row in ds:
        # Use the first ~max_length chars of the article as prompt
        text = row["article"][:max_length]
        prompts.append(text)
    return prompts


def _quality_for_exit(
    wrapper: EarlyExitLlamaWrapper,
    exit_layer_idx: int,
    prompts: List[str],
    tokenizer,
    max_length: int = 512,
) -> Dict[str, float]:
    """Forward-pass quality metrics for one exit layer."""
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(wrapper.device)
        labels = input_ids.clone()

        outputs = wrapper(input_ids=input_ids)

        if exit_layer_idx in outputs.exit_logits:
            logits = outputs.exit_logits[exit_layer_idx]
        else:
            logits = outputs.base_logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        mask = shift_labels != -100
        preds = shift_logits.argmax(dim=-1)
        correct = (preds == shift_labels) & mask

        total_loss += loss.item()
        total_correct += correct.sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": round(avg_loss, 4),
        "perplexity": round(math.exp(min(avg_loss, 100)), 2),
        "accuracy": round(total_correct / max(total_tokens, 1), 4),
        "n_tokens": total_tokens,
    }


@torch.no_grad()
def benchmark_quality(
    wrapper: EarlyExitLlamaWrapper,
    prompts: List[str],
    tokenizer,
    max_length: int = 512,
) -> Dict[str, Dict[str, float]]:
    """Quality metrics for each exit + final layer."""
    results = {}

    for idx in wrapper.exit_layer_indices:
        print(f"  Quality: exit layer {idx}...")
        results[f"exit_{idx}"] = _quality_for_exit(wrapper, idx, prompts, tokenizer, max_length)

    num_layers = len(wrapper.base_model.model.layers)
    print(f"  Quality: base model (layer {num_layers})...")
    results["base_final"] = _quality_for_exit(wrapper, num_layers, prompts, tokenizer, max_length)

    return results


def benchmark_latency_energy(
    generator,
    prompts: List[str],
    max_new_tokens: int = 128,
    warmup: int = 3,
) -> Dict[str, float]:
    """
    Run generation on prompts, aggregate latency and energy.

    Runs a few warmup prompts first (important after torch.compile).
    """
    # Warmup (torch.compile compiles on first few runs)
    for i in range(min(warmup, len(prompts))):
        generator.generate(prompts[i], max_new_tokens=32)
    if hasattr(generator, "reset_statistics"):
        generator.reset_statistics()

    ttfts = []
    per_token_lats = []
    e2e_lats = []
    total_energy = 0.0
    total_tokens = 0

    for i, prompt in enumerate(prompts):
        result = generator.generate(prompt, max_new_tokens=max_new_tokens)
        ttfts.append(result["ttft_sec"])
        per_token_lats.append(result["per_token_latency_sec"])
        e2e_lats.append(result["end_to_end_sec"])
        total_energy += result["total_energy_j"]
        total_tokens += result["n_tokens"]

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(prompts)}] avg TTFT={sum(ttfts)/len(ttfts):.4f}s")

    tokens_per_joule = total_tokens / total_energy if total_energy > 0 else 0.0

    return {
        "ttft_sec_mean": round(sum(ttfts) / len(ttfts), 6),
        "ttft_sec_p50": round(sorted(ttfts)[len(ttfts) // 2], 6),
        "per_token_latency_sec_mean": round(sum(per_token_lats) / len(per_token_lats), 6),
        "end_to_end_sec_mean": round(sum(e2e_lats) / len(e2e_lats), 6),
        "total_tokens": total_tokens,
        "total_energy_j": round(total_energy, 4),
        "tokens_per_joule": round(tokens_per_joule, 2),
    }


def _compile_exit_heads(exit_heads: Dict[int, torch.nn.Module]) -> Dict[int, torch.nn.Module]:
    """Compile each exit head so EE latency is measured with compiled heads."""
    compiled: Dict[int, torch.nn.Module] = {}
    for idx, head in exit_heads.items():
        compiled[idx] = torch.compile(head)
    return compiled


def run_full_benchmark(
    base_model_name: str,
    exit_heads_repo_or_dir: str,
    exit_layer_indices: List[int],
    n_samples: int = 100,
    max_new_tokens: int = 128,
    confidence_threshold: float = 0.9,
    torch_dtype=torch.bfloat16,
    output_path: Optional[str] = None,
    push_results_to_hub_repo: Optional[str] = None,
    push_results_path_in_repo: Optional[str] = None,
) -> Dict:
    """
    Full benchmark: EE model vs baseline on CNN/DailyMail.

    Steps:
      1. Load CNN/DailyMail prompts
      2. Load base model + exit heads
      3. torch.compile both
      4. Run quality benchmark (perplexity/accuracy per exit)
      5. Run latency/energy benchmark for EE
      6. Run latency/energy benchmark for baseline
      7. Save JSON results
      8. Optional: push JSON results to HF Hub
    """
    print(f"=== Loading CNN/DailyMail ({n_samples} samples) ===")
    prompts = load_cnn_dailymail(n_samples)

    # ---- Load models ----
    print(f"\n=== Loading base model: {base_model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch_dtype, device_map="auto"
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    freeze_base_model(base_model)

    # ---- Load exit heads ----
    print(f"\n=== Loading exit heads from: {exit_heads_repo_or_dir} ===")
    # Try as HF repo first, then as local path
    if os.path.isdir(exit_heads_repo_or_dir):
        heads_dir = exit_heads_repo_or_dir
    else:
        from huggingface_hub import snapshot_download
        heads_dir = snapshot_download(exit_heads_repo_or_dir)

    head_device = "cuda" if torch.cuda.is_available() else "cpu"
    exit_heads, _ = load_exit_heads(heads_dir, device=head_device)

    wrapper = EarlyExitLlamaWrapper(
        base_model=base_model,
        exit_layer_indices=exit_layer_indices,
        hidden_size=base_model.config.hidden_size,
        vocab_size=base_model.config.vocab_size,
        norm_eps=base_model.config.rms_norm_eps,
        init_from_base=False,
    )
    for idx, head in exit_heads.items():
        wrapper.exit_heads[str(idx)].load_state_dict(head.state_dict())
    wrapper.register_hooks()
    wrapper.cuda()

    # ---- torch.compile ----
    print("\n=== Compiling models with torch.compile ===")
    print("Compiling full backbone (shared by baseline + EE) and EE heads/runtime...")
    compiled_base = torch.compile(base_model)
    compiled_exit_heads = _compile_exit_heads(exit_heads)
    for idx in exit_layer_indices:
        wrapper.exit_heads[str(idx)] = torch.compile(wrapper.exit_heads[str(idx)])
    # Compile base model layers used in EE forward
    wrapper.base_model = compiled_base

    results = {
        "config": {
            "base_model": base_model_name,
            "exit_layers": exit_layer_indices,
            "n_samples": n_samples,
            "max_new_tokens": max_new_tokens,
            "confidence_threshold": confidence_threshold,
            "torch_dtype": str(torch_dtype),
        }
    }

    # ---- Quality (forward-pass perplexity/accuracy) ----
    print("\n=== Quality Benchmark (perplexity / accuracy) ===")
    results["quality"] = benchmark_quality(wrapper, prompts, tokenizer)

    # Print quality table
    print("\n  Model       | Loss   | Perplexity | Accuracy")
    print("  ------------+--------+------------+---------")
    for key, r in results["quality"].items():
        print(f"  {key:11s} | {r['loss']:.4f} | {r['perplexity']:10.2f} | {r['accuracy']:.4f}")

    # ---- Latency + Energy: Early Exit ----
    print(f"\n=== Latency/Energy: Early Exit (threshold={confidence_threshold}) ===")
    ee_gen = EarlyExitGenerator(
        compiled_base,
        compiled_exit_heads,
        tokenizer,
        confidence_threshold,
        compile_runtime=True,
    )
    results["ee_latency"] = benchmark_latency_energy(ee_gen, prompts, max_new_tokens)
    ee_gen.print_exit_statistics()

    # ---- Latency + Energy: Baseline (full model) ----
    print("\n=== Latency/Energy: Baseline (full model) ===")
    baseline_gen = BaselineGenerator(compiled_base, tokenizer)
    results["baseline_latency"] = benchmark_latency_energy(baseline_gen, prompts, max_new_tokens)

    # ---- Comparison summary ----
    ee = results["ee_latency"]
    bl = results["baseline_latency"]
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30s} {'Baseline':>12s} {'Early Exit':>12s} {'Speedup':>10s}")
    print("-" * 64)

    for key, label in [
        ("ttft_sec_mean", "TTFT (sec)"),
        ("per_token_latency_sec_mean", "Per-token lat (sec)"),
        ("end_to_end_sec_mean", "End-to-end (sec)"),
        ("tokens_per_joule", "Tokens/Joule"),
    ]:
        bv = bl[key]
        ev = ee[key]
        if key == "tokens_per_joule":
            speedup = f"{ev/bv:.2f}x" if bv > 0 else "N/A"
        else:
            speedup = f"{bv/ev:.2f}x" if ev > 0 else "N/A"
        print(f"  {label:<28s} {bv:>12.4f} {ev:>12.4f} {speedup:>10s}")

    print("=" * 60)

    # ---- Save ----
    if output_path is None:
        output_path = "benchmark_results.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    if push_results_to_hub_repo:
        results_url = push_benchmark_results_to_hub(
            results_json_path=output_path,
            repo_id=push_results_to_hub_repo,
            path_in_repo=push_results_path_in_repo,
        )
        print(f"Results pushed to Hub: {results_url}")

    wrapper.remove_hooks()
    return results
