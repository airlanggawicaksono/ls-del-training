"""
Full benchmark suite: quality (perplexity/accuracy/ROUGE) + latency + energy.

Compares early-exit model vs baseline (original LLaMA) on CNN/DailyMail.
Both models are torch.compiled before benchmarking.
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.llama.modeling_llama as _llama_mod

_orig_apply_rotary = _llama_mod.apply_rotary_pos_emb
TORCHDYNAMO_VERBOSE = 1


def _apply_rotary_pos_emb_patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    import traceback
    print(f"[RoPE] q={q.shape} k={k.shape} cos={cos.shape} sin={sin.shape}")
    traceback.print_stack(limit=6)
    cos = cos[..., :q.shape[-1]]
    sin = sin[..., :q.shape[-1]]
    return _orig_apply_rotary(q, k, cos, sin)


_llama_mod.apply_rotary_pos_emb = _apply_rotary_pos_emb_patched

from .hub import load_exit_heads, push_benchmark_results_to_hub
from .inference import BaselineGenerator, EarlyExitGenerator
from .model_wrapper import EarlyExitLlamaWrapper
from .utils import freeze_base_model


def load_cnn_dailymail(
    n_samples: int = 100,
    max_prompt_length: int = 512,
) -> List[Dict[str, str]]:
    """Load CNN/DailyMail articles + reference summaries for benchmarking."""
    ds = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{n_samples}]")
    samples = []
    for row in ds:
        samples.append(
            {
                "prompt": row["article"][:max_prompt_length],
                "reference": row["highlights"],
            }
        )
    return samples


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-2 and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge2", "rougeL"], use_stemmer=True)
    r2_scores = []
    rl_scores = []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r2_scores.append(scores["rouge2"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)
    return {
        "rouge2_f1": round(sum(r2_scores) / len(r2_scores), 4),
        "rougeL_f1": round(sum(rl_scores) / len(rl_scores), 4),
    }


@torch.no_grad()
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
    samples: List[Dict[str, str]],
    tokenizer,
    max_length: int = 512,
) -> Dict[str, Dict[str, float]]:
    """Quality metrics (perplexity/accuracy) for each exit + final layer."""
    prompts = [s["prompt"] for s in samples]
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
    samples: List[Dict[str, str]],
    max_new_tokens: int = 128,
    warmup: int = 3,
) -> Dict[str, float]:
    """
    Run generation on samples, aggregate latency, energy, and ROUGE.

    Runs a few warmup prompts first (important after torch.compile).
    """
    prompts = [s["prompt"] for s in samples]
    references = [s["reference"] for s in samples]

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
    predictions = []

    for i, prompt in enumerate(prompts):
        result = generator.generate(prompt, max_new_tokens=max_new_tokens)
        ttfts.append(result["ttft_sec"])
        per_token_lats.append(result["per_token_latency_sec"])
        e2e_lats.append(result["end_to_end_sec"])
        total_energy += result["total_energy_j"]
        total_tokens += result["n_tokens"]
        predictions.append(result["text"])

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(prompts)}] avg TTFT={sum(ttfts)/len(ttfts):.4f}s")

    tokens_per_joule = total_tokens / total_energy if total_energy > 0 else 0.0

    # ROUGE scores against reference summaries
    rouge = compute_rouge(predictions, references)

    return {
        "ttft_sec_mean": round(sum(ttfts) / len(ttfts), 6),
        "ttft_sec_p50": round(sorted(ttfts)[len(ttfts) // 2], 6),
        "per_token_latency_sec_mean": round(sum(per_token_lats) / len(per_token_lats), 6),
        "end_to_end_sec_mean": round(sum(e2e_lats) / len(e2e_lats), 6),
        "total_tokens": total_tokens,
        "total_energy_j": round(total_energy, 4),
        "tokens_per_joule": round(tokens_per_joule, 2),
        **rouge,
    }


def _compile_exit_heads(exit_heads: Dict[int, torch.nn.Module]) -> Dict[int, torch.nn.Module]:
    """Compile each exit head so EE latency is measured with compiled heads.

    dynamic=True because each token step has a different seq_len (no KV
    cache). Without this, torch.compile recompiles for every shape and
    benchmark takes forever.
    """
    compiled: Dict[int, torch.nn.Module] = {}
    for idx, head in exit_heads.items():
        compiled[idx] = torch.compile(head, dynamic=True)
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
    samples = load_cnn_dailymail(n_samples)

    # ---- Load models ----
    print(f"\n=== Loading base model: {base_model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch_dtype, device_map="auto")
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
    # Both EE and baseline get fully compiled.
    #   - EE: compile_runtime=True compiles the entire step fn (embed → layers → exit check)
    #   - Baseline: torch.compile(model) compiles the whole model for .generate()
    #   - Quality benchmark runs first BEFORE compile (just perplexity, no speed measurement)
    print("\n=== Quality Benchmark (perplexity / accuracy) ===")
    print("  (runs before compile — quality doesn't need speed optimization)")
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
    results["quality"] = benchmark_quality(wrapper, samples, tokenizer)

    print("\n  Model       | Loss   | Perplexity | Accuracy")
    print("  ------------+--------+------------+---------")
    for key, r in results["quality"].items():
        print(f"  {key:11s} | {r['loss']:.4f} | {r['perplexity']:10.2f} | {r['accuracy']:.4f}")

    # Done with wrapper/hooks for quality — clean up before compile
    wrapper.remove_hooks()

    # ---- Compile per-layer (NOT the whole model) ----
    # We compile each decoder layer and each exit head with dynamic=True.
    # Both EE and baseline benefit: baseline's generate() calls layers which
    # are compiled; EE's manual loop calls the same compiled layers + compiled
    # exit heads. Fair comparison — same components compiled on both paths.
    #
    # Why NOT torch.compile(base_model)?
    #   - Nested with per-layer compile causes tracing conflicts
    #   - Per-layer compile already covers all heavy matmuls
    # Why dynamic=True?
    #   - Each token step has a different seq_len (no KV cache). Without
    #     dynamic shapes, torch.compile recompiles on every shape (1..128)
    #     and the benchmark takes forever.
    print("\n=== Compiling decoder layers + exit heads (dynamic shapes) ===")
    for layer in base_model.model.layers:
        layer.self_attn = torch.compile(layer.self_attn, dynamic=True)
        layer.mlp = torch.compile(layer.mlp, dynamic=True)
    compiled_exit_heads = _compile_exit_heads(exit_heads)
    print(f"  Compiled {len(base_model.model.layers)} decoder layers + {len(compiled_exit_heads)} exit heads")

    # ---- Latency + Energy + ROUGE: Early Exit ----
    print(f"\n=== Latency/Energy/ROUGE: Early Exit (threshold={confidence_threshold}) ===")
    ee_gen = EarlyExitGenerator(
        base_model,
        compiled_exit_heads,
        tokenizer,
        confidence_threshold,
        compile_runtime=False,  # step fn is eager; its heavy components are compiled
    )
    results["ee_latency"] = benchmark_latency_energy(ee_gen, samples, max_new_tokens)
    ee_gen.print_exit_statistics()

    # ---- Latency + Energy + ROUGE: Baseline ----
    print("\n=== Latency/Energy/ROUGE: Baseline (full model) ===")
    baseline_gen = BaselineGenerator(base_model, tokenizer)
    results["baseline_latency"] = benchmark_latency_energy(baseline_gen, samples, max_new_tokens)

    # ---- Comparison summary ----
    ee = results["ee_latency"]
    bl = results["baseline_latency"]
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30s} {'Baseline':>12s} {'Early Exit':>12s} {'Speedup':>10s}")
    print("-" * 64)

    for key, label, higher_is_better in [
        ("ttft_sec_mean", "TTFT (sec)", False),
        ("per_token_latency_sec_mean", "Per-token lat (sec)", False),
        ("end_to_end_sec_mean", "End-to-end (sec)", False),
        ("tokens_per_joule", "Tokens/Joule", True),
        ("rouge2_f1", "ROUGE-2 F1", True),
        ("rougeL_f1", "ROUGE-L F1", True),
    ]:
        bv = bl[key]
        ev = ee[key]
        if higher_is_better:
            ratio = f"{ev/bv:.2f}x" if bv > 0 else "N/A"
        else:
            ratio = f"{bv/ev:.2f}x" if ev > 0 else "N/A"
        print(f"  {label:<28s} {bv:>12.4f} {ev:>12.4f} {ratio:>10s}")

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

    return results
