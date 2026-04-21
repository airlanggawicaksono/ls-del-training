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

from .hub import load_exit_heads
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
    gpu_utils = []
    gpu_mem_utils = []
    vram_gbs = []
    power_ws = []
    cpu_pcts = []
    ram_gbs = []
    per_sample_exit_layers = []
    predictions = []
    generations = []

    for i, prompt in enumerate(prompts):
        result = generator.generate(prompt, max_new_tokens=max_new_tokens)
        ttfts.append(result["ttft_sec"])
        per_token_lats.append(result["per_token_latency_sec"])
        e2e_lats.append(result["end_to_end_sec"])
        total_energy += result["total_energy_j"]
        total_tokens += result["n_tokens"]
        if result.get("avg_gpu_util_pct", 0) > 0:
            gpu_utils.append(result["avg_gpu_util_pct"])
        if result.get("avg_gpu_mem_util_pct", 0) > 0:
            gpu_mem_utils.append(result["avg_gpu_mem_util_pct"])
        if result.get("avg_vram_gb", 0) > 0:
            vram_gbs.append(result["avg_vram_gb"])
        if result.get("avg_power_w", 0) > 0:
            power_ws.append(result["avg_power_w"])
        if result.get("avg_cpu_pct", 0) > 0:
            cpu_pcts.append(result["avg_cpu_pct"])
        if result.get("avg_ram_gb", 0) > 0:
            ram_gbs.append(result["avg_ram_gb"])
        if "exit_layers" in result:
            per_sample_exit_layers.append(result["exit_layers"])
        predictions.append(result["text"])
        generations.append({
            "sample_idx": i,
            "prompt": prompt,
            "reference": references[i],
            "generated": result["text"],
            "tokens": result.get("tokens", []),
            "exit_layers": result.get("exit_layers", []),
            "confidences": result.get("confidences", []),
            "exit_stats": result.get("exit_stats", {}),
            "n_tokens": result["n_tokens"],
            "ttft_sec": result["ttft_sec"],
            "per_token_latency_sec": result["per_token_latency_sec"],
            "end_to_end_sec": result["end_to_end_sec"],
            "total_energy_j": result["total_energy_j"],
            "joules_per_token": result.get("joules_per_token", 0.0),
            "tokens_per_joule": result.get("tokens_per_joule", 0.0),
            "avg_gpu_util_pct": result.get("avg_gpu_util_pct", 0.0),
            "avg_gpu_mem_util_pct": result.get("avg_gpu_mem_util_pct", 0.0),
            "avg_vram_gb": result.get("avg_vram_gb", 0.0),
            "avg_power_w": result.get("avg_power_w", 0.0),
            "avg_cpu_pct": result.get("avg_cpu_pct", 0.0),
            "avg_ram_gb": result.get("avg_ram_gb", 0.0),
            "avg_gpu_sm_clock_mhz": result.get("avg_gpu_sm_clock_mhz", 0.0),
            "avg_gpu_mem_clock_mhz": result.get("avg_gpu_mem_clock_mhz", 0.0),
        })

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(prompts)}] avg TTFT={sum(ttfts)/len(ttfts):.4f}s")

    tokens_per_joule = total_tokens / total_energy if total_energy > 0 else 0.0
    joules_per_token = total_energy / total_tokens if total_tokens > 0 else 0.0
    rouge = compute_rouge(predictions, references)

    def _mean(lst): return round(sum(lst) / len(lst), 2) if lst else 0.0

    # Token-level exit layer distribution across all samples
    all_exit_layers = [l for sample in per_sample_exit_layers for l in sample]
    exit_layer_dist: Dict[str, int] = {}
    for l in all_exit_layers:
        exit_layer_dist[str(l)] = exit_layer_dist.get(str(l), 0) + 1

    stats = {
        "ttft_sec_mean": round(sum(ttfts) / len(ttfts), 6),
        "ttft_sec_p50": round(sorted(ttfts)[len(ttfts) // 2], 6),
        "per_token_latency_sec_mean": round(sum(per_token_lats) / len(per_token_lats), 6),
        "end_to_end_sec_mean": round(sum(e2e_lats) / len(e2e_lats), 6),
        "total_tokens": total_tokens,
        "total_energy_j": round(total_energy, 4),
        "tokens_per_joule": round(tokens_per_joule, 2),
        "joules_per_token": round(joules_per_token, 6),
        "avg_gpu_util_pct": _mean(gpu_utils),
        "avg_gpu_mem_util_pct": _mean(gpu_mem_utils),
        "avg_vram_gb": _mean(vram_gbs),
        "avg_power_w": _mean(power_ws),
        "avg_cpu_pct": _mean(cpu_pcts),
        "avg_ram_gb": _mean(ram_gbs),
        "exit_layer_distribution": exit_layer_dist,
        **rouge,
    }
    return stats, generations


def benchmark_per_exit(
    base_model,
    exit_heads: Dict[int, torch.nn.Module],
    tokenizer,
    samples: List[Dict[str, str]],
    exit_layer_indices: List[int],
    max_new_tokens: int = 128,
    warmup: int = 3,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """
    Separate truncated run per exit layer using force_exit_layer.

    Each exit layer gets its own EarlyExitGenerator that stops decode at that
    layer — KV cache entries are only built up to the forced exit layer during
    decode, giving a real isolated hardware profile (power, VRAM, clocks) per exit.
    Prefill still runs all 32 layers once to populate the KV cache.

    Returns ({exit_8: stats, exit_16: stats, ...}, flat per-sample generations list)
    """
    all_stats: Dict[str, Dict] = {}
    all_generations: List[Dict] = []

    for idx in sorted(exit_layer_indices):
        key = f"exit_{idx}"
        print(f"\n  --- Per-exit benchmark: {key} (force_exit_layer={idx}) ---")
        gen = EarlyExitGenerator(
            base_model,
            exit_heads,
            tokenizer,
            confidence_threshold=1.1,
            use_kv_cache=True,
            force_exit_layer=idx,
        )
        stats, gens = benchmark_latency_energy(gen, samples, max_new_tokens, warmup)
        all_stats[key] = stats
        for g in gens:
            g["forced_exit_layer"] = idx
            g["exit_key"] = key
        all_generations.extend(gens)

    # Base: no force exit, threshold=1.1 → always runs all 32 layers
    print(f"\n  --- Per-exit benchmark: base (all 32 layers) ---")
    base_gen = EarlyExitGenerator(
        base_model,
        exit_heads,
        tokenizer,
        confidence_threshold=1.1,
        use_kv_cache=True,
        force_exit_layer=None,
    )
    base_stats, base_gens = benchmark_latency_energy(base_gen, samples, max_new_tokens, warmup)
    all_stats["base"] = base_stats
    for g in base_gens:
        g["forced_exit_layer"] = None
        g["exit_key"] = "base"
    all_generations.extend(base_gens)

    return all_stats, all_generations


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
    torch.set_float32_matmul_precision("high")

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

    # Newer transformers strips the batch dim before LlamaAttention under torch.compile
    # (hidden_states enters as 2D → input_shape is 1-tuple → wrong q format).
    # Patch at class level so Dynamo traces the corrected path.
    import transformers.models.llama.modeling_llama as _llama_mod
    if not getattr(_llama_mod.LlamaAttention.forward, "_batch_patched", False):
        _orig_attn_fwd = _llama_mod.LlamaAttention.forward
        def _attn_fwd_patched(self, hidden_states, *args, **kwargs):
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
            return _orig_attn_fwd(self, hidden_states, *args, **kwargs)
        _attn_fwd_patched._batch_patched = True
        _llama_mod.LlamaAttention.forward = _attn_fwd_patched

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
    # MLP layers + exit heads compiled; self_attn left eager.
    # compile_runtime=True causes Dynamo to trace through LlamaAttention where newer
    # transformers produces q in (seq, head_dim, num_heads) layout under fake-tensor
    # propagation, breaking the mul with cos. MLP is the actual compute bottleneck;
    # attention runs fast via SDPA regardless.
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

    # ---- Compile MLP per-layer + exit heads ----
    print("\n=== Compiling decoder layers + exit heads (dynamic shapes) ===")
    for layer in base_model.model.layers:
        layer.mlp = torch.compile(layer.mlp, dynamic=True)
    compiled_exit_heads = _compile_exit_heads(exit_heads)
    print(f"  Compiled {len(base_model.model.layers)} MLP layers + {len(compiled_exit_heads)} exit heads")

    # ---- Per-exit: separate truncated run per exit layer.
    # Each exit layer runs its own EarlyExitGenerator(force_exit_layer=idx).
    # Decode stops at the forced layer — real isolated hardware profile per exit.
    print("\n=== Per-Exit Benchmark (separate truncated run per exit layer) ===")
    results["per_exit"], results["per_exit_generations"] = benchmark_per_exit(
        base_model, compiled_exit_heads, tokenizer, samples, exit_layer_indices, max_new_tokens
    )
    _pe_cols = [
        ("ttft_sec_mean",              "TTFT(s)",     8,  ".4f"),
        ("per_token_latency_sec_mean", "Per-tok(s)", 10,  ".6f"),
        ("end_to_end_sec_mean",        "E2E(s)",      8,  ".4f"),
        ("total_energy_j",             "Energy(J)",  10,  ".2f"),
        ("joules_per_token",           "J/tok",       7,  ".4f"),
        ("avg_power_w",                "Power(W)",    8,  ".1f"),
        ("avg_vram_gb",                "VRAM(GB)",    8,  ".3f"),
        ("avg_ram_gb",                 "RAM(GB)",     7,  ".3f"),
        ("avg_gpu_util_pct",           "GPU%",        6,  ".1f"),
        ("avg_gpu_sm_clock_mhz",       "SM(MHz)",     7,  ".0f"),
        ("rouge2_f1",                  "R2-F1",       7,  ".4f"),
        ("rougeL_f1",                  "RL-F1",       7,  ".4f"),
    ]
    hdr = "  " + f"{'Exit':<10s}" + "".join(f" {c[1]:>{c[2]}s}" for c in _pe_cols)
    print("\n" + hdr)
    print("  " + "-" * (10 + sum(c[2] + 1 for c in _pe_cols)))
    for key, r in results["per_exit"].items():
        row_s = f"  {key:<10s}" + "".join(f" {r.get(c[0], 0):{c[2]}{c[3]}}" for c in _pe_cols)
        print(row_s)

    # ---- Dynamic early exit: per-token confidence check at each exit layer.
    # Prefill runs full 32-layer forward; decode uses KV cache.
    # Each decode step exits at the first layer where max softmax prob >= confidence_threshold.
    # Tokens that never reach threshold fall through to L31 (base model output).
    print(f"\n=== Dynamic Early Exit (confidence threshold={confidence_threshold}, KV cache) ===")
    ee_gen = EarlyExitGenerator(
        base_model,
        compiled_exit_heads,
        tokenizer,
        confidence_threshold,
        use_kv_cache=True,
    )
    results["ee_latency"], results["ee_generations"] = benchmark_latency_energy(ee_gen, samples, max_new_tokens)
    ee_gen.print_exit_statistics()

    # ---- Latency + Energy + ROUGE: Baseline ----
    print("\n=== Latency/Energy/ROUGE: Baseline (full model) ===")
    baseline_gen = BaselineGenerator(base_model, tokenizer)
    results["baseline_latency"], results["baseline_generations"] = benchmark_latency_energy(baseline_gen, samples, max_new_tokens)

    # ---- Comparison summary ----
    bl = results["baseline_latency"]
    print("\n" + "=" * 72)
    print("COMPARISON SUMMARY")
    print("  per_exit_*  : fixed exit layer, shared KV, all 32 layers always run")
    print("  dynamic_ee  : confidence-based exit (thr={:.2f}), KV cache, exits early".format(confidence_threshold))
    print("  baseline    : standard model.generate(), no exit heads")
    print("=" * 72)

    col_keys = [
        ("ttft_sec_mean",             "TTFT (s)",          False),
        ("per_token_latency_sec_mean","Per-tok lat (s)",   False),
        ("end_to_end_sec_mean",       "E2E (s)",           False),
        ("tokens_per_joule",          "Tok/J",             True),
        ("joules_per_token",          "J/tok",             False),
        ("avg_power_w",               "Power (W)",         False),
        ("avg_vram_gb",               "VRAM (GB)",         False),
        ("avg_ram_gb",                "RAM (GB)",          False),
        ("avg_gpu_util_pct",          "GPU util%",         True),
        ("avg_gpu_sm_clock_mhz",      "SM clk(MHz)",       True),
        ("rouge2_f1",                 "R-2 F1",            True),
        ("rougeL_f1",                 "R-L F1",            True),
    ]

    header = f"  {'Layer':<12s}" + "".join(f" {lbl:>13s}" for _, lbl, _ in col_keys)
    print(header)
    print("  " + "-" * (12 + 13 * len(col_keys)))

    def _row(label, row):
        parts = [f"  {label:<12s}"]
        for key, _, higher in col_keys:
            bv = bl.get(key, 0)
            rv = row.get(key, 0)
            ratio = ""
            if bv > 0:
                ratio = f"{rv/bv:.2f}x" if higher else f"{bv/rv:.2f}x"
            parts.append(f" {rv:>7.3f}{ratio:>6s}")
        print("".join(parts))

    _row("baseline", bl)
    for key in results["per_exit"]:
        _row(key, results["per_exit"][key])
    _row("dynamic_ee", results["ee_latency"])

    print("=" * 72)

    # ---- Push to Hub ----
    if not push_results_to_hub_repo:
        raise ValueError("push_results_to_hub_repo is required — results are pushed to HF Hub, not saved to disk.")

    import tempfile
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(push_results_to_hub_repo, repo_type="dataset", exist_ok=True)

    hub_dir = push_results_path_in_repo or "benchmark"

    files_to_push = {
        "benchmark_results.json": {k: v for k, v in results.items() if "generations" not in k},
        "samples_per_exit.json":   results.get("per_exit_generations", []),
        "samples_dynamic_ee.json": results.get("ee_generations", []),
        "samples_baseline.json":   results.get("baseline_generations", []),
    }

    for fname, data in files_to_push.items():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp_path = tmp.name
        url = api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=f"{hub_dir}/{fname}",
            repo_id=push_results_to_hub_repo,
            repo_type="dataset",
            commit_message=f"Upload {fname}",
        )
        os.unlink(tmp_path)
        print(f"Pushed {fname} → {url}")

    return results
