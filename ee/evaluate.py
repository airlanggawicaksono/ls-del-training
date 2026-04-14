import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from .model_wrapper import EarlyExitLlamaWrapper


@torch.no_grad()
def evaluate_single_exit(
    wrapper: EarlyExitLlamaWrapper,
    exit_layer_idx: int,
    dataloader: DataLoader,
) -> Dict[str, float]:
    """
    Evaluate one exit head.

    Returns:
        loss:       average cross-entropy
        perplexity: exp(loss)
        accuracy:   top-1 next-token accuracy
    """
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(wrapper.device)
        labels = batch["labels"].to(wrapper.device)

        outputs = wrapper(input_ids=input_ids)

        if exit_layer_idx in outputs.exit_logits:
            logits = outputs.exit_logits[exit_layer_idx]
        else:
            # Final layer
            logits = outputs.base_logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        # Accuracy
        mask = shift_labels != -100
        preds = shift_logits.argmax(dim=-1)
        correct = (preds == shift_labels) & mask

        n_tokens = mask.sum().item()
        total_loss += loss.item()
        total_correct += correct.sum().item()
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": round(avg_loss, 4),
        "perplexity": round(math.exp(min(avg_loss, 100)), 2),
        "accuracy": round(total_correct / max(total_tokens, 1), 4),
    }


@torch.no_grad()
def evaluate_all_exits(
    wrapper: EarlyExitLlamaWrapper,
    eval_dataset,
    tokenizer,
    batch_size: int = 4,
) -> Dict[int, Dict[str, float]]:
    """
    Run evaluate_single_exit for each exit + the base final layer.

    Returns:
        {
            8:  {"loss": ..., "perplexity": ..., "accuracy": ...},
            16: {"loss": ..., "perplexity": ..., "accuracy": ...},
            24: {"loss": ..., "perplexity": ..., "accuracy": ...},
            32: {"loss": ..., "perplexity": ..., "accuracy": ...},  # base
        }
    """
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collator)

    results: Dict[int, Dict[str, float]] = {}

    # Evaluate each exit head
    for idx in wrapper.exit_layer_indices:
        print(f"  Evaluating exit at layer {idx}...")
        results[idx] = evaluate_single_exit(wrapper, idx, dataloader)

    # Evaluate base model (final layer)
    num_layers = len(wrapper.base_model.model.layers)
    print(f"  Evaluating base model (layer {num_layers})...")
    results[num_layers] = evaluate_single_exit(wrapper, num_layers, dataloader)

    # Print comparison table
    print("\n  Layer | Loss   | Perplexity | Accuracy")
    print("  ------+--------+------------+---------")
    for layer_idx in sorted(results):
        r = results[layer_idx]
        tag = "*" if layer_idx in wrapper.exit_layer_indices else " "
        print(
            f"  {tag}{layer_idx:4d}  | {r['loss']:.4f} | {r['perplexity']:10.2f} | {r['accuracy']:.4f}"
        )

    return results
