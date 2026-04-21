import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import torch
from datasets import DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    set_seed,
)

from config_types_ee import EETrainConfig
from trainer_utils import (
    _build_training_arguments,
    _load_raw_dataset,
    _resolve_dtype,
    _trainer_tokenizer_kwargs,
)

from .callbacks import TrainingMetricsCallback, TorchProfilerCallback
from .hub import push_exit_heads_to_hub, push_training_logs_to_hub, save_exit_heads
from .model_wrapper import EarlyExitLlamaWrapper
from .trainer import EarlyExitTrainer
from .utils import count_parameters, freeze_base_model


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def _prepare_dataset(
    raw_dataset: DatasetDict,
    tokenizer,
    cfg: EETrainConfig,
) -> DatasetDict:
    """Tokenize and pack dataset into fixed-length blocks (same as base trainer)."""
    text_col = (
        cfg.text_column
        if cfg.text_column in raw_dataset["train"].column_names
        else raw_dataset["train"].column_names[0]
    )

    model_max = tokenizer.model_max_length
    if model_max is None or model_max > 100_000:
        model_max = cfg.max_seq_length
    block_size = min(cfg.max_seq_length, model_max)

    tokenized = raw_dataset.map(
        lambda batch: tokenizer(batch[text_col]),
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
        desc="Tokenizing",
    )

    def group_texts(examples):
        merged = {k: sum(examples[k], []) for k in examples}
        total = (len(merged["input_ids"]) // block_size) * block_size
        result = {
            k: [v[i : i + block_size] for i in range(0, total, block_size)]
            for k, v in merged.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized.map(group_texts, batched=True, desc="Packing")


def run_ee_training(
    cfg: EETrainConfig,
    resume_from_checkpoint: Optional[str] = None,
) -> None:
    """
    Full EE-Tuning pipeline:
      1. Load base model + tokenizer
      2. Freeze base model
      3. Wrap with exit heads
      4. Train (only exit heads get gradients)
      5. Save / upload exit heads
    """
    set_seed(cfg.seed)

    # ---- Dataset ----
    raw_dataset = _load_raw_dataset(cfg)

    print(f"[EE] Train samples: {len(raw_dataset['train']):,}")
    if "validation" in raw_dataset:
        print(f"[EE] Validation samples: {len(raw_dataset['validation']):,}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        use_fast=True,
        trust_remote_code=cfg.trust_remote_code,
    )
    tokenizer.padding_side = cfg.padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm_dataset = _prepare_dataset(raw_dataset, tokenizer, cfg)

    # Print token count estimate
    n_train_seqs = len(lm_dataset["train"])
    est_tokens = n_train_seqs * cfg.max_seq_length
    print(f"[EE] Packed sequences: {n_train_seqs:,} x {cfg.max_seq_length} = ~{est_tokens/1e6:.0f}M tokens")

    # ---- Model ----
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=_resolve_dtype(cfg.torch_dtype),
        trust_remote_code=cfg.trust_remote_code,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    freeze_base_model(base_model)

    wrapper = EarlyExitLlamaWrapper(
        base_model=base_model,
        exit_layer_indices=cfg.exit_layer_indices,
        hidden_size=base_model.config.hidden_size,
        vocab_size=base_model.config.vocab_size,
        norm_eps=base_model.config.rms_norm_eps,
        init_from_base=cfg.init_exit_from_base,
    )
    wrapper.register_hooks()

    # Print param summary
    params = count_parameters(wrapper)
    print(f"\n[EE] Total params:     {params['total']:,}")
    print(f"[EE] Trainable params: {params['trainable']:,}")
    print(f"[EE] Frozen params:    {params['frozen']:,}")
    print(f"[EE] Exit layers:      {cfg.exit_layer_indices}")
    print(f"[EE] Exit weights:     {cfg.exit_loss_weights}\n")

    # ---- Training ----
    exit_weights: Dict[int, float] = dict(
        zip(cfg.exit_layer_indices, cfg.exit_loss_weights)
    )

    training_args = _build_training_arguments(
        {
            "output_dir": cfg.output_dir,
            "overwrite_output_dir": cfg.overwrite_output_dir,
            "per_device_train_batch_size": cfg.per_device_train_batch_size,
            "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "num_train_epochs": cfg.num_train_epochs,
            "logging_steps": cfg.logging_steps,
            "save_steps": cfg.save_steps,
            "eval_steps": cfg.eval_steps,
            "save_total_limit": cfg.save_total_limit,
            "gradient_checkpointing": False,  # no benefit when backbone frozen
            "report_to": cfg.report_to_list,
            "evaluation_strategy": "steps",
            "load_best_model_at_end": False,  # we track per-exit, not single metric
            "push_to_hub": False,  # we handle hub upload separately
        }
    )

    trainer_kwargs: Dict[str, object] = {
        "exit_weights": exit_weights,
        "model": wrapper,
        "args": training_args,
        "train_dataset": lm_dataset["train"],
        "eval_dataset": lm_dataset["validation"],
        "data_collator": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    }
    metrics_cb = TrainingMetricsCallback(seq_length=cfg.max_seq_length)
    callbacks = [metrics_cb]
    if getattr(cfg, "profile_steps", 0) > 0:
        callbacks.append(TorchProfilerCallback(
            output_dir=cfg.output_dir,
            warmup_steps=2,
            active_steps=cfg.profile_steps,
        ))
    trainer_kwargs["callbacks"] = callbacks
    trainer_kwargs.update(_trainer_tokenizer_kwargs(tokenizer))
    trainer = EarlyExitTrainer(**trainer_kwargs)

    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_eval_metrics = trainer.evaluate()

    # ---- Save logs/artifacts ----
    logs_dir = os.path.join(cfg.output_dir, "logs", "train")
    os.makedirs(logs_dir, exist_ok=True)
    run_meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume_from_checkpoint": resume_from_checkpoint,
        "config": cfg.model_dump(),
    }
    with open(os.path.join(logs_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, default=_json_default)
    with open(os.path.join(logs_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(train_output.metrics, f, indent=2, default=_json_default)
    with open(os.path.join(logs_dir, "eval_metrics_final.json"), "w", encoding="utf-8") as f:
        json.dump(final_eval_metrics, f, indent=2, default=_json_default)
    with open(os.path.join(logs_dir, "log_history.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=_json_default)
    with open(os.path.join(logs_dir, "epoch_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_cb.epoch_metrics, f, indent=2, default=_json_default)
    print(f"[EE] Training logs saved to: {logs_dir}")

    # ---- Save ----
    heads_dir = save_exit_heads(wrapper, cfg.output_dir)
    print(f"\n[EE] Exit heads saved to: {heads_dir}")

    if cfg.hub_exit_heads_repo:
        url = push_exit_heads_to_hub(heads_dir, cfg.hub_exit_heads_repo)
        print(f"[EE] Pushed to Hub: {url}")
        logs_url = push_training_logs_to_hub(logs_dir, cfg.hub_exit_heads_repo)
        print(f"[EE] Training logs pushed to Hub: {logs_url}")

    wrapper.remove_hooks()
