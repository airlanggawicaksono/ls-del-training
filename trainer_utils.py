from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config_types import TrainConfig


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]

    low = value.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"none", "null"}:
        return None

    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_value(raw: str) -> Any:
    value = raw.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in inner.split(",")]
    return _parse_scalar(value)


def _parse_config_file(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    parsed: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith(";"):
                continue
            if "=" not in stripped:
                raise ValueError(f"Invalid config line {line_no}: expected key = value")
            key, raw_value = stripped.split("=", 1)
            parsed[key.strip()] = _parse_value(raw_value)
    return parsed


def load_train_config(config_path: str) -> TrainConfig:
    raw_config = _parse_config_file(config_path)
    return TrainConfig.model_validate(raw_config)


def _dataset_extension(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".txt"):
        return "text"
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".json") or lower.endswith(".jsonl"):
        return "json"
    raise ValueError("train/validation files must be .txt, .csv, .json, or .jsonl")


def _load_raw_dataset(cfg: TrainConfig) -> DatasetDict:
    if cfg.dataset_name:
        dataset = load_dataset(cfg.dataset_name, cfg.dataset_config_name)
    else:
        data_files: Dict[str, str] = {"train": cfg.train_file}
        if cfg.validation_file:
            data_files["validation"] = cfg.validation_file
        dataset = load_dataset(_dataset_extension(cfg.train_file), data_files=data_files)

    if "validation" not in dataset:
        split = dataset["train"].train_test_split(
            test_size=cfg.validation_split_percentage / 100.0,
            seed=cfg.seed,
        )
        return DatasetDict({"train": split["train"], "validation": split["test"]})
    return dataset


def _resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def run_training(cfg: TrainConfig, resume_from_checkpoint: Optional[str] = None) -> None:
    set_seed(cfg.seed)
    raw_dataset = _load_raw_dataset(cfg)

    text_column = cfg.text_column if cfg.text_column in raw_dataset["train"].column_names else raw_dataset["train"].column_names[0]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        use_fast=True,
        trust_remote_code=cfg.trust_remote_code,
    )
    tokenizer.padding_side = cfg.padding_side

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=_resolve_dtype(cfg.torch_dtype),
        trust_remote_code=cfg.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    if cfg.gradient_checkpointing:
        model.config.use_cache = False

    model_max_length = tokenizer.model_max_length
    if model_max_length is None or model_max_length > 100_000:
        model_max_length = cfg.max_seq_length
    block_size = min(cfg.max_seq_length, model_max_length)

    tokenized = raw_dataset.map(
        lambda batch: tokenizer(batch[text_column]),
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
        desc="Tokenizing",
    )

    def group_texts(examples):
        merged = {k: sum(examples[k], []) for k in examples}
        total = (len(merged["input_ids"]) // block_size) * block_size
        result = {k: [v[i : i + block_size] for i in range(0, total, block_size)] for k, v in merged.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(group_texts, batched=True, desc="Packing")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=cfg.overwrite_output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=cfg.report_to_list,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    trainer.save_state()

    if cfg.push_to_hub:
        trainer.push_to_hub()
