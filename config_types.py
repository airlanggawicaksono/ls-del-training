from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B"
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    text_column: str = "text"
    output_dir: str = "outputs/llama"
    max_seq_length: int = 2048
    validation_split_percentage: int = 5
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    num_train_epochs: float = 3.0
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 250
    save_total_limit: int = 2
    seed: int = 42
    gradient_checkpointing: bool = True
    padding_side: Literal["left", "right"] = "right"
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    trust_remote_code: bool = False
    report_to: List[str] = Field(default_factory=lambda: ["none"])
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    overwrite_output_dir: bool = False

    @model_validator(mode="after")
    def check_dataset_source(self) -> "TrainConfig":
        if not self.dataset_name and not self.train_file:
            raise ValueError("Config must define dataset_name or train_file")
        return self

    @property
    def report_to_list(self) -> List[str]:
        cleaned = [x.strip() for x in self.report_to if x and x.strip()]
        if len(cleaned) == 1 and cleaned[0].lower() == "none":
            return []
        return cleaned
