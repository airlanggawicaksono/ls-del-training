import json
import os
from typing import Dict, Tuple

import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file

from .exit_head import ExitHead
from .model_wrapper import EarlyExitLlamaWrapper


def save_exit_heads(wrapper: EarlyExitLlamaWrapper, output_dir: str) -> str:
    """
    Save exit head weights in safetensors format.

    Creates:
        output_dir/
            exit_heads/
                exit_head_8.safetensors
                exit_head_16.safetensors
                exit_head_24.safetensors
                config.json
    """
    heads_dir = os.path.join(output_dir, "exit_heads")
    os.makedirs(heads_dir, exist_ok=True)

    config = {
        "exit_layer_indices": wrapper.exit_layer_indices,
        "hidden_size": wrapper.hidden_size,
        "vocab_size": wrapper.vocab_size,
    }

    for idx in wrapper.exit_layer_indices:
        head = wrapper.exit_heads[str(idx)]
        state = {k: v.contiguous() for k, v in head.state_dict().items()}
        path = os.path.join(heads_dir, f"exit_head_{idx}.safetensors")
        save_file(state, path)

    with open(os.path.join(heads_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return heads_dir


def load_exit_heads(
    exit_heads_dir: str,
    device: str = "cuda",
) -> Tuple[Dict[int, ExitHead], dict]:
    """
    Load saved exit heads and their config.

    Returns:
        heads: {layer_idx: ExitHead} on the specified device
        config: the saved config dict
    """
    with open(os.path.join(exit_heads_dir, "config.json")) as f:
        config = json.load(f)

    heads: Dict[int, ExitHead] = {}
    for idx in config["exit_layer_indices"]:
        head = ExitHead(config["hidden_size"], config["vocab_size"])
        state = load_file(
            os.path.join(exit_heads_dir, f"exit_head_{idx}.safetensors"),
            device=device,
        )
        head.load_state_dict(state)
        head.to(device)
        head.eval()
        heads[idx] = head

    return heads, config


def push_exit_heads_to_hub(
    exit_heads_dir: str,
    repo_id: str,
    token: str | None = None,
) -> str:
    """
    Upload exit head weights + config to a HuggingFace Hub repository.

    Returns the repo URL.
    """
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=exit_heads_dir,
        repo_id=repo_id,
        commit_message="Upload early-exit heads",
    )
    return f"https://huggingface.co/{repo_id}"


def push_training_logs_to_hub(
    logs_dir: str,
    repo_id: str,
    token: str | None = None,
    path_in_repo: str = "logs/train",
) -> str:
    """
    Upload local training/eval log artifacts (JSON files) to Hugging Face Hub.
    """
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=logs_dir,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        commit_message="Upload early-exit training logs",
    )
    return f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}"


def push_benchmark_results_to_hub(
    results_json_path: str,
    repo_id: str,
    token: str | None = None,
    path_in_repo: str | None = None,
) -> str:
    """
    Upload one benchmark/evaluation JSON artifact to Hugging Face Hub.
    """
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True)

    target_path = path_in_repo or f"logs/benchmark/{os.path.basename(results_json_path)}"
    api.upload_file(
        path_or_fileobj=results_json_path,
        path_in_repo=target_path,
        repo_id=repo_id,
        commit_message="Upload benchmark results",
    )
    return f"https://huggingface.co/{repo_id}/blob/main/{target_path}"
