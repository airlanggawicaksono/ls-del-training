# Early-Exit Tuning for LLaMA-3-8B

Train lightweight early-exit heads on a frozen LLaMA-3-8B backbone.  
Each exit head (RMSNorm + Linear) is attached at an intermediate transformer layer and learns to predict the next token from that layer's hidden state — without modifying the base model.

Based on the EE-LLM approach ([pan-x-c/EE-LLM](https://github.com/pan-x-c/EE-LLM)), adapted for single-GPU HuggingFace Trainer.

## Why

Not all tokens need 32 layers. Function words, common continuations, and predictable patterns can be resolved much earlier. Early-exit heads let you measure *where* in the network predictions become "good enough" and optionally skip the remaining layers at inference time.

## How It Works

```
Input tokens
    |
    v
[Layer 0] -> [Layer 1] -> ... -> [Layer 7] -> [Layer 8] ---> ExitHead_8 ---> logits
                                                  |
                                              [Layer 9] -> ... -> [Layer 16] ---> ExitHead_16 ---> logits
                                                                      |
                                                                  [Layer 17] -> ... -> [Layer 24] ---> ExitHead_24 ---> logits
                                                                                          |
                                                                                      [Layer 25] -> ... -> [Layer 31] -> norm -> lm_head -> logits
```

**Training**: All 32 layers run (backbone frozen). Forward hooks capture hidden states at exit layers. Each exit head computes cross-entropy loss against the same next-token labels. Total loss = weighted sum across exits.

**Inference**: Layers run one-by-one. At each exit, if `max(softmax(logits)) > threshold`, accept the token and skip remaining layers.

## Project Structure

```
finetune.py              # original training entry point (unchanged)
finetune_ee.py           # early-exit training entry point
config_types.py          # base TrainConfig (unchanged)
config_types_ee.py       # EETrainConfig (extends TrainConfig)
trainer_utils.py         # shared utilities (unchanged)
train_config.example     # base training config sample
ee_train_config.example  # early-exit training config sample
requirements.txt

ee/
  exit_head.py           # ExitHead module: RMSNorm -> Linear(4096, vocab)
  model_wrapper.py       # EarlyExitLlamaWrapper: frozen base + hooks + heads
  loss.py                # multi-exit weighted cross-entropy
  trainer.py             # EarlyExitTrainer (saves only exit heads, not 8B base)
  callbacks.py           # GPU metrics: VRAM, step time, tokens/sec
  train.py               # training orchestrator
  evaluate.py            # per-exit perplexity / accuracy comparison
  inference.py           # confidence-based early-exit generation
  hub.py                 # save/load/push exit heads (safetensors)
  utils.py               # freeze, param counting
```

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login  # for gated model access + hub upload
```

## Training

1. Copy and edit the config:

```bash
cp ee_train_config.example my_config
# edit my_config: set train_file, output_dir, hub repo, etc.
```

2. Run:

```bash
python finetune_ee.py --config my_config
```

What happens:
- Loads LLaMA-3-8B, freezes all 8B parameters
- Attaches exit heads at layers 8, 16, 24 (initialized from base model's final norm + lm_head)
- Trains only the exit heads (~1.6B trainable params across 3 heads)
- Logs per-exit loss + GPU metrics at each step
- Saves exit heads in safetensors format (not the 16GB base model)
- Optionally uploads to HuggingFace Hub

## Config Reference

All base config fields from `train_config.example` are supported, plus:

| Field | Default | Description |
|---|---|---|
| `exit_layer_indices` | `[8, 16, 24]` | Which layers get exit heads (0-indexed) |
| `exit_loss_weights` | `[1.0, 1.0, 1.0]` | Loss weight per exit |
| `init_exit_from_base` | `true` | Init heads from base model's norm + lm_head |
| `exit_confidence_threshold` | `0.9` | Softmax threshold for early exit at inference |
| `hub_exit_heads_repo` | `none` | HF Hub repo for uploading exit heads |

## Evaluation Output

After training, per-exit evaluation prints a comparison table:

```
  Layer | Loss   | Perplexity | Accuracy
  ------+--------+------------+---------
  *   8 | 3.8100 |      45.15 | 0.3100
  *  16 | 3.1000 |      22.20 | 0.4200
  *  24 | 2.5500 |      12.81 | 0.5100
    32  | 2.1000 |       8.17 | 0.5800
```

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ee.hub import load_exit_heads
from ee.inference import EarlyExitGenerator

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
exit_heads, config = load_exit_heads("outputs/llama-ee/exit_heads", device="cuda")

generator = EarlyExitGenerator(base_model, exit_heads, tokenizer, confidence_threshold=0.9)
result = generator.generate("The capital of France is", max_new_tokens=64)

print(result["text"])
generator.print_exit_statistics()
```

```
Exit statistics (42 tokens total):
  Layer | Count | Percent
  ------+-------+--------
  EE   8 |    12 |  28.6%
  EE  16 |    15 |  35.7%
  EE  24 |     9 |  21.4%
  FL  31 |     6 |  14.3%
```

## KV Cache Note

The current inference implementation re-processes the full sequence at each generation step (no KV cache). This is intentional — it gives correct results for research/comparison without the complexity of managing KV cache gaps when tokens exit at different layers. For production use, KV cache support would need to be added separately.

## Original Training

The base (non-early-exit) training pipeline is untouched:

```bash
python finetune.py --config train_config.example
```
