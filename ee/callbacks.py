import time
from collections import defaultdict
from typing import Dict, List

import psutil
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

_nvml_available = False
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_available = True
except Exception:
    pass


def _device_caps() -> Dict:
    """Static device capacity — total VRAM, power limit, CPU count, RAM."""
    caps: Dict = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        caps["gpu_name"] = props.name
        caps["gpu_vram_total_gb"] = round(props.total_memory / (1024 ** 3), 2)
    if _nvml_available:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            caps["gpu_power_limit_w"] = round(pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0, 1)
        except Exception:
            pass
    caps["cpu_count_physical"] = psutil.cpu_count(logical=False)
    caps["cpu_count_logical"] = psutil.cpu_count(logical=True)
    caps["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    return caps


def _gpu_utilization() -> Dict[str, float]:
    if not _nvml_available:
        return {}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        return {
            "gpu/utilization_pct": float(util.gpu),
            "gpu/memory_util_pct": float(util.memory),
            "gpu/temperature_c": float(temp),
            "gpu/power_w": round(power, 1),
        }
    except Exception:
        return {}


def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    return {
        "mean": round(sum(vals) / len(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


class TrainingMetricsCallback(TrainerCallback):
    """
    Tracks hardware + energy per step, per-exit loss per epoch.

    epoch_metrics structure (one entry per epoch):
    {
        "epoch": 1,
        "hardware": {
            "energy_j": ...,
            "energy_wh": ...,
            "time_min": ...,
            "gpu_util_pct":   {"mean": ..., "min": ..., "max": ...},
            "gpu_power_w":    {"mean": ..., "min": ..., "max": ...},
            "gpu_vram_gb":    {"mean": ..., "peak": ...},
            "cpu_usage_pct":  {"mean": ..., "max": ...},
            "tokens_per_sec": {"mean": ..., "min": ...},
        },
        "exits": {
            "8":  {"loss": {"mean": ..., "min": ..., "max": ...}},
            "16": {"loss": {"mean": ..., "min": ..., "max": ...}},
            "24": {"loss": {"mean": ..., "min": ..., "max": ...}},
        },
        "base_final": {"loss": {"mean": ..., "min": ..., "max": ...}},
    }

    Step-level metrics (injected into trainer log pipeline per logging_steps):
      gpu/*, cpu/*, throughput/*, energy/*
    """

    def __init__(self, seq_length: int = 2048):
        self.seq_length = seq_length
        self.device_caps = _device_caps()
        self._step_start: float = 0.0
        self._train_start: float = 0.0
        self._epoch_start: float = 0.0
        self._process = psutil.Process()
        self._process.cpu_percent()

        self._pending: Dict[str, float] = {}
        self._total_energy_j: float = 0.0

        # Hardware buffers — reset each epoch
        self._hw: Dict[str, List[float]] = defaultdict(list)
        self._epoch_energy_j: float = 0.0

        # Per-exit loss buffers — reset each epoch
        # { "8": [loss, loss, ...], "16": [...], ... }
        self._exit_losses: Dict[str, List[float]] = defaultdict(list)
        self._base_losses: List[float] = []

        # Final output — one entry per epoch
        self.epoch_metrics: List[Dict] = []

    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start = time.perf_counter()
        self._epoch_start = self._train_start
        self._total_energy_j = 0.0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.perf_counter()
        self._epoch_energy_j = 0.0
        self._hw.clear()
        self._exit_losses.clear()
        self._base_losses.clear()

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._step_start = time.perf_counter()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        elapsed = time.perf_counter() - self._step_start
        m: Dict[str, float] = {}

        if torch.cuda.is_available():
            m["gpu/vram_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 3)
            m["gpu/vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)

        gpu_info = _gpu_utilization()
        m.update(gpu_info)

        m["cpu/usage_pct"] = round(self._process.cpu_percent(), 1)
        m["cpu/ram_used_gb"] = round(self._process.memory_info().rss / (1024**3), 3)

        m["throughput/step_time_sec"] = round(elapsed, 4)
        batch_tokens = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.seq_length
        )
        m["throughput/tokens_per_sec"] = round(batch_tokens / elapsed, 1) if elapsed > 0 else 0.0

        power_w = gpu_info.get("gpu/power_w", 0.0)
        step_energy_j = power_w * elapsed
        self._total_energy_j += step_energy_j
        self._epoch_energy_j += step_energy_j

        m["energy/step_j"] = round(step_energy_j, 3)
        m["energy/total_j"] = round(self._total_energy_j, 1)
        m["energy/total_wh"] = round(self._total_energy_j / 3600, 4)
        m["timing/elapsed_min"] = round((time.perf_counter() - self._train_start) / 60, 2)

        # Buffer hardware metrics for epoch aggregation
        for key in ("gpu/utilization_pct", "gpu/power_w", "gpu/vram_allocated_gb",
                    "gpu/vram_peak_gb", "cpu/usage_pct", "throughput/tokens_per_sec"):
            if key in m:
                self._hw[key].append(m[key])

        self._pending = m

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict = None,
        **kwargs,
    ) -> None:
        if logs is None:
            return
        # Inject hardware metrics into live log pipeline
        if self._pending:
            logs.update(self._pending)

        # Scrape per-exit losses from the trainer's log dict
        for key, val in logs.items():
            if key.startswith("loss_exit_"):
                layer_id = key[len("loss_exit_"):]
                self._exit_losses[layer_id].append(float(val))
            elif key == "loss_base_final":
                self._base_losses.append(float(val))

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        hw = self._hw
        epoch_time = time.perf_counter() - self._epoch_start

        hardware = {
            "energy_j": round(self._epoch_energy_j, 1),
            "energy_wh": round(self._epoch_energy_j / 3600, 4),
            "time_min": round(epoch_time / 60, 2),
            "gpu_util_pct": _stats(hw.get("gpu/utilization_pct", [])),
            "gpu_power_w": _stats(hw.get("gpu/power_w", [])),
            "gpu_vram_gb": {
                "mean": round(sum(hw.get("gpu/vram_allocated_gb", [0])) / max(len(hw.get("gpu/vram_allocated_gb", [1])), 1), 3),
                "peak": round(max(hw.get("gpu/vram_peak_gb", [0])), 3),
            },
            "cpu_usage_pct": _stats(hw.get("cpu/usage_pct", [])),
            "tokens_per_sec": _stats(hw.get("throughput/tokens_per_sec", [])),
        }

        exits = {
            layer_id: {"loss": _stats(losses)}
            for layer_id, losses in sorted(self._exit_losses.items(), key=lambda x: int(x[0]))
        }

        record = {
            "epoch": round(float(state.epoch), 1),
            "device_caps": self.device_caps,
            "hardware": hardware,
            "exits": exits,
        }
        if self._base_losses:
            record["base_final"] = {"loss": _stats(self._base_losses)}

        self.epoch_metrics.append(record)

        if state.is_world_process_zero:
            exit_str = "  ".join(
                f"L{lid}={v['loss'].get('mean', '?'):.3f}"
                for lid, v in exits.items()
            )
            print(
                f"\n[Epoch {state.epoch:.0f}] "
                f"time={hardware['time_min']:.1f}min  "
                f"energy={hardware['energy_j']:.0f}J  "
                f"gpu={hardware['gpu_util_pct'].get('mean', 0):.0f}%  "
                f"vram={hardware['gpu_vram_gb']['peak']:.2f}GB  "
                f"{exit_str}"
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        total_sec = time.perf_counter() - self._train_start
        caps = self.device_caps
        print("\n--- Training Summary ---")
        print(f"  GPU:           {caps.get('gpu_name', 'unknown')}")
        print(f"  Total time:    {total_sec / 60:.1f} min")
        print(f"  Total energy:  {self._total_energy_j:.0f} J  |  {self._total_energy_j / 3600:.3f} Wh")
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            total_gb = caps.get("gpu_vram_total_gb", "?")
            print(f"  GPU peak VRAM: {peak_gb:.2f} GB / {total_gb} GB")
        if _nvml_available:
            info = _gpu_utilization()
            if info:
                limit_w = caps.get("gpu_power_limit_w", "?")
                print(f"  GPU power now: {info.get('gpu/power_w', '?')} W / {limit_w} W limit")
                print(f"  GPU temp:      {info.get('gpu/temperature_c', '?')} C")
        cpu_gb = self._process.memory_info().rss / (1024 ** 3)
        ram_total = caps.get("ram_total_gb", "?")
        print(f"  CPU RAM (now): {cpu_gb:.2f} GB / {ram_total} GB")
        print(f"  CPU cores:     {caps.get('cpu_count_physical', '?')} physical / {caps.get('cpu_count_logical', '?')} logical")
        print("------------------------\n")
