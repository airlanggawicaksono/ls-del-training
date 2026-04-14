import time
from typing import Dict

import psutil
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

# pynvml for GPU utilization % (not just VRAM)
_nvml_available = False
try:
    import pynvml

    pynvml.nvmlInit()
    _nvml_available = True
except Exception:
    pass


def _gpu_utilization() -> Dict[str, float]:
    """Query nvidia-smi via pynvml for GPU util %, temp, power."""
    if not _nvml_available:
        return {}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
        return {
            "gpu/utilization_pct": float(util.gpu),
            "gpu/memory_util_pct": float(util.memory),
            "gpu/temperature_c": float(temp),
            "gpu/power_w": round(power, 1),
        }
    except Exception:
        return {}


class TrainingMetricsCallback(TrainerCallback):
    """
    Logs hardware metrics at each training step:

      GPU:
        - gpu/vram_allocated_gb    current VRAM in use
        - gpu/vram_peak_gb         peak VRAM since last step
        - gpu/utilization_pct      GPU compute utilization (nvidia-smi)
        - gpu/memory_util_pct      GPU memory controller utilization
        - gpu/temperature_c        GPU temperature
        - gpu/power_w              GPU power draw

      CPU:
        - cpu/usage_pct            process CPU usage since last step
        - cpu/ram_used_gb          process resident memory

      Throughput:
        - throughput/step_time_sec    wall-clock time for the step
        - throughput/tokens_per_sec   tokens processed per second
    """

    def __init__(self, seq_length: int = 2048):
        self.seq_length = seq_length
        self._step_start: float = 0.0
        self._process = psutil.Process()
        # Prime cpu_percent so first call returns meaningful value
        self._process.cpu_percent()

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
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
        metrics: Dict[str, float] = {}

        # -- GPU VRAM (torch) --
        if torch.cuda.is_available():
            metrics["gpu/vram_allocated_gb"] = round(
                torch.cuda.memory_allocated() / (1024**3), 3
            )
            metrics["gpu/vram_peak_gb"] = round(
                torch.cuda.max_memory_allocated() / (1024**3), 3
            )

        # -- GPU utilization (pynvml) --
        metrics.update(_gpu_utilization())

        # -- CPU --
        metrics["cpu/usage_pct"] = round(self._process.cpu_percent(), 1)
        mem_info = self._process.memory_info()
        metrics["cpu/ram_used_gb"] = round(mem_info.rss / (1024**3), 3)

        # -- Throughput --
        metrics["throughput/step_time_sec"] = round(elapsed, 4)
        batch_tokens = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.seq_length
        )
        if elapsed > 0:
            metrics["throughput/tokens_per_sec"] = round(batch_tokens / elapsed, 1)

        # Attach to log history
        if state.is_world_process_zero:
            state.log_history.append({"step": state.global_step, **metrics})

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        print("\n--- Training Hardware Summary ---")
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  GPU peak VRAM:  {peak:.2f} GB")
        print(f"  CPU RAM (now):  {self._process.memory_info().rss / (1024**3):.2f} GB")
        if _nvml_available:
            info = _gpu_utilization()
            if info:
                print(f"  GPU temp:       {info.get('gpu/temperature_c', '?')} C")
                print(f"  GPU power:      {info.get('gpu/power_w', '?')} W")
        print("--------------------------------\n")
