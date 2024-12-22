"""Global configuration"""

from typing import Union, Tuple
import os
from time import sleep

import torch
import numpy as np


def _get_device() -> Tuple[str, int]:
    if not torch.cuda.is_available():
        return "mps" if torch.backends.mps.is_available() else "cpu", -1

    n_devices = torch.cuda.device_count()
    if n_devices == 1:
        return "cuda:0", 0

    def _get_devices_usage():
        mems = [[] for _ in range(n_devices)]
        utils = [[] for _ in range(n_devices)]

        devices = [torch.cuda.device(i) for i in range(n_devices)]

        for _ in range(10):
            for i in range(n_devices):
                mems[i].append(torch.cuda.memory_usage(device=devices[i]))
                utils[i].append(torch.cuda.utilization(device=devices[i]))
            sleep(0.01)
        return [sum(ms) / len(mems[0]) for ms in mems], [sum(usg) / len(utils[0]) for usg in utils]

    ave_mems, ave_utils = _get_devices_usage()
    min_mem_idx = min(range(n_devices), key=lambda i: ave_mems[i])
    min_util_idx = min(range(n_devices), key=lambda i: ave_utils[i])

    print("Jrl/config.py: _get_device()", flush=True)
    print(f"  Average memory usage, per device: {ave_mems}", flush=True)
    print(f"  Average utilization, per device:  {ave_utils}", flush=True)
    print(f"  Lowest memory device:             'cuda:{min_mem_idx}'", flush=True)
    print(f"  Lowest utilization device:        'cuda:{min_util_idx}'", flush=True)

    # If same device has lowest memory and utilization, return that one
    if min_mem_idx == min_util_idx:
        print(
            f"  Using device 'cuda:{min_mem_idx}' - it has both the lowest memory and utilization percent", flush=True
        )
        return f"cuda:{min_mem_idx}", min_mem_idx

    device_pct_sums = [ave_mems[i] + ave_utils[i] for i in range(n_devices)]
    min_pct_sum_idx = min(range(n_devices), key=lambda i: device_pct_sums[i])
    print(
        f"  Using device 'cuda:{min_pct_sum_idx}' - it has the lowest sum of memory and utilization percentages",
        flush=True,
    )

    # Warn if chosen device has high memory or utilization
    if ave_mems[min_pct_sum_idx] > 20:
        print(
            f"  WARNING: Chosen device 'cuda:{min_pct_sum_idx}' has high memory usage:"
            f" {ave_mems[min_pct_sum_idx]:.1f}%",
            flush=True,
        )
    if ave_utils[min_pct_sum_idx] > 20:
        print(
            f"  WARNING: Chosen device 'cuda:{min_pct_sum_idx}' has high utilization:"
            f" {ave_utils[min_pct_sum_idx]:.1f}%",
            flush=True,
        )

    return f"cuda:{min_pct_sum_idx}", min_pct_sum_idx


DEVICE, GPU_IDX = _get_device()
ACCELERATOR_AVAILABLE = torch.cuda.is_available() or torch.backends.mps.is_available()
DEFAULT_TORCH_DTYPE = torch.float32

PT_NP_TYPE = Union[np.ndarray, torch.Tensor]

URDF_DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), ".cache/jrl/urdfs")

torch.set_default_dtype(DEFAULT_TORCH_DTYPE)
torch.set_default_device(DEVICE)
