"""Global configuration"""

from typing import Union
import os
from time import sleep

import torch
import numpy as np


def _get_device() -> str:
    if not torch.cuda.is_available():
        return "mps" if torch.backends.mps.is_available() else "cpu", -1

    n_devices = torch.cuda.device_count()
    if n_devices == 1:
        return "cuda:0", 0

    def _mem_and_utilitization_ave_usage_pct(device_idx: int):
        device = torch.cuda.device(device_idx)
        mems = []
        utils = []
        for _ in range(2):
            mems.append(torch.cuda.memory_usage(device=device))
            utils.append(torch.cuda.utilization(device=device))
            sleep(0.5)
        return sum(mems) / len(mems), sum(utils) / len(utils)

    min_mem = 100
    min_util = 100
    for i in range(n_devices):
        mem_pct, util_pct = _mem_and_utilitization_ave_usage_pct(i)
        min_mem = min(min_mem, mem_pct)
        min_util = min(min_util, util_pct)
        if mem_pct < 5.0 and util_pct < 9:
            return f"cuda:{i}", i
    raise EnvironmentError(f"No unused GPU's available. Minimum memory, utilization: {min_mem, min_util}%")


DEVICE, GPU_IDX = _get_device()
ACCELERATOR_AVAILABLE = torch.cuda.is_available() or torch.backends.mps.is_available()
DEFAULT_TORCH_DTYPE = torch.float32

PT_NP_TYPE = Union[np.ndarray, torch.Tensor]

URDF_DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), ".cache/jrl/temp_urdfs")

torch.set_default_dtype(DEFAULT_TORCH_DTYPE)
torch.set_default_device(DEVICE)
