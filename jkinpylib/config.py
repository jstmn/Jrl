"""Global configuration"""
from typing import Union
import os

import torch
import numpy as np

# /
DEVICE = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ACCELERATOR_AVAILABLE = torch.cuda.is_available() or torch.backends.mps.is_available()
DEVICE = "cpu"
ACCELERATOR_AVAILABLE = False
DEFAULT_TORCH_DTYPE = torch.float32

PT_NP_TYPE = Union[np.ndarray, torch.Tensor]

URDF_DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), ".cache/jkinpylib/temp_urdfs")

torch.set_default_dtype(DEFAULT_TORCH_DTYPE)
torch.set_default_device(DEVICE)
