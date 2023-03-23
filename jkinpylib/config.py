"""Global configuration"""
from typing import Union
import os

import torch
import numpy as np

# /
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_TORCH_DTYPE = torch.float32

PT_NP_TYPE = Union[np.ndarray, torch.Tensor]

URDF_DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), ".cache/jkinpylib/temp_urdfs")
