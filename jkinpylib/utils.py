import os
import random

import pkg_resources
import torch
import numpy as np

from jkinpylib.config import DEVICE, DEFAULT_TORCH_DTYPE
from jkinpylib.conversions import PT_NP_TYPE


def get_filepath(local_filepath: str):
    return pkg_resources.resource_filename(__name__, local_filepath)


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(0)
    print("set_seed() - random int: ", torch.randint(0, 1000, (1, 1)).item())


def to_torch(x: PT_NP_TYPE, device: str = DEVICE, dtype: torch.dtype = DEFAULT_TORCH_DTYPE) -> torch.Tensor:
    """Return a numpy array as a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, device=device, dtype=dtype)


def to_numpy(x: PT_NP_TYPE) -> np.ndarray:
    """Return a tensor/np array as a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x
