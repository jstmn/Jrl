import os
import random
import pathlib

import pkg_resources
import torch
import numpy as np

from jkinpylib.config import DEVICE, DEFAULT_TORCH_DTYPE
from jkinpylib.config import PT_NP_TYPE


def safe_mkdir(dir_name: str):
    """Create a directory `dir_name`. May include multiple levels of new directories"""
    pathlib.Path(dir_name).mkdir(exist_ok=True, parents=True)


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


# Borrowed from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#random_quaternions
def random_quaternions(n: int, device: DEVICE) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """

    def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Return a tensor where each element has the absolute value taken from the,
        corresponding element of a, with sign taken from the corresponding
        element of b. This is like the standard copysign floating-point operation,
        but is not careful about negative 0 and NaN.

        Args:
            a: source tensor.
            b: tensor whose signs will be used, of the same shape as a.

        Returns:
            Tensor of the same shape as a with the signs of b.
        """
        signs_differ = (a < 0) != (b < 0)
        return torch.where(signs_differ, -a, a)

    o = torch.randn((n, 4), dtype=DEFAULT_TORCH_DTYPE, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o
