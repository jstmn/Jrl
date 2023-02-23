import os
import random

from typing import Union

import pkg_resources
import torch
import numpy as np

from jkinpylib.config import DEVICE, DEFAULT_TORCH_DTYPE

PT_NP_TYPE = Union[np.ndarray, torch.Tensor]


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


def quatconj(q: PT_NP_TYPE):
    """
    Given rows of quaternions q, compute quaternion conjugate
    """
    if isinstance(q, torch.Tensor):
        stacker = torch.hstack
    if isinstance(q, np.ndarray):
        stacker = np.hstack

    w, x, y, z = tuple(q[:, i] for i in range(4))
    return stacker((w, -x, -y, -z)).reshape(q.shape)


def quatmul(q1: PT_NP_TYPE, q2: PT_NP_TYPE):
    """
    Given rows of quaternions q1 and q2, compute the Hamilton product q1 * q2
    """
    assert q1.shape[1] == 4
    assert q1.shape == q2.shape
    if isinstance(q1, torch.Tensor) and isinstance(q2, torch.Tensor):
        stacker = torch.hstack
    if isinstance(q1, np.ndarray) and isinstance(q2, np.ndarray):
        stacker = np.hstack

    w1, x1, y1, z1 = tuple(q1[:, i] for i in range(4))
    w2, x2, y2, z2 = tuple(q2[:, i] for i in range(4))

    return stacker(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    ).reshape(q1.shape)
