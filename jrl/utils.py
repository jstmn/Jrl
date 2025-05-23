import os
import random
import pathlib
import colorsys

import torch
import numpy as np

from jrl.config import DEVICE, DEFAULT_TORCH_DTYPE, PT_NP_TYPE


def cm_to_m(x: float):
    return x / 100.0


def mm_to_m(x: float):
    return x / 1000.0


def safe_mkdir(dir_name: str):
    """Create a directory `dir_name`. May include multiple levels of new directories"""
    pathlib.Path(dir_name).mkdir(exist_ok=True, parents=True)


def get_filepath(local_filepath: str):
    return os.path.join(os.path.dirname(__file__), local_filepath)


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


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def make_text_green_or_red(text: str, print_green: bool) -> str:
    if print_green:
        s = bcolors.GREEN
    else:
        s = bcolors.FAIL
    return s + str(text) + bcolors.ENDC


def evenly_spaced_colors(n: int):
    base_color = 100  # Base color in HSL (green)
    colors = []
    for i in range(n):
        hue = (base_color + (i * (360 / n))) % 360  # Adjust the step size (30 degrees in this case)
        rgb = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
        colors.append(rgb)
    return colors


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
