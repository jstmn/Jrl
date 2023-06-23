import os
import random
import pathlib

import pkg_resources
import torch
import numpy as np

from jrl.config import DEVICE, DEFAULT_TORCH_DTYPE
from jrl.config import PT_NP_TYPE


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


class QP:
    def __init__(self, Q, p, G, h, A=None, b=None):
        """Solve a quadratic program of the form
        min 1/2 x^T Q x + p^T x
        s.t. Gx <= h, Ax = b

        Args:
            Q (torch.Tensor): [nbatch x dim x dim] positive semidefinite matrix
            p (torch.Tensor): [nbatch x dim] vector
            G (torch.Tensor): [nbatch x nc x dim] matrix
            h (torch.Tensor): [nbatch x nc] vector
            A (torch.Tensor, optional): [nbatch x neq x dim] matrix. Defaults to
            None.
            b (torch.Tensor, optional): [nbatch x neq] vector. Defaults to None.

        Raises:
            NotImplementedError: Equality constraints not implemented

        Returns:
            torch.Tensor: [nbatch x dim] solution
        """
        self.nbatch = Q.shape[0]
        self.dim = Q.shape[1]
        self.nc = G.shape[1]
        self.Q = Q
        assert len(p.shape) == 2, f"p.shape: {p.shape}"
        self.p = p.unsqueeze(2)
        self.G = G
        assert len(h.shape) == 2, f"h.shape: {h.shape}"
        self.h = h.unsqueeze(2)

        assert (
            self.nbatch == self.p.shape[0] == self.G.shape[0] == self.h.shape[0] == self.Q.shape[0]
        ), f"{self.nbatch}, {self.p.shape[0]}, {self.G.shape[0]}, {self.h.shape[0]}, {self.Q.shape[0]}"
        assert (
            self.dim == self.p.shape[1] == self.G.shape[2] == self.Q.shape[1] == self.Q.shape[2]
        ), f"{self.dim}, {self.p.shape[1]}, {self.G.shape[2]}, {self.Q.shape[1]}, {self.Q.shape[2]}"
        assert self.nc == self.G.shape[1] == self.h.shape[1]

        if A is not None or b is not None:
            raise NotImplementedError("Equality constraints not implemented")
        self.A = A
        self.b = b

    def solve(self, trace=False, iterlimit=None):
        x = torch.zeros((self.nbatch, self.dim, 1), dtype=torch.float32, device=self.Q.device)
        if trace:
            trace = [x]
        working_set = torch.zeros((self.nbatch, self.nc, 1), dtype=torch.bool, device=x.device)
        converged = torch.zeros((self.nbatch, 1, 1), dtype=torch.bool, device=x.device)

        if iterlimit is None:
            iterlimit = 10 * self.nc

        iterations = 0
        while not torch.all(converged):
            A = self.G * working_set
            b = self.h * working_set

            g = self.Q.bmm(x) + self.p
            h = A.bmm(x) - b

            AT = A.transpose(1, 2)
            AQinvAT = A.bmm(torch.linalg.solve(self.Q, AT))
            rhs = h - A.bmm(torch.linalg.solve(self.Q, g))
            diag = torch.diag_embed(~working_set.squeeze(dim=2))
            lmbd = torch.linalg.solve(AQinvAT + diag, rhs)
            p = torch.linalg.solve(self.Q, -AT.bmm(lmbd) - g)

            at_ws_sol = torch.linalg.norm(p, dim=1, keepdim=True) < 1e-3

            lmbdmin, lmbdmin_idx = torch.min(
                torch.where(
                    at_ws_sol & working_set,
                    lmbd,
                    torch.inf,
                ),
                dim=1,
                keepdim=True,
            )

            converged = converged | (at_ws_sol & (lmbdmin >= 0))

            # remove inverted constraints from working set
            working_set = working_set.scatter(
                1,
                lmbdmin_idx,
                (~at_ws_sol | (lmbdmin >= 0)) & working_set.gather(1, lmbdmin_idx),
            )

            # check constraint violations
            mask = ~at_ws_sol & ~working_set & (self.G.bmm(p) > 0)
            alpha, alpha_idx = torch.min(
                torch.where(
                    mask,
                    (self.h - self.G.bmm(x)) / (self.G.bmm(p)),
                    torch.inf,
                ),
                dim=1,
                keepdim=True,
            )

            # add violated constraints to working set
            working_set = working_set.scatter(
                1,
                alpha_idx,
                working_set.gather(1, alpha_idx) | (~at_ws_sol & (alpha < 1)),
            )
            alpha = torch.clamp(alpha, max=1)

            # update solution
            x = x + alpha * p * (~at_ws_sol & ~converged)
            if trace:
                trace.append(x.squeeze(2))

            iterations += 1
            if iterations > iterlimit:
                Qs = self.Q[~converged.flatten()]
                ps = self.p[~converged.flatten()]
                Gs = self.G[~converged.flatten()]
                hs = self.h[~converged.flatten()]
                xs = x[~converged.flatten()]
                for i in range(torch.sum(~converged)):
                    print(f"Qs[{i}]:\n{Qs[i]}")
                    print(f"ps[{i}]:\n{ps[i]}")
                    print(f"Gs[{i}]:\n{Gs[i]}")
                    print(f"hs[{i}]:\n{hs[i]}")
                    print(f"xs[{i}]:\n{xs[i]}")
                raise RuntimeError(
                    f"Failed to converge in {iterlimit} iterations\n\n{torch.sum(~converged).item()} out of"
                    f" {self.nbatch} not converged"
                )

        if trace:
            return x.squeeze(2), trace

        return x.squeeze(2)
