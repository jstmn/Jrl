import torch
import numpy as np

from jrl.utils import to_torch
from jrl.math_utils import geodesic_distance_between_quaternions
from jrl.config import DEVICE, DEFAULT_TORCH_DTYPE, PT_NP_TYPE

_DEFAULT_MAX_ALLOWABLE_L2_ERR = 5e-4
_DEFAULT_MAX_ALLOWABLE_ANG_ERR = 3.141592 / 180 * 0.06


def assert_pose_positions_almost_equal(
    endpoints1: PT_NP_TYPE,
    endpoints2: PT_NP_TYPE,
    source_1: str = "",
    source_2: str = "",
    threshold: float = _DEFAULT_MAX_ALLOWABLE_L2_ERR,
):
    """Check that the position of each pose is nearly the same"""
    if isinstance(endpoints1, torch.Tensor):
        l2_errors = torch.norm(endpoints1[:, 0:3] - endpoints2[:, 0:3], dim=1)
    else:
        l2_errors = np.linalg.norm(endpoints1[:, 0:3] - endpoints2[:, 0:3], axis=1)
    for i in range(l2_errors.shape[0]):
        assert (
            l2_errors[i] < threshold
        ), f"Position of poses '{source_1}', '{source_2}' are not equal (error={l2_errors[i]})"


def assert_pose_rotations_almost_equal(
    endpoints1: np.array,
    endpoints2: np.array,
    source_1: str = "",
    source_2: str = "",
    threshold: float = _DEFAULT_MAX_ALLOWABLE_ANG_ERR,
):
    """Check that the rotation of each pose is nearly the same"""
    endpoints1 = to_torch(endpoints1)
    endpoints2 = to_torch(endpoints2)
    assert endpoints1.shape[1] == 7
    assert endpoints2.shape[1] == 7
    assert endpoints1.shape[0] == endpoints2.shape[0]
    errors = geodesic_distance_between_quaternions(endpoints1[:, 3 : 3 + 4], endpoints2[:, 3 : 3 + 4])
    for i in range(errors.shape[0]):
        assert errors[i] < threshold, (
            f"Rotation of poses '{source_1}({endpoints1[i, :]})', '{source_2} ({endpoints2[i, :]})' are not equal"
            f" (error={errors[i]})"
        )
