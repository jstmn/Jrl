""" This file contains conversion functions between rotation representations, as well as implementations for various
mathematical operations.

A couple notes:
    1. Quaternions are assumed to be in w,x,y,z format
    2. RPY format is a rotation about x, y, z axes in that order
    3. Functions that end with '_np' exclusively accept numpy arrays, those that end with '_pt' exclusively accept torch
        tensors
"""

from typing import Tuple, Callable, Optional, Union
import warnings

import torch
import numpy as np
import roma.mappings

from jkinpylib.config import DEFAULT_TORCH_DTYPE, DEVICE, PT_NP_TYPE

_TORCH_EPS_CPU = torch.tensor(1e-8, dtype=DEFAULT_TORCH_DTYPE, device="cpu")
_TORCH_EPS_CUDA = torch.tensor(1e-8, dtype=DEFAULT_TORCH_DTYPE, device="cuda")


def enforce_pt_np_input(func: Callable):
    """Performs the following checks:
    1. The function recieves either 1 or 2 arguments
    2. Each argument is either a np.ndarray or a torch.Tensor
    3. If there are two arguments, both must be of the same type
    """

    def wrapper(*args, **kwargs):
        are_2_args = len(args) == 2
        assert len(args) == 1 or are_2_args, f"Expected 1 or 2 arguments, got {len(args)}"
        for arg in args:
            assert isinstance(arg, (np.ndarray, torch.Tensor)), f"Expected np.ndarray or torch.Tensor, got {type(arg)}"
        if are_2_args and isinstance(args[0], torch.Tensor):
            assert type(args[0]) is type(
                args[1]
            ), f"Expected both arguments to be of the same type, got {type(args[0])} and {type(args[1])}"
        return func(*args, **kwargs)

    return wrapper


# batch*n
def normalize_vector(v: torch.Tensor, return_mag: bool = False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch

    if v.is_cuda:
        v_mag = torch.max(v_mag, _TORCH_EPS_CUDA)
    else:
        v_mag = torch.max(v_mag, _TORCH_EPS_CPU)
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if return_mag:
        return v, v_mag[:, 0]
    return v


# ======================================================================================================================
#  Rotation matrix functions
#


@enforce_pt_np_input
def rotation_matrix_to_quaternion(m: PT_NP_TYPE) -> PT_NP_TYPE:
    """Converts a batch of rotation matrices to quaternions

    Args:
        m (PT_NP_TYPE): [batch x 3 x 3] tensor of rotation matrices

    Returns:
        PT_NP_TYPE: [batch x 4] tensor of quaternions
    """
    is_np = False
    if isinstance(m, np.ndarray):
        m = torch.tensor(m, dtype=DEFAULT_TORCH_DTYPE, device=DEVICE)
        is_np = True

    quat = roma.mappings.rotmat_to_unitquat(m)
    quat = quaternion_xyzw_to_wxyz(quat)

    if is_np:
        return quat.cpu().numpy()
    return quat


@enforce_pt_np_input
def geodesic_distance_between_rotation_matrices(m1: torch.Tensor, m2: torch.Tensor):
    """Computes the minimum angular distance between the two rotation matrices. In other terms, what's the minimum
    amount of rotation that must be performed to align the two orientations.

    Args:
        m1 (torch.Tensor): [batch x 3 x 3] tensor of rotation matrices
        m2 (torch.Tensor): [batch x 3 x 3] tensor of rotation matrices

    Returns:
        torch.Tensor: [batch] tensor of angles in radians between 0 and pi
    """
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    # See https://github.com/pytorch/pytorch/issues/8069#issuecomment-700397641
    # Note: Decreasing this value below 1e-7 greates NaN gradients for nearby quaternions.
    epsilon = 1e-7
    theta = torch.acos(torch.clamp(cos, -1 + epsilon, 1 - epsilon))
    return theta


# ======================================================================================================================
# Quaternion conversions
#


@enforce_pt_np_input
def quaternion_xyzw_to_wxyz(quaternion: PT_NP_TYPE) -> PT_NP_TYPE:
    """Convert a batch of quaternions from xyzw to wxyz format

    Args:
        quaternion (PT_NP_TYPE): A [batch x 4] tensor or numpy array of quaternions
    """
    if isinstance(quaternion, np.ndarray):
        return np.concatenate([quaternion[:, 3:4], quaternion[:, 0:3]], axis=1)
    if isinstance(quaternion, torch.Tensor):
        return torch.cat([quaternion[:, 3:4], quaternion[:, 0:3]], dim=1)

    raise ValueError(f"quaternion must be a torch.Tensor or np.ndarray (got {type(quaternion)})")


@enforce_pt_np_input
def quaternion_to_rotation_matrix(quat: torch.Tensor):
    """_summary_

    Args:
        quat (torch.Tensor): [batch x 4] tensor of quaternions

    Returns:
        _type_: _description_
    """
    batch, dim = quat.shape
    assert dim == 4
    absnorm = torch.abs(torch.linalg.norm(quat, dim=1))
    assert torch.all(absnorm - 1 < 1e-5), f"Max deviation from unit quaternion: {torch.max(absnorm - 1)}"

    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    matrix = torch.stack(
        (
            torch.stack((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), dim=1),
            torch.stack((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), dim=1),
            torch.stack((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), dim=1),
        ),
        dim=1,
    )

    return matrix


@enforce_pt_np_input
def quaternion_to_rpy(q: PT_NP_TYPE) -> PT_NP_TYPE:
    """Convert a batch of quaternions to roll-pitch-yaw angles"""
    assert len(q.shape) == 2
    assert q.shape[1] == 4

    n = q.shape[0]
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if isinstance(q, np.ndarray):
        rpy = np.zeros((n, 3))
        p = np.arcsin(2 * (q0 * q2 - q3 * q1))
        atan2 = np.arctan2
    elif isinstance(q, torch.Tensor):
        rpy = torch.zeros((n, 3), device=q.device, dtype=DEFAULT_TORCH_DTYPE)
        p = torch.arcsin(2 * (q0 * q2 - q3 * q1))
        atan2 = torch.arctan2
    else:
        raise ValueError(f"q must be a numpy array or a torch tensor (got {type(q)})")

    # handle singularity
    rpy[:, 0] = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    rpy[:, 1] = p
    rpy[:, 2] = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return rpy


@enforce_pt_np_input
def quaternion_conjugate(qs: PT_NP_TYPE) -> PT_NP_TYPE:
    """TODO: document"""
    assert len(qs.shape) == 2
    assert qs.shape[1] == 4

    if isinstance(qs, np.ndarray):
        q_conj = np.zeros(qs.shape)
    elif isinstance(qs, torch.Tensor):
        q_conj = torch.zeros(qs.shape, device=qs.device, dtype=DEFAULT_TORCH_DTYPE)
    else:
        raise ValueError(f"qs must be a numpy array or a torch tensor (got {type(qs)})")

    q_conj[:, 0] = qs[:, 0]
    q_conj[:, 1] = -qs[:, 1]
    q_conj[:, 2] = -qs[:, 2]
    q_conj[:, 3] = -qs[:, 3]
    return q_conj


@enforce_pt_np_input
def quatconj(q: PT_NP_TYPE) -> PT_NP_TYPE:
    """
    Given rows of quaternions q, compute quaternion conjugate

    Author: dmillard
    """
    if isinstance(q, torch.Tensor):
        stacker = torch.vstack
    if isinstance(q, np.ndarray):
        stacker = np.vstack

    w, x, y, z = tuple(q[:, i] for i in range(4))
    return stacker((w, -x, -y, -z)).T


@enforce_pt_np_input
def quaternion_norm(qs: PT_NP_TYPE) -> PT_NP_TYPE:
    """TODO: document"""
    assert len(qs.shape) == 2
    assert qs.shape[1] == 4
    if isinstance(qs, np.ndarray):
        return np.linalg.norm(qs, axis=1)
    if isinstance(qs, torch.Tensor):
        return torch.norm(qs, dim=1)
    raise ValueError(f"qs must be a numpy array or a torch tensor (got {type(qs)})")


@enforce_pt_np_input
def quaternion_inverse(qs: PT_NP_TYPE) -> PT_NP_TYPE:
    """Per "CS184: Using Quaternions to Represent Rotation": The inverse of a unit quaternion is its conjugate, q-1=q'
    (https://personal.utdallas.edu/~sxb027100/dock/quaternion.html#)

    Check that the quaternion is a unit quaternion, then return its conjugate
    """
    assert len(qs.shape) == 2
    assert qs.shape[1] == 4

    # Check that the quaternions are valid
    norms = quaternion_norm(qs)
    if max(norms) > 1.01 or min(norms) < 0.99:
        raise RuntimeError("quaternion is not a unit quaternion")

    return quaternion_conjugate(qs)


@enforce_pt_np_input
def quaternion_product(qs_1: PT_NP_TYPE, qs_2: PT_NP_TYPE) -> PT_NP_TYPE:
    """TODO: document"""
    assert (len(qs_1.shape) == 2) and (len(qs_2.shape) == 2)
    assert (qs_1.shape[1] == 4) and (qs_2.shape[1] == 4)

    w1 = qs_1[:, 0]
    x1 = qs_1[:, 1]
    y1 = qs_1[:, 2]
    z1 = qs_1[:, 3]
    w2 = qs_2[:, 0]
    x2 = qs_2[:, 1]
    y2 = qs_2[:, 2]
    z2 = qs_2[:, 3]

    if isinstance(qs_1, np.ndarray):
        q = np.zeros(qs_1.shape)
    elif isinstance(qs_1, torch.Tensor):
        q = torch.zeros(qs_1.shape, device=qs_1.device, dtype=DEFAULT_TORCH_DTYPE)
    else:
        raise ValueError(f"qs_1 must be a numpy array or a torch tensor (got {type(qs_1)})")

    q[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return q


@enforce_pt_np_input
def quatmul(q1: PT_NP_TYPE, q2: PT_NP_TYPE) -> PT_NP_TYPE:
    """
    Given rows of quaternions q1 and q2, compute the Hamilton product q1 * q2
    """
    assert q1.shape[1] == 4, f"q1.shape[1] is {q1.shape[1]}, should be 4"
    assert q1.shape == q2.shape
    if isinstance(q1, torch.Tensor) and isinstance(q2, torch.Tensor):
        stacker = torch.vstack
    if isinstance(q1, np.ndarray) and isinstance(q2, np.ndarray):
        stacker = np.vstack

    w1, x1, y1, z1 = tuple(q1[:, i] for i in range(4))
    w2, x2, y2, z2 = tuple(q2[:, i] for i in range(4))

    return stacker(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    ).T


def axisangle_to_quat(axis: PT_NP_TYPE, angle: PT_NP_TYPE) -> PT_NP_TYPE:
    """
    Given rows of axis and angle, compute rows of quaternions.
    """
    assert axis.shape[1] == 3
    assert angle.shape[1] == 1
    assert axis.shape[0] == angle.shape[0]

    if isinstance(axis, torch.Tensor) and isinstance(angle, torch.Tensor):
        sin, cos, stacker = torch.sin, torch.cos, torch.hstack
    if isinstance(axis, np.ndarray) and isinstance(angle, np.ndarray):
        sin, cos, stacker = np.sin, np.cos, np.hstack

    return stacker((cos(angle / 2), sin(angle / 2) * normalize_vector(axis)))


# TODO: Benchmark speed when running this with numpy. Does it matter if its slow?


@enforce_pt_np_input
def geodesic_distance_between_quaternions(
    q1: PT_NP_TYPE, q2: PT_NP_TYPE, acos_epsilon: Optional[float] = None
) -> PT_NP_TYPE:
    """
    Given rows of quaternions q1 and q2, compute the geodesic distance between each
    """
    assert q1.shape[1] == 4, f"q1.shape[1] is {q1.shape[1]}, should be 4"
    assert len(q1.shape) == 2
    assert q1.shape == q2.shape
    # Note: Decreasing this value to 1e-8 greates NaN gradients for nearby quaternions.
    acos_clamp_epsilon = 1e-7
    if acos_epsilon is not None:
        acos_clamp_epsilon = acos_epsilon

    if isinstance(q1, np.ndarray):
        dot = np.clip(np.sum(q1 * q2, axis=1), -1, 1)
        # Note: Updated by @jstmn on Feb24 2023
        distance = 2 * np.arccos(np.clip(dot, -1 + acos_clamp_epsilon, 1 - acos_clamp_epsilon))
        # distance = 2 * np.arccos(dot)
        distance = np.abs(np.remainder(distance + np.pi, 2 * np.pi) - np.pi)
        assert distance.size == q1.shape[0], (
            f"Error, {distance.size} distance values calculated (np)- should be {q1.shape[0]} (distance.shape ="
            f" {distance.shape})"
        )
        return distance

    if isinstance(q1, torch.Tensor):
        dot = torch.clip(torch.sum(q1 * q2, dim=1), -1, 1)
        # Note: Updated by @jstmn on Feb24 2023
        distance = 2 * torch.acos(torch.clamp(dot, -1 + acos_clamp_epsilon, 1 - acos_clamp_epsilon))
        # distance = 2 * torch.acos(dot)
        distance = torch.abs(torch.remainder(distance + torch.pi, 2 * torch.pi) - torch.pi)
        assert distance.numel() == q1.shape[0], (
            f"Error, {distance.numel()} distance values calculated - should be {q1.shape[0]} (distance.shape ="
            f" {distance.shape})"
        )
        return distance


# ======================================================================================================================
# angle-axis conversions
#

# TODO: Consider reimplmenting


@enforce_pt_np_input
def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert 3d vector of axis-angle rotation to 3x3 rotation matrix

    Args:
        angle_axis (torch.Tensor): [N x 3] tensor of 3d vector of axis-angle rotations.

    Returns:
        torch.Tensor: [N x 3 x 3] tensor of 3x3 rotation matrices.
    """

    if not isinstance(angle_axis, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(angle_axis)}")

    if not angle_axis.shape[-1] == 3:
        raise ValueError(f"Input size must be a (*, 3) tensor. Got {angle_axis.shape}")

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        """A simple fix is to add the already previously defined eps to theta2 instead of to theta. Although that could
        result in theta being very small compared to eps, so I've included theta2+eps and theta+eps.
        """

        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0

        # With eps. fix
        theta = torch.sqrt(theta2 + eps)
        wxyz = angle_axis / (theta + eps)

        # Original code
        # theta = torch.sqrt(theta2)
        # wxyz = angle_axis / (theta + eps)

        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h
    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(3).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 3, 3).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4


# ======================================================================================================================
#  euler/rpy conversions
#


def rpy_tuple_to_rotation_matrix(
    rpy: Tuple[float, float, float], device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Convert a single rpy tuple to a single rotation matrix.

    Args:
        rpy (Tuple[float, float, float]): An orientation descibed by a tuple of roll, pitch, yaw.
        device (torch.device, optional): The device the returned tensor should live on. Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: [3 x 3] rotation matrix
    """
    rpy = np.array(rpy)
    sr, sp, sy = np.sin(rpy)
    cr, cp, cy = np.cos(rpy)

    Rx = torch.eye(3, dtype=DEFAULT_TORCH_DTYPE, device=device)
    Rx[1, 1] = cr  # TODO: wtf, why is this not using torch.cos?
    Rx[1, 2] = -sr
    Rx[2, 1] = sr
    Rx[2, 2] = cr

    Ry = torch.eye(3, dtype=DEFAULT_TORCH_DTYPE, device=device)
    Ry[0, 0] = cp
    Ry[0, 2] = sp
    Ry[2, 0] = -sp
    Ry[2, 2] = cp

    Rz = torch.eye(3, dtype=DEFAULT_TORCH_DTYPE, device=device)
    Rz[0, 0] = cy
    Rz[0, 1] = -sy
    Rz[1, 0] = sy
    Rz[1, 1] = cy

    R = Rz @ Ry @ Rx
    return R


def rpy_to_quat(rpy: Union[Tuple[float, float, float], torch.Tensor, np.ndarray], device: torch.device) -> torch.Tensor:
    """Convert RPY tuple (XYZ intrinsic) to a unit quaternion."""
    rpy = np.array(rpy)
    sr, sp, sy = np.sin(rpy / 2)
    cr, cp, cy = np.cos(rpy / 2)
    quat = torch.tensor(
        (
            +cr * cp * cy + sr * sp * sy,
            -cr * sp * sy + sr * cp * cy,
            +cr * sp * cy + sr * cp * sy,
            +cr * cp * sy - sr * sp * cy,
        ),
        dtype=DEFAULT_TORCH_DTYPE,
        device=device,
    )
    return quat / torch.linalg.norm(quat)


def quatvecrot(quat: torch.Tensor, vec: torch.Tensor):
    assert quat.shape[1] == 4
    assert vec.shape[1] == 3
    assert quat.shape[0] == vec.shape[0]
    purevec = torch.hstack((torch.zeros((vec.shape[0], 1), dtype=vec.dtype, device=vec.device), vec))
    return quatmul(quatmul(quat, purevec), quatconj(quat))[:, 1:]


# ======================================================================================================================
#  Axis angle conversions
#


def poseposemul(a_T_b: torch.Tensor, b_T_c: torch.Tensor):
    # qw, qx, qy, qz, x, y, z
    assert a_T_b.shape[1] == 7
    assert b_T_c.shape[1] == 7
    assert a_T_b.shape[0] == b_T_c.shape[0]
    a_t_b, a_R_b = a_T_b[:, :3], a_T_b[:, 3:]
    b_t_c, b_R_c = b_T_c[:, :3], b_T_c[:, 3:]

    a_R_c = quatmul(a_R_b, b_R_c)
    a_t_c = quatvecrot(a_R_b, b_t_c) + a_t_b

    return torch.hstack((a_t_c, a_R_c))


def single_axis_angle_to_rotation_matrix(
    axis: Tuple[float, float, float], ang: torch.Tensor, out_device: str
) -> torch.Tensor:
    """Convert a single axis vector, to a batch of rotation matrices. The axis vector is not batched, but the 'ang' is.
    axis: (3,) vector
    ang:  (batch_sz, 1) matrix

    Args:
        axis (Tuple[float, float, float]): The axis of rotation
        ang (torch.Tensor): [ batch x 1 ] tensor of rotation amounts
        device (str): The device the output tensor should live on

    Returns:
        torch.Tensor: [ batch x 3 x 3 ] batch of rotation matrices
    """
    angleaxis = torch.tensor(axis, device=out_device).unsqueeze(0).repeat(ang.shape[0], 1)
    ang = ang.view(-1, 1)
    angleaxis = angleaxis * ang
    R = angle_axis_to_rotation_matrix(angleaxis)
    return R
