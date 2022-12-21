""" This file contains conversion functions between rotation representations, as well as implementations for various 
mathematical operations. 

A couple notes:
    1. Quaternions are assumed to be in w,x,y,z format 
    2. RPY format is a rotation about x, y, z axes in that order
    3. Functions that end with '_np' accept numpy arrays, those that end with '_pt' accept torch tensors
    4. All functions except those that end with '_single' accept batches of inputs.  
"""

import torch
import numpy as np

from jkinpylib import config


def geodesic_distance_between_rotation_matrices(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """Calculate the geodesic distance between rotation matrices

    Args:
        m1 (torch.Tensor): [batch x 3 x 3] rotation matrix
        m2 (torch.Tensor): [batch x 3 x 3] rotation matrix

    Returns:
        torch.Tensor: [batch] rotational differences between m1, m2. Between 0 and pi for each element
    """
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(config.device)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(config.device)) * -1)
    theta = torch.acos(cos)
    return theta


# TODO: Reimplement. There must be a more efficient way
def geodesic_distance_between_quaternions_np(q1: np.array, q2: np.array) -> np.array:
    """Given rows of quaternions q1 and q2, compute the geodesic distance between each

    Returns:
        np.array: [batch] rotational differences between q1, q2. Between 0 and pi for each comparison
    """
    assert len(q1.shape) == 2
    assert len(q2.shape) == 2
    assert q1.shape[0] == q2.shape[0]
    assert q1.shape[1] == q2.shape[1]

    q1_R9 = quaternion_to_rotation_matrix_pt(torch.tensor(q1, device=config.device))
    q2_R9 = quaternion_to_rotation_matrix_pt(torch.tensor(q2, device=config.device))
    return geodesic_distance_between_rotation_matrices(q1_R9, q2_R9).cpu().data.numpy()


def normalize_vector(v: torch.Tensor) -> torch.Tensor:
    """TODO: document

    Args:
        v (torch.Tensor): [batch x n]

    Returns:
        torch.Tensor: [batch x n]
    """
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(config.device)))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


# ======================================================================================================================
# quaternion conversions
#


def quaternion_to_rotation_matrix_pt(quaternion: torch.Tensor) -> torch.Tensor:
    """TODO: document

    Args:
        quaternion (torch.Tensor): [batch x 4]

    Returns:
        torch.Tensor: [batch x 3 x 3]
    """
    batch = quaternion.shape[0]

    quat = normalize_vector(quaternion).contiguous()

    qw = quat[..., 0].contiguous().view(batch, 1)
    qx = quat[..., 1].contiguous().view(batch, 1)
    qy = quat[..., 2].contiguous().view(batch, 1)
    qz = quat[..., 3].contiguous().view(batch, 1)

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

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3
    return matrix


def quaternion_to_rpy_np_single(q: np.array):
    """Return roll pitch yaw"""
    roll = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    pitch = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([roll, pitch, yaw])


def quaternion_to_rpy_pt(q: torch.Tensor, device: str) -> torch.Tensor:
    assert len(q.shape) == 2
    assert q.shape[1] == 4
    batch = q.shape[0]
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]
    p = torch.asin(2 * (q0 * q2 - q3 * q1))
    rpy = torch.zeros((batch, 3), device=device)
    rpy[:, 0] = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    rpy[:, 1] = p
    rpy[:, 2] = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return rpy


def quaternion_to_rpy_np(q: np.ndarray) -> np.ndarray:
    assert len(q.shape) == 2
    assert q.shape[1] == 4

    n = q.shape[0]
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]
    p = np.arcsin(2 * (q0 * q2 - q3 * q1))
    rpy = np.zeros((n, 3))
    rpy[:, 0] = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    rpy[:, 1] = p
    rpy[:, 2] = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return rpy


def quaternion_conjugate_np(qs: np.ndarray) -> np.ndarray:
    """ """
    assert len(qs.shape) == 2
    assert qs.shape[1] == 4
    q_conj = np.zeros(qs.shape)
    q_conj[:, 0] = qs[:, 0]
    q_conj[:, 1] = -qs[:, 1]
    q_conj[:, 2] = -qs[:, 2]
    q_conj[:, 3] = -qs[:, 3]
    return q_conj


def quaternion_norm_np(qs: np.ndarray) -> np.ndarray:
    """ """
    assert len(qs.shape) == 2
    assert qs.shape[1] == 4
    return np.linalg.norm(qs, axis=1)


def quaternion_inverse_np(qs: np.ndarray) -> np.ndarray:
    """Per "CS184: Using Quaternions to Represent Rotation": The inverse of a unit quaternion is its conjugate, q-1=q'
    (https://personal.utdallas.edu/~sxb027100/dock/quaternion.html#)

    Check that the quaternion is a unit quaternion, then return its conjugate
    """
    assert len(qs.shape) == 2
    assert qs.shape[1] == 4
    norms = quaternion_norm_np(qs)
    np.testing.assert_allclose(norms, np.ones(norms.shape), atol=1e-4)
    return quaternion_conjugate_np(qs)


def quaternion_multiply_np(qs_1: np.ndarray, qs_2: np.ndarray) -> np.ndarray:
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

    q = np.zeros(qs_1.shape)
    q[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return q


# ======================================================================================================================
# __ conversion
#

# TODO: Consider reimplmenting
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


def rpy_to_rotation_matrix(rpy, device: str):
    """_summary_

    Args:
        rpy (_type_): _description_
        device (str): _description_

    Returns:
        _type_: _description_
    """

    r = rpy[0]
    p = rpy[1]
    y = rpy[2]

    Rx = torch.eye(3, device=device)
    Rx[1, 1] = np.cos(r)
    Rx[1, 2] = -np.sin(r)
    Rx[2, 1] = np.sin(r)
    Rx[2, 2] = np.cos(r)

    Ry = torch.eye(3, device=device)
    Ry[0, 0] = np.cos(p)
    Ry[0, 2] = np.sin(p)
    Ry[2, 0] = -np.sin(p)
    Ry[2, 2] = np.cos(p)

    Rz = torch.eye(3, device=device)
    Rz[0, 0] = np.cos(y)
    Rz[0, 1] = -np.sin(y)
    Rz[1, 0] = np.sin(y)
    Rz[1, 1] = np.cos(y)

    R = Rz.mm(Ry.mm(Rx))
    return R


def axis_angle_to_rotation_matrix(axis, ang: torch.tensor, device: str):
    """
    axis: (3,) vector
    ang:  (batch_sz, 1) matrix
    """
    angleaxis = torch.tensor(axis, device=device).unsqueeze(0).repeat(ang.shape[0], 1)
    ang = ang.view(-1, 1)
    angleaxis = angleaxis * ang
    R = angle_axis_to_rotation_matrix(angleaxis)
    return R
