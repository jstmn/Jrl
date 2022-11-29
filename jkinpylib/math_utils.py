from jkinpylib import config

import torch
import numpy as np


def geodesic_distance(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
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


def geodesic_distance_between_quaternions(q1: np.array, q2: np.array) -> np.array:
    """Given rows of quaternions q1 and q2, compute the geodesic distance between each

    Args:
        q1 (np.array): _description_
        q2 (np.array): _description_

    Returns:
        np.array: _description_
    """

    assert len(q1.shape) == 2
    assert len(q2.shape) == 2
    assert q1.shape[0] == q2.shape[0]
    assert q1.shape[1] == q2.shape[1]

    q1_R9 = rotation_matrix_from_quaternion(torch.Tensor(q1).to(config.device))
    q2_R9 = rotation_matrix_from_quaternion(torch.Tensor(q2).to(config.device))
    return geodesic_distance(q1_R9, q2_R9).cpu().data.numpy()


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


def rotation_matrix_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
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


def quaternion_to_rpy(q: np.array):
    """Return roll pitch yaw"""
    roll = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    pitch = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([roll, pitch, yaw])


def quaternion_to_rpy_batch(q: torch.Tensor, device: str) -> torch.Tensor:
    assert len(q.shape) == 2
    assert q.shape[1] == 4

    batch = q.shape[0]
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    p = torch.asin(2 * (q0 * q2 - q3 * q1))

    rpy = torch.zeros((batch, 3)).to(device)
    rpy[:, 0] = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    rpy[:, 1] = p
    rpy[:, 2] = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return rpy


# TODO: Consider reimplmenting
def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert 3d vector of axis-angle rotation to 3x3 rotation matrix

    Args:
        angle_axis (torch.Tensor): [N x 3] tensor of 3d vector of axis-angle rotations.

    Returns:
        torch.Tensor: [N x 3 x 3] tensor of 3x3 rotation matrices.
    """

    if not isinstance(angle_axis, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(angle_axis)))

    if not angle_axis.shape[-1] == 3:
        raise ValueError("Input size must be a (*, 3) tensor. Got {}".format(angle_axis.shape))

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        """A simple fix is to add the already previously defined eps to theta2 instead of to theta. Although that could result in theta being very small compared to eps, so I've included theta2+eps and theta+eps.
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


def R_from_rpy_batch(rpy, device: str):
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

    Rx = torch.eye(3).to(device)
    Rx[1, 1] = np.cos(r)
    Rx[1, 2] = -np.sin(r)
    Rx[2, 1] = np.sin(r)
    Rx[2, 2] = np.cos(r)

    Ry = torch.eye(3).to(device)
    Ry[0, 0] = np.cos(p)
    Ry[0, 2] = np.sin(p)
    Ry[2, 0] = -np.sin(p)
    Ry[2, 2] = np.cos(p)

    Rz = torch.eye(3).to(device)
    Rz[0, 0] = np.cos(y)
    Rz[0, 1] = -np.sin(y)
    Rz[1, 0] = np.sin(y)
    Rz[1, 1] = np.cos(y)

    R = Rz.mm(Ry.mm(Rx))
    return R


def R_from_axis_angle(axis, ang: torch.tensor, device: str):
    """
    axis: (3,) vector
    ang:  (batch_sz, 1) matrix
    """
    angleaxis = torch.tensor(axis).unsqueeze(0).repeat(ang.shape[0], 1).to(device)
    ang = ang.view(-1, 1)
    angleaxis = angleaxis * ang
    R = angle_axis_to_rotation_matrix(angleaxis)
    return R
