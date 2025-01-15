"""This file contains conversion functions between rotation representations, as well as implementations for various
mathematical operations.

A couple notes:
    1. Quaternions are assumed to be in w,x,y,z format
    2. RPY format is a rotation about x, y, z axes in that order
"""

from typing import Tuple, Optional

import torch
import numpy as np

from jrl.config import DEFAULT_TORCH_DTYPE, ACCELERATOR_AVAILABLE

_TORCH_EPS_CPU = torch.tensor(1e-8, dtype=DEFAULT_TORCH_DTYPE, device="cpu")
_TORCH_EPS_CUDA = (
    torch.tensor(
        1e-8,
        dtype=DEFAULT_TORCH_DTYPE,
        device="mps" if torch.backends.mps.is_available() else "cuda",
    )
    if ACCELERATOR_AVAILABLE
    else None
)


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

    def solve(self, x0=None, trace=False, iterlimit=None):
        if x0 is None:
            x = torch.zeros((self.nbatch, self.dim, 1), dtype=torch.float32, device=self.Q.device)
        else:
            x = x0
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
                # Qs = self.Q[~converged.flatten()]
                # ps = self.p[~converged.flatten()]
                # Gs = self.G[~converged.flatten()]
                # hs = self.h[~converged.flatten()]
                # xs = x[~converged.flatten()]
                # for i in range(torch.sum(~converged)):
                #     print(f"Qs[{i}]:\n{Qs[i]}")
                #     print(f"ps[{i}]:\n{ps[i]}")
                #     print(f"Gs[{i}]:\n{Gs[i]}")
                #     print(f"hs[{i}]:\n{hs[i]}")
                #     print(f"xs[{i}]:\n{xs[i]}")
                raise RuntimeError(
                    f"Failed to converge in {iterlimit} iterations\n\n{torch.sum(~converged).item()} out of"
                    f" {self.nbatch} not converged"
                )

        if trace:
            return x.squeeze(2), trace

        return x.squeeze(2)


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


def calculate_points_in_world_frame_from_local_frame_batch(
    world__T__local_frame: torch.Tensor, points_in_local_frame: torch.Tensor
) -> torch.Tensor:
    """Calculate the position of `points_in_local_frame` in world frame. The transform from 'world' frame to
    'local_frame' is given by 'world__T__local_frame'

    Args:
        world__T__local_frame (torch.Tensor): [n x 7] a single pose describing the transform between 'world' frame and
                                                'local_frame'
        points_in_local_frame (torch.Tensor): [n x m x 3] tensor of positions represented in 'local_frame'. There may be
                                                an arbitrary number of points (m) provided

    Returns:
        torch.Tensor: [n x m x 3] tensor of the positions represented in world frame
    """
    n, m, _ = points_in_local_frame.shape

    # rotate then translate
    points_in_world_frame = []
    for i in range(m):
        points_quat = torch.cat(
            [
                torch.zeros((n, 1), device=world__T__local_frame.device),
                points_in_local_frame[:, i],
            ],
            dim=1,
        )
        pts_rotated = quatmul(
            quatmul(world__T__local_frame[:, 3:7], points_quat),
            quatconj(world__T__local_frame[:, 3:7]),
        )[:, 1:]
        pts = pts_rotated + world__T__local_frame[:, 0:3]
        points_in_world_frame.append(pts[:, None, :])

    return torch.cat(points_in_world_frame, dim=1)


def angular_subtraction(angles_1: torch.Tensor, angles_2: torch.Tensor) -> torch.Tensor:
    """Subtraction of elements on a circle. Specifically, implementation of the subtraction operator for elements in the
    '1-sphere' (see https://en.wikipedia.org/wiki/N-sphere).

    Args:
        angles_1 (torch.Tensor): [batch x d] tensor of angles
        angles_2 (torch.Tensor): [batch x d] tensor of angles

    Returns:
        torch.Tensor: [batch x d] tensor of angles, calculated as angles_1 - angles_2
    """
    assert angles_1.shape == angles_2.shape
    d_angles = angles_1 - angles_2
    return torch.remainder(d_angles + torch.pi, 2 * torch.pi) - torch.pi


# ======================================================================================================================
#  Rotation matrix functions
#


# Borrowed from RoMa library (https://github.com/naver/roma/tree/master)
def rotmat_to_unitquat(R):
    """
    Converts rotation matrix to unit quaternion representation.

    Args:
        R (...x3x3 tensor): batch of rotation matrices.
    Returns:
        batch of unit quaternions (...x4 tensor, XYZW convention).
    """

    def unflatten_batch_dims(tensor, batch_shape):
        """
        :meta private:
        Revert flattening of a tensor.
        """
        # Note: alternative to tensor.unflatten(dim=0, sizes=batch_shape) that was not supported by PyTorch 1.6.0.
        return tensor.reshape(batch_shape + tensor.shape[1:]) if len(batch_shape) > 0 else tensor.squeeze(0)

    def flatten_batch_dims(tensor, end_dim):
        """
        :meta private:
        Utility function: flatten multiple batch dimensions into a single one, or add a batch dimension if there is none.
        """
        batch_shape = tensor.shape[: end_dim + 1]
        flattened = tensor.flatten(end_dim=end_dim) if len(batch_shape) > 0 else tensor.unsqueeze(0)
        return flattened, batch_shape

    matrix, batch_shape = flatten_batch_dims(R, end_dim=-3)
    num_rotations, D1, D2 = matrix.shape
    assert (D1, D2) == (3, 3), "Input should be a Bx3x3 tensor."

    # Adapted from SciPy:
    # https://github.com/scipy/scipy/blob/7cb3d751756907238996502b92709dc45e1c6596/scipy/spatial/transform/rotation.py#L480

    decision_matrix = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)
    decision_matrix[:, :3] = matrix.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)

    ind = torch.nonzero(choices != 3, as_tuple=True)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat = quat / torch.norm(quat, dim=1)[:, None]
    return unflatten_batch_dims(quat, batch_shape)


def rotation_matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    """Converts a batch of rotation matrices to quaternions

    Args:
        m (torch.Tensor): [batch x 3 x 3] tensor of rotation matrices

    Returns:
        torch.Tensor: [batch x 4] tensor of quaternions
    """
    quat = rotmat_to_unitquat(m)
    return quaternion_xyzw_to_wxyz(quat)


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


def quaternion_xyzw_to_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert a batch of quaternions from xyzw to wxyz format

    Args:
        quaternion (torch.Tensor): A [batch x 4] tensor or numpy array of quaternions
    """
    return torch.cat([quaternion[:, 3:4], quaternion[:, 0:3]], dim=1)


def quaternion_to_rotation_matrix(quaternion: torch.Tensor):
    """_summary_

    Args:
        quaternion (torch.Tensor): [batch x 4] tensor of quaternions

    Returns:
        _type_: _description_
    """
    batch = quaternion.shape[0]

    # TODO: Should we normalize the quaternion here? Maybe just verify its almost normalized instead?
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


# TODO: implement, test, compare runtime for quaternion_difference_to_rpy()
def quaternion_difference_to_rpy(q_target: torch.Tensor, q_current: torch.Tensor) -> torch.Tensor:
    del q_target, q_current
    raise NotImplementedError()
    # Default implementation
    # current_inv = quaternion_inverse(q_current)
    # rotation_error_quat = quaternion_product(q_target, current_inv)
    # rotation_error_rpy = quaternion_to_rpy(rotation_error_quat)
    # return rotation_error_rpy
    return


def quaternion_to_rpy(q: torch.Tensor) -> torch.Tensor:
    """Convert a batch of quaternions to roll-pitch-yaw angles"""
    assert len(q.shape) == 2
    assert q.shape[1] == 4

    n = q.shape[0]
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    rpy = torch.zeros((n, 3), device=q.device, dtype=DEFAULT_TORCH_DTYPE)
    p = torch.arcsin(2 * (q0 * q2 - q3 * q1))
    atan2 = torch.arctan2

    # handle singularity
    rpy[:, 0] = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    rpy[:, 1] = p
    rpy[:, 2] = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    return rpy


def quaternion_conjugate(qs: torch.Tensor) -> torch.Tensor:
    """TODO: document"""
    assert len(qs.shape) == 2
    assert qs.shape[1] == 4

    q_conj = torch.zeros(qs.shape, device=qs.device, dtype=DEFAULT_TORCH_DTYPE)
    q_conj[:, 0] = qs[:, 0]
    q_conj[:, 1] = -qs[:, 1]
    q_conj[:, 2] = -qs[:, 2]
    q_conj[:, 3] = -qs[:, 3]
    return q_conj


def quatconj(q: torch.Tensor) -> torch.Tensor:
    """
    Given rows of quaternions q, compute quaternion conjugate

    Author: dmillard
    """
    w, x, y, z = tuple(q[:, i] for i in range(4))
    return torch.vstack((w, -x, -y, -z)).T


def quaternion_norm(qs: torch.Tensor) -> torch.Tensor:
    """TODO: document"""
    assert len(qs.shape) == 2
    assert qs.shape[1] == 4
    return torch.norm(qs, dim=1)


def quaternion_inverse(qs: torch.Tensor) -> torch.Tensor:
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


def quaternion_product(qs_1: torch.Tensor, qs_2: torch.Tensor) -> torch.Tensor:
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

    q = torch.zeros(qs_1.shape, device=qs_1.device, dtype=DEFAULT_TORCH_DTYPE)
    q[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return q


def quatmul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Given rows of quaternions q1 and q2, compute the Hamilton product q1 * q2
    """
    assert q1.shape[1] == 4, f"q1.shape[1] is {q1.shape[1]}, should be 4"
    assert q1.shape == q2.shape

    w1, x1, y1, z1 = tuple(q1[:, i] for i in range(4))
    w2, x2, y2, z2 = tuple(q2[:, i] for i in range(4))

    return torch.vstack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )).T


def geodesic_distance_between_quaternions(
    q1: torch.Tensor, q2: torch.Tensor, acos_epsilon: Optional[float] = None
) -> torch.Tensor:
    """
    Given rows of quaternions q1 and q2, compute the geodesic distance between each
    """
    # Note: Decreasing this value to 1e-8 greates NaN gradients for nearby quaternions.
    acos_clamp_epsilon = 1e-7
    if acos_epsilon is not None:
        acos_clamp_epsilon = acos_epsilon

    dot = torch.clip(torch.sum(q1 * q2, dim=1), -1, 1)
    distance = 2 * torch.acos(torch.clamp(dot, -1 + acos_clamp_epsilon, 1 - acos_clamp_epsilon))
    distance = torch.abs(torch.remainder(distance + torch.pi, 2 * torch.pi) - torch.pi)  # TODO: do we need this?
    return distance


# ======================================================================================================================
# angle-axis conversions
#

# TODO: Consider reimplmenting


def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert 3d vector of axis-angle rotation to 3x3 rotation matrix

    Args:
        angle_axis (torch.Tensor): [N x 3] tensor of 3d vector of axis-angle rotations.

    Returns:
        torch.Tensor: [N x 3 x 3] tensor of 3x3 rotation matrices.
    """

    assert isinstance(angle_axis, torch.Tensor), f"Input type is not a torch.Tensor. Got {type(angle_axis)}"
    assert angle_axis.shape[-1] == 3, f"Input size must be a (*, 3) tensor. Got {angle_axis.shape}"

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
    r = rpy[0]
    p = rpy[1]
    y = rpy[2]

    Rx = torch.eye(3, dtype=DEFAULT_TORCH_DTYPE, device=device)
    Rx[1, 1] = np.cos(r)  # TODO: wtf, why is this not using torch.cos?
    Rx[1, 2] = -np.sin(r)
    Rx[2, 1] = np.sin(r)
    Rx[2, 2] = np.cos(r)

    Ry = torch.eye(3, dtype=DEFAULT_TORCH_DTYPE, device=device)
    Ry[0, 0] = np.cos(p)
    Ry[0, 2] = np.sin(p)
    Ry[2, 0] = -np.sin(p)
    Ry[2, 2] = np.cos(p)

    Rz = torch.eye(3, dtype=DEFAULT_TORCH_DTYPE, device=device)
    Rz[0, 0] = np.cos(y)
    Rz[0, 1] = -np.sin(y)
    Rz[1, 0] = np.sin(y)
    Rz[1, 1] = np.cos(y)

    R = Rz.mm(Ry.mm(Rx))
    return R


# ======================================================================================================================
#  Axis angle conversions
#


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Convert a vector to a skew symmetric matrix

    Args:
        v (torch.Tensor): [3] vector

    Returns:
        torch.Tensor: [3 x 3] skew symmetric matrix
    """

    skew = torch.zeros(3, 3, dtype=v.dtype, device=v.device)
    skew[0, 1] = -v[2]
    skew[0, 2] = v[1]
    skew[1, 2] = -v[0]
    skew[1, 0] = v[2]
    skew[2, 0] = -v[1]
    skew[2, 1] = v[0]
    return skew


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

    angleaxis = torch.tensor(axis, device=out_device)
    angleaxis_skew = skew_symmetric(angleaxis)
    angleaxis_skew_sq = angleaxis_skew.mm(angleaxis_skew)

    return (
        torch.eye(3, dtype=ang.dtype, device=ang.device).unsqueeze(0)
        + torch.sin(ang).view(-1, 1, 1) * angleaxis_skew.unsqueeze(0)
        + (1 - torch.cos(ang).unsqueeze(1)).view(-1, 1, 1) * angleaxis_skew_sq.unsqueeze(0)
    )
