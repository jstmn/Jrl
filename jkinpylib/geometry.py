from dataclasses import dataclass
import torch

from jkinpylib.conversions import calculate_points_in_world_frame_from_local_frame_batch
import qpth


def capsule_capsule_distance_batch(
    caps1: torch.Tensor, T1: torch.Tensor, caps2: torch.Tensor, T2: torch.Tensor
) -> float:
    """Returns the minimum distance between any two points on the given batch of capsules

    This function implements the capsule-capsule minimum distance equation from the paper "Efficient Calculation of
    Minimum Distance Between Capsules and Its Use in Robotics"
    (https://hal.science/hal-02050431/document).

    Each capsule is defined by a radius and height.
    The height does not include the rounded ends of the capsule.
    The origin of the local frame is at the center of one sphere end of the
        capsule, and the capsule extends up the +z axis in its local frame.

    Args:
        caps1 (torch.Tensor): [n x 2] tensor descibing a batch of capsules. Column 0 is radius, column 1 is height.
        T1 (torch.Tensor): [n x 7] tensor (xyz + quat wxyz) describing the psoe of the caps1 capsules
        caps2 (torch.Tensor): [n x 2] tensor descibing a batch of capsules. Column 0 is radius, column 1 is height.
        T2 (torch.Tensor): [n x 7] tensor (xyz + quat wxyz) describing the psoe of the caps1 capsules

    Returns:
        float: [n x 1] tensor with the minimum distance between each n capsules
    """
    dtype = caps1.dtype
    device = caps1.device

    n = caps1.shape[0]
    assert T1.shape == T2.shape == (n, 7)
    assert caps1.shape == caps2.shape == (n, 2)

    r1, h1 = caps1[:, 0], caps1[:, 1]
    r2, h2 = caps2[:, 0], caps2[:, 1]

    # Local points are at the origin and top of capsule along the +z axis.
    c1_local_points = torch.zeros((n, 2, 3))
    c1_local_points[:, 1, 2] = h1
    c1_world = calculate_points_in_world_frame_from_local_frame_batch(T1, c1_local_points)

    c2_local_points = torch.zeros((n, 2, 3))
    c2_local_points[:, 1, 2] = h2
    c2_world = calculate_points_in_world_frame_from_local_frame_batch(T2, c2_local_points)

    p1 = c1_world[:, 0, :]
    s1 = c1_world[:, 1, :] - p1
    p2 = c2_world[:, 0, :]
    s2 = c2_world[:, 1, :] - p2

    A = torch.stack((s2, -s1), dim=2)
    y = (p2 - p1).unsqueeze(2)

    # Construct the QP
    Q = 2 * A.transpose(1, 2).bmm(A)
    # Semidefiniteness arises with parallel capsules
    Q = Q + 1e-6 * torch.eye(2, dtype=dtype, device=device).expand(n, -1, -1)
    p = 2 * A.transpose(1, 2).bmm(y).squeeze(2)

    # Inequality constraints
    G = torch.tensor(([[1, 0], [0, 1], [-1, 0], [0, -1]]), dtype=dtype, device=device)
    h = torch.tensor([1, 1, 0, 0], dtype=dtype, device=device)

    # Solve the QP
    e = torch.Tensor()  # Dummy equality constraint
    sol = qpth.qp.QPFunction(verbose=False)(Q, p, G, h, e, e)
    sol = sol.unsqueeze(2)

    dist = torch.norm(A.bmm(sol) + y, dim=1) - r1.unsqueeze(1) - r2.unsqueeze(1)

    return dist
