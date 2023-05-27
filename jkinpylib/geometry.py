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

    Each capsule is defined by two points in local frame, and a radius.
    The memory layout is [nx7]: [x1, y1, z1, x2, y2, z2, r1].

    Args:
        caps1 (torch.Tensor): [n x 7] tensor descibing a batch of capsules.
        T1 (torch.Tensor): [n x 4 x 4] tensor (xyz + quat wxyz) describing the pose of the caps1 capsules
        caps2 (torch.Tensor): [n x 7] tensor descibing a batch of capsules.
        T2 (torch.Tensor): [n x 4 x 4] tensor (xyz + quat wxyz) describing the pose of the caps1 capsules

    Returns:
        float: [n x 1] tensor with the minimum distance between each n capsules
    """
    dtype = caps1.dtype
    device = caps1.device
    print("caps1.device", caps1.device)
    print("T1.device", T1.device)

    n = caps1.shape[0]
    assert T1.shape == T2.shape == (n, 4, 4)
    assert caps1.shape == caps2.shape == (n, 7)

    # Local points are at the origin and top of capsule along the +z axis.
    r1 = caps1[:, 6]
    T1[:, :3, :3]
    caps1[:, 0:3].unsqueeze(2)
    T1[:, :3, 3]
    c1_world1 = T1[:, :3, :3].bmm(caps1[:, 0:3].unsqueeze(2)).squeeze(2) + T1[:, :3, 3]
    c1_world2 = T1[:, :3, :3].bmm(caps1[:, 3:6].unsqueeze(2)).squeeze(2) + T1[:, :3, 3]

    r2 = caps2[:, 6]
    c2_world1 = T2[:, :3, :3].bmm(caps2[:, 0:3].unsqueeze(2)).squeeze(2) + T2[:, :3, 3]
    c2_world2 = T2[:, :3, :3].bmm(caps2[:, 3:6].unsqueeze(2)).squeeze(2) + T2[:, :3, 3]

    p1 = c1_world1
    s1 = c1_world2 - p1
    p2 = c2_world1
    s2 = c2_world2 - p2

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
