from dataclasses import dataclass
import torch

from jkinpylib.conversions import calculate_points_in_world_frame_from_local_frame_batch


@dataclass
class CapsuleGeometry:

    """Representation of a capsule geometry. height doesn't include the rounded section of the capsule. This means the
    total height is equal to `height + 2*radius`
    """

    radius: float
    height: float


def capsule_capsule_distance_batch(
    caps1: torch.Tensor, pose_c1: torch.Tensor, caps2: torch.Tensor, pose_c2: torch.Tensor
) -> float:
    """Returns the minimum distance between any two points on the given batch of capsules

    This function implemants the capsule-capsule minimum distance equation from the paper "Efficient Calculation of
    Minimum Distance Between Capsules and Its Use in Robotics" (https://hal.science/hal-02050431/document).

    Args:
        caps1 (torch.Tensor): [n x 2] tensor descibing a batch of capsules. Column 0 is radius, column 1 is height.
        pose_c1 (torch.Tensor): [n x 7] tensor describing the psoe of the caps1 capsules
        caps2 (torch.Tensor): [n x 2] tensor descibing a batch of capsules. Column 0 is radius, column 1 is height.
        pose_c2 (torch.Tensor): [n x 7] tensor describing the psoe of the caps1 capsules

    Returns:
        float: [n x 1] tensor with the minimum distance between each n capsules
    """
    n = caps1.shape[0]
    assert pose_c1.shape == pose_c2.shape == (n, 7)
    assert caps1.shape == pose_c2.shape == (n, 7)

    caps1_radius = caps1[:, 0]
    caps1_height = caps1[:, 1]
    caps2_radius = caps2[:, 0]
    caps2_height = caps2[:, 1]

    # Local points are at the top and bottom of capsule along the +z axis. They end at the center of the sphere on each
    # end
    c1_local_points = torch.zeros((n, 2, 3))  # this probably isn't right
    c1_local_points[:, 0, :] = caps1_height / 2
    c1_local_points[:, 1, :] = -caps1_height / 2

    c2_local_points = torch.zeros((n, 2, 3))

    # TODO: Use calculate_points_in_world_frame_from_local_frame_batch and figure out a good way to batch
    # caps1/c2_local_points. 'calculate_points_in_world_frame_from_local_frame_batch()' has unit tests already so should
    # be assumed to be correct.
    segment_points = calculate_points_in_world_frame_from_local_frame_batch(
        torch.cat([pose_c1, pose_c2], dim=0), local_points
    )
    return torch.zeros((n, 1))
