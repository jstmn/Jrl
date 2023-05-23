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



def capsule_capsule_distance(c1: CapsuleGeometry, pose_c1: torch.Tensor, c2: CapsuleGeometry, pose_c2: torch.Tensor) -> float:
    """ Returns the minimum distance between any two points on the given capsules
    """
    assert pose_c1.shape == pose_c2.shape == (1, 7)
    c1_local_points = torch.tensor(
        [[
        [0, 0, c1.height/2],
        [0, 0, -c1.height/2]
        ]], device="cpu")
    c2_local_points = torch.tensor([
        [[0, 0, c2.height/2],
        [0, 0, -c2.height/2]]
    ], device="cpu")

    # local_points = 
    TODO: either use calculate_points_in_world_frame_from_local_frame_batch and figure out a good way to batch 
    c1/c2_local_points, or create calculate_points_in_world_frame_from_local_frame and call that twice. I think using 
    the batch function is a better solution, that way you won't need to add more unit tests

    segment_points = calculate_points_in_world_frame_from_local_frame_batch(torch.cat([pose_c1, pose_c2], dim=0), local_points)

    print("c1_segment_points:", c1_segment_points)
    print("c2_segment_points:\n", c2_segment_points)

    return 0.0