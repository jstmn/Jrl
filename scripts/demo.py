from jrl.robots import Panda
from jrl.utils import set_seed, make_text_green_or_red
from jrl.math_utils import rpy_tuple_to_rotation_matrix, geodesic_distance_between_quaternions
import torch

set_seed()


def pose_errors_cm_deg(poses_1: torch.Tensor, poses_2: torch.Tensor):
    """Return the positional and rotational angular error between two batch of poses."""
    positional_errors = torch.norm(poses_1[:, 0:3] - poses_2[:, 0:3], dim=1)
    rotational_errors = geodesic_distance_between_quaternions(poses_1[:, 3 : 3 + 4], poses_2[:, 3 : 3 + 4])
    return 100 * positional_errors, torch.rad2deg(rotational_errors)


def assert_poses_almost_equal(poses_1, poses_2):
    pos_errors_cm, rot_errors_deg = pose_errors_cm_deg(poses_1, poses_2)
    assert (pos_errors_cm.max().item() < 0.01) and (rot_errors_deg.max().item() < 0.1)
    print(make_text_green_or_red("poses are nearly equal", True))


robot = Panda()
joint_angles, poses = robot.sample_joint_angles_and_poses(
    n=5, return_torch=True
)  # sample 5 random joint angles and matching poses

# Run forward-kinematics
poses_fk = robot.forward_kinematics_batch(joint_angles)
assert_poses_almost_equal(poses, poses_fk)

# Run inverse-kinematics
ik_sols = joint_angles + 0.1 * torch.randn_like(joint_angles)
for i in range(5):
    ik_sols = robot.inverse_kinematics_single_step_levenburg_marquardt(poses, ik_sols)
assert_poses_almost_equal(poses, robot.forward_kinematics_batch(ik_sols))

# Check for robot-robot collisions
self_coll_distances = robot.self_collision_distances_batch(joint_angles)
print(f"configs self-colliding:", torch.min(self_coll_distances, dim=1)[0] < 0.0)

# Check for robot-environment collisions
cuboid = torch.tensor(
    [-0.1, -0.5, -0.5, 0.1, 0.5, 0.5]
)  # [-size_x/2, -size_y/2, -size_z/2, size_x/2, size_y/2, size_z/2]
Tcuboid = torch.zeros((4, 4))
Tcuboid[0, 3] = 0.25
Tcuboid[1, 3] = 0.25
Tcuboid[2, 3] = 0.25  # cuboid is centered at x=y=z=0.25
Tcuboid[:3, :3] = rpy_tuple_to_rotation_matrix((0.0, 0.0, 0.0))
cuboid_to_link_dists = robot.env_collision_distances_batch(joint_angles, cuboid, Tcuboid)
print(f"configs env-colliding:", torch.min(cuboid_to_link_dists, dim=1)[0] < 0.0)
