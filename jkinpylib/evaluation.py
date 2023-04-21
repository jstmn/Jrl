from typing import Tuple, List
from time import time

import numpy as np
import torch

from jkinpylib.robot import Robot
from jkinpylib.robots import Panda
from jkinpylib.conversions import geodesic_distance_between_quaternions, enforce_pt_np_input, PT_NP_TYPE
from jkinpylib.config import DEVICE


DEVICE_TEMP = "cuda"

""" Description of 'SOLUTION_EVALUATION_RESULT_TYPE':
- torch.Tensor: [n] tensor of positional errors of the IK solutions. The error is the L2 norm of the realized poses of
                    the solutions and the target pose.
- torch.Tensor: [n] tensor of angular errors of the IK solutions. The error is the angular geodesic distance between the
                    realized orientation of the robot's end effector from the IK solutions and the targe orientation.
- torch.Tensor: [n] tensor of bools indicating whether each IK solutions has exceeded the robots joint limits.
- torch.Tensor: [n] tensor of bools indicating whether each IK solutions is self colliding.
- float: Runtime
"""
SOLUTION_EVALUATION_RESULT_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]


_DEFAULT_MAX_ALLOWABLE_L2_ERR = 5e-4
_DEFAULT_MAX_ALLOWABLE_ANG_ERR = 3.14 / 180 * 0.06


def assert_pose_positions_almost_equal(
    endpoints1: np.array,
    endpoints2: np.array,
    source_1: str = "",
    source_2: str = "",
    threshold: float = _DEFAULT_MAX_ALLOWABLE_L2_ERR,
):
    """Check that the position of each pose is nearly the same"""
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
    assert endpoints1.shape[1] == 7
    assert endpoints2.shape[1] == 7
    assert endpoints1.shape[0] == endpoints2.shape[0]
    errors = geodesic_distance_between_quaternions(endpoints1[:, 3 : 3 + 4], endpoints2[:, 3 : 3 + 4])
    for i in range(errors.shape[0]):
        assert errors[i] < threshold, (
            f"Rotation of poses '{source_1}({endpoints1[i, :]})', '{source_2} ({endpoints2[i, :]})' are not equal"
            f" (error={errors[i]})"
        )


def _get_target_pose_batch(target_pose: PT_NP_TYPE, n_solutions: int) -> torch.Tensor:
    """Return a batch of target poses. If the target_pose is a single pose, then it will be tiled to match the number of
    solutions. If the target_pose is a batch of poses, then it will be returned as is.

    Args:
        target_pose (PT_NP_TYPE): [7] or [n x 7] the target pose(s) to calculate errors for
        n_solutions (int): The number of solutions we are calculating errors for
    """
    if target_pose.shape[0] == 7:
        if isinstance(target_pose, torch.Tensor):
            return target_pose.repeat(n_solutions, 1)
        return np.tile(target_pose, (n_solutions, 1))
    return target_pose


@enforce_pt_np_input
def pose_errors(poses_1: PT_NP_TYPE, poses_2: PT_NP_TYPE) -> Tuple[PT_NP_TYPE, PT_NP_TYPE]:
    """Return the positional and rotational angular error between two batch of poses."""
    assert poses_1.shape == poses_2.shape

    if isinstance(poses_1, torch.Tensor):
        l2_errors = torch.norm(poses_1[:, 0:3] - poses_2[:, 0:3], dim=1)
    else:
        l2_errors = np.linalg.norm(poses_1[:, 0:3] - poses_2[:, 0:3], axis=1)
    angular_errors = geodesic_distance_between_quaternions(poses_1[:, 3 : 3 + 4], poses_2[:, 3 : 3 + 4])
    assert l2_errors.shape == angular_errors.shape
    return l2_errors, angular_errors


@enforce_pt_np_input
def pose_errors_cm_deg(poses_1: PT_NP_TYPE, poses_2: PT_NP_TYPE) -> Tuple[PT_NP_TYPE, PT_NP_TYPE]:
    """Return the positional and rotational angular error between two batch of poses in cm and degrees"""
    assert poses_1.shape == poses_2.shape
    l2_errors, angular_errors = pose_errors(poses_1, poses_2)

    if isinstance(poses_1, torch.Tensor):
        return 100 * l2_errors, torch.rad2deg(angular_errors)
    return 100 * l2_errors, np.rad2deg(angular_errors)


def solution_pose_errors(
    robot: Robot, solutions: torch.Tensor, target_poses: PT_NP_TYPE
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the L2 and angular errors of calculated ik solutions for a given target_pose. Note: this function expects
    multiple solutions but only a single target_pose. All of the solutions are assumed to be for the given target_pose

    Args:
        robot (Robot): The Robot which contains the FK function we will use
        solutions (Union[torch.Tensor, np.ndarray]): [n x 7] IK solutions for the given target pose
        target_pose (np.ndarray): [7] the target pose the IK solutions were generated for

    Returns:
        Tuple[np.ndarray, np.ndarray]: The L2, and angular (rad) errors of IK solutions for the given target_pose
    """
    assert isinstance(
        target_poses, (np.ndarray, torch.Tensor)
    ), f"target_poses must be a torch.Tensor or np.ndarray (got {type(target_poses)})"
    assert isinstance(solutions, torch.Tensor), f"solutions must be a torch.Tensor (got {type(solutions)})"
    n_solutions = solutions.shape[0]
    if n_solutions >= 1000:
        print("Heads up: It may be faster to run solution_pose_errors() with pytorch directly on the cpu/gpu")

    if isinstance(target_poses, torch.Tensor):
        target_poses = target_poses.detach().cpu().numpy()
    target_poses = _get_target_pose_batch(target_poses, solutions.shape[0])

    ee_pose_ikflow = robot.forward_kinematics(solutions[:, 0 : robot.n_dofs].detach().cpu().numpy())
    rot_output = ee_pose_ikflow[:, 3:]

    # Positional Error
    l2_errors = np.linalg.norm(ee_pose_ikflow[:, 0:3] - target_poses[:, 0:3], axis=1)
    rot_target = target_poses[:, 3:]
    assert rot_target.shape == rot_output.shape

    # Surprisingly, this is almost always faster to calculate on the gpu than on the cpu. I would expect the opposite
    # for low number of solutions (< 200).
    q_target_pt = torch.tensor(rot_target, device=DEVICE, dtype=torch.float32)
    q_current_pt = torch.tensor(rot_output, device=DEVICE, dtype=torch.float32)
    ang_errors = geodesic_distance_between_quaternions(q_target_pt, q_current_pt).detach().cpu().numpy()
    return l2_errors, ang_errors


def calculate_joint_limits_exceeded(configs: torch.Tensor, joint_limits: List[Tuple[float, float]]) -> torch.Tensor:
    """Calculate if the given configs have exceeded the specified joint limits

    Args:
        configs (torch.Tensor): [batch x n_dofs] tensor of robot configurations
        joint_limits (List[Tuple[float, float]]): The joint limits for the robot. Should be a list of tuples, where each
                                                    tuple contains (lower, upper).
    Returns:
        torch.Tensor: [batch] tensor of bools indicating if the given configs have exceeded the specified joint limits
    """
    toolarge = configs > torch.tensor([x[1] for x in joint_limits], dtype=torch.float32, device=configs.device)
    toosmall = configs < torch.tensor([x[0] for x in joint_limits], dtype=torch.float32, device=configs.device)
    return torch.logical_or(toolarge, toosmall).any(dim=1)


def calculate_self_collisions(robot: Robot, configs: torch.Tensor) -> torch.Tensor:
    """Calculate if the given configs will cause the robot to collide with itself

    Args:
        robot (Robot): The robot
        configs (torch.Tensor): [batch x n_dofs] tensor of robot configurations

    Returns:
        torch.Tensor: [batch] tensor of bools indicating if the given configs will self-collide
    """
    is_self_colliding = [robot.config_self_collides(config) for config in configs]
    return torch.tensor(is_self_colliding, dtype=torch.bool)


# TODO: What type should this return? Right now its a mix of torch.Tensor and np.ndarray
def evaluate_solutions(
    robot: Robot, target_poses: PT_NP_TYPE, solutions: torch.Tensor
) -> SOLUTION_EVALUATION_RESULT_TYPE:
    """Return detailed information about a batch of solutions. See the comment at the top of this file for a description
    of the SOLUTION_EVALUATION_RESULT_TYPE type - that's the type that this function returns

    Example usage:
        l2_errors, ang_errors, joint_limits_exceeded, self_collisions = evaluate_solutions(
            robot, ee_pose_target, samples
        )
    """
    assert isinstance(
        target_poses, (np.ndarray, torch.Tensor)
    ), f"target_poses must be a torch.Tensor or np.ndarray (got {type(target_poses)})"
    assert isinstance(solutions, torch.Tensor), f"solutions must be a torch.Tensor (got {type(solutions)})"
    target_poses = _get_target_pose_batch(target_poses, solutions.shape[0])
    l2_errors, angular_errors = solution_pose_errors(robot, solutions, target_poses)
    joint_limits_exceeded = calculate_joint_limits_exceeded(solutions, robot.actuated_joints_limits)
    self_collisions_respected = calculate_self_collisions(robot, solutions)
    return l2_errors, angular_errors, joint_limits_exceeded, self_collisions_respected


@enforce_pt_np_input
def angular_changes_old(qpath: PT_NP_TYPE) -> PT_NP_TYPE:
    """Computes the change in the configuration space path. Respects jumps from 0 <-> 2pi

    WARNING: Results may be negative. Be sure to call .abs() if calculating mjac

    Returns: a [n x ndofs] array of the change in each joint angle over the n timesteps.
    """
    angle_vector_1 = qpath[0:-1]
    angle_vector_2 = qpath[1:]
    if isinstance(qpath, np.ndarray):
        return np.arctan2(np.sin(angle_vector_2 - angle_vector_1), np.cos(angle_vector_2 - angle_vector_1))
    if isinstance(qpath, torch.Tensor):
        return torch.arctan2(torch.sin(angle_vector_2 - angle_vector_1), torch.cos(angle_vector_2 - angle_vector_1))


@enforce_pt_np_input
def angular_changes(qpath: PT_NP_TYPE) -> PT_NP_TYPE:
    """Same as angular_changes but ~2x faster to run"""
    dqs = qpath[1:] - qpath[0:-1]
    if isinstance(qpath, torch.Tensor):
        return torch.remainder(dqs + torch.pi, 2 * torch.pi) - torch.pi
    return np.remainder(dqs + np.pi, 2 * np.pi) - np.pi


@enforce_pt_np_input
def calculate_mean_cspace_diff_deg(x: PT_NP_TYPE):
    """Calculate the mean change in the configuration space path per joint, per timestep. Respects jumps from 0 <-> 2pi.
    """
    if isinstance(x, np.ndarray):
        return float(np.absolute(np.rad2deg(angular_changes(x))).mean())
    return float(torch.abs(torch.rad2deg(angular_changes(x))).mean())


@enforce_pt_np_input
def calculate_max_cspace_diff_deg(x: PT_NP_TYPE) -> float:
    """Calculate the maximum change in configuration space over a path in configuration space."""
    if isinstance(x, np.ndarray):
        return float(np.absolute(np.rad2deg(angular_changes(x))).max())
    return float(torch.abs(torch.rad2deg(angular_changes(x))).max())


@enforce_pt_np_input
def calculate_max_cspace_diff_per_timestep_deg(x: PT_NP_TYPE) -> PT_NP_TYPE:
    """Calculate the maximum change in configuration space over a path in configuration space."""
    if isinstance(x, np.ndarray):
        return np.rad2deg(np.max(np.absolute(angular_changes(x)), axis=1))
    raise NotImplementedError()
    # TODO: Unit test
    return torch.mean(torch.abs(torch.rad2deg(angular_changes(x))), dim=1)


""" Benchmarking solution_pose_errors():

python jkinpylib/evaluation.py

"""

if __name__ == "__main__":
    robot_ = Panda()

    print("\n=====================")
    print("Implementation: numpy")
    print("\nNumber of solutions | Time to calculate")
    print("------------------- | -----------------")
    for n_solutions_ in range(0, 2000, 100):
        solutions_np_ = np.random.randn(n_solutions_, 7)
        solutions_pt_ = torch.tensor(solutions_np_, device=DEVICE_TEMP, dtype=torch.float32)
        target_pose_ = robot_.forward_kinematics(robot_.sample_joint_angles(1))[0]

        runtimes = []
        for _ in range(5):
            t0 = time()
            l2_errors, ang_errors = solution_pose_errors(robot_, solutions_pt_, target_pose_)
            runtimes.append(time() - t0)
        runtime = float(np.mean(runtimes))
        print(f"{n_solutions_}  \t| {round(1000*(runtime), 5)}ms")

        # Benchmarking result: it is faster to convert solutions to numpy to calculate errors when n_solutions < 1000.
        # after that, it becomes faster to calculate errors with pytorch on the CPU. At some point longer in the future
        # it will be faster to calculate errors with pytorch on the GPU.

        # Current implementation:
        # =====================
        # Implementation: numpy

        # Number of solutions | Time to calculate
        # ------------------- | -----------------
        # 0	    | 0.52063ms
        # 100	| 2.02473ms
        # 200	| 4.13291ms
        # 300	| 4.01187ms
        # 400	| 5.37546ms
        # 500	| 5.68899ms
        # 600	| 6.61222ms
        # 700	| 7.88768ms
        # 800	| 8.47348ms
        # 900	| 9.6999ms
        # 1000	| 10.7499ms
        # 1100	| 11.24612ms
        # 1200	| 12.73076ms
        # 1300	| 13.11556ms
        # 1400	| 15.02244ms
        # 1500	| 15.03865ms
        # 1600	| 16.01609ms
        # 1700	| 16.87686ms
        # 1800	| 17.74081ms
        # 1900	| 19.45353ms
