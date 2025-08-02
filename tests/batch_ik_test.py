from typing import Tuple
import unittest

import numpy as np
import torch

from jrl.robot import Robot
from jrl.utils import set_seed, to_torch
from jrl.math_utils import geodesic_distance_between_quaternions
from jrl.config import DEVICE, PT_NP_TYPE
from jrl.testing_utils import all_robots

# Set seed to ensure reproducibility
set_seed()
np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True)


def _pose_errors(poses_1: PT_NP_TYPE, poses_2: PT_NP_TYPE) -> Tuple[PT_NP_TYPE, PT_NP_TYPE]:
    """Return the positional and rotational angular error between two batch of poses."""
    assert poses_1.shape == poses_2.shape, f"Poses are of different shape: {poses_1.shape} != {poses_2.shape}"

    if isinstance(poses_1, torch.Tensor):
        assert isinstance(poses_2, torch.Tensor), f"poses_1 is a tensor but poses_2 is not: {type(poses_2)}"
        l2_errors = torch.norm(poses_1[:, 0:3] - poses_2[:, 0:3], dim=1)
    else:
        assert isinstance(poses_2, np.ndarray), f"poses_1 is a numpy array but poses_2 is not: {type(poses_2)}"
        l2_errors = np.linalg.norm(poses_1[:, 0:3] - poses_2[:, 0:3], axis=1)
    angular_errors = geodesic_distance_between_quaternions(poses_1[:, 3 : 3 + 4], poses_2[:, 3 : 3 + 4])
    assert l2_errors.shape == angular_errors.shape
    return l2_errors, angular_errors


def _joint_angles_all_in_joint_limits(robot: Robot, x: PT_NP_TYPE, eps: float = 1e-6) -> bool:
    """Return whether all joint angles are within joint limits

    Args:
        x (PT_NP_TYPE): [batch x ndofs] tensor of joint angles
    """
    assert x.shape[1] == robot.ndof
    for i, (lower, upper) in enumerate(robot.actuated_joints_limits):
        if x[:, i].min() < lower - eps:
            print("error: x[:, i].min() < lower", x[:, i].min(), lower)
            return False
        if x[:, i].max() > upper + eps:
            print("error: x[:, i].max() > lower", x[:, i].max(), upper)
            return False
    return True


class TestBatchIK(unittest.TestCase):
    @classmethod
    def setUpClass(clc):
        clc.robots = all_robots

    def get_current_joint_angle_and_target_pose(self, robot: Robot, center: str) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        assert center in {"lower", "upper", "middle", "lower_out_of_bounds", "upper_out_of_bounds"}
        center_to_diff = {
            "lower": 0.1,
            "upper": -0.1,
            "middle": 0.1,
            "lower_out_of_bounds": -0.05,
            "upper_out_of_bounds": 0.05,
        }
        should_be_oob = "out_of_bounds" in center
        ndofs = robot.ndof

        # Get the current poses (these will be the seeds)
        x_current = torch.ones((3, ndofs))
        for i, (lower, upper) in enumerate(robot.actuated_joints_limits):
            if center == "lower":
                x_current[:, i] = lower
            elif center == "upper":
                x_current[:, i] = upper
            elif center == "middle":
                midpoint = (lower + upper) / 2
                x_current[:, i] = midpoint
            # Intentionally out joint limit joint angles
            elif center == "lower_out_of_bounds":
                x_current[:, i] = lower + 0.1
            elif center == "upper_out_of_bounds":
                x_current[:, i] = upper - 0.1

        poses_current = robot.forward_kinematics(x_current)

        # Get the target poses
        diff = center_to_diff[center]
        _qs_for_target_pose = x_current.clone().detach()
        _qs_for_target_pose[0, :] += diff
        _qs_for_target_pose[1, :] += 2 * diff
        _qs_for_target_pose[2, :] += 3 * diff

        if not should_be_oob:
            _qs_for_target_pose = robot.clamp_to_joint_limits(_qs_for_target_pose)

        poses_target = to_torch(robot.forward_kinematics(_qs_for_target_pose))

        # Sanity check joint angles used to get target poses plus the target poses
        if not should_be_oob:
            self.assertTrue(
                _joint_angles_all_in_joint_limits(robot, _qs_for_target_pose),
                msg=f"joint angles out of limits\nn_qs_for_target_pose={_qs_for_target_pose}\nrobot={robot}",
            )
        else:
            self.assertFalse(
                _joint_angles_all_in_joint_limits(robot, _qs_for_target_pose),
                msg=f"joint angles should be out of limits\nn_qs_for_target_pose={_qs_for_target_pose}\nrobot={robot}",
            )

        l2_diffs, angular_diffs = _pose_errors(poses_target, poses_current)
        for i, (l2_diff_i, angular_diff_i) in enumerate(zip(l2_diffs, angular_diffs)):
            self.assertGreater(l2_diff_i, 0.005, msg=f"l2_diff = {l2_diff_i} should be > 0.005 (l2_dif={l2_diff_i})")
            self.assertGreater(
                angular_diff_i,
                0.005,
                msg=f"angular_diff = {angular_diff_i} should be > 0.005 (angular_dif={angular_diff_i})",
            )

        return x_current, poses_current, poses_target

    def assert_pose_errors_decreased(
        self, _poses_target: np.ndarray, _poses_original: np.ndarray, _poses_updated: np.ndarray, test_description: str
    ):
        """Check that the pose errors decreased"""
        _poses_target = torch.tensor(_poses_target)
        _poses_updated = torch.tensor(_poses_updated)
        l2_errs_original, angular_errs_original = _pose_errors(_poses_target, _poses_original)
        l2_errs_final, angular_errs_final = _pose_errors(_poses_target, _poses_updated)
        l2_errs_differences = l2_errs_final - l2_errs_original
        angular_errs_differences = angular_errs_final - angular_errs_original
        for i, (l2_err_diff_i, angular_errs_diff_i) in enumerate(zip(l2_errs_differences, angular_errs_differences)):
            self.assertLess(
                l2_err_diff_i,
                0.0,
                msg=(
                    f"(position) error_final - error_initial should be < 0, is {l2_err_diff_i}. This means the"
                    f" positional error increased ({test_description})"
                ),
            )
            self.assertLess(
                angular_errs_diff_i,
                0.0,
                msg=(
                    f"(rotational) error_final - error_initial should be < 0, is {angular_errs_diff_i}. This means the"
                    f" angular error increased ({test_description})"
                ),
            )

    def assert_batch_ik_step_makes_progress(
        self, robot: Robot, alpha: float, x_current: np.ndarray, poses_current: np.ndarray, poses_target: np.ndarray
    ):
        # Run ik
        poses_target_pt = torch.tensor(poses_target, device=DEVICE)
        x_updated_pt = robot.inverse_kinematics_step_levenburg_marquardt(poses_target_pt, x_current, alpha)

        self.assertTrue(
            _joint_angles_all_in_joint_limits(robot, x_updated_pt),
            f"joint angles out of limits\nx_updated_pt={x_updated_pt}\nrobot={robot}",
        )
        poses_updated_pt = robot.forward_kinematics(x_updated_pt)
        self.assert_pose_errors_decreased(poses_target, poses_current, poses_updated_pt, f"{robot.name}, jac-pinv")

    # ==================================================================================================================
    #  -- Tests
    #

    def test_batch_ik_step_functions(self):
        """Test that ik steps made with inverse_kinematics_step_levenburg_marquardt()  makes progress"""

        alpha = 0.05

        for robot in self.robots:
            print()
            print(robot)

            if not robot._batch_fk_enabled:
                print("ignoring robot - batch_fk is disabled")
                continue

            # Check progress is made when joints are near their lower and upper limits + when near the center
            for center in ("lower", "middle", "upper", "lower_out_of_bounds", "upper_out_of_bounds"):
                print(f"  center = '{center}' ->\t", end="")
                x_current, poses_current, poses_target = self.get_current_joint_angle_and_target_pose(
                    robot, center=center
                )
                self.assert_batch_ik_step_makes_progress(robot, alpha, x_current, poses_current, poses_target)
                print("passed")
        print("all passed")


if __name__ == "__main__":
    unittest.main()
