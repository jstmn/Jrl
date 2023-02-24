from typing import Tuple
import unittest

import numpy as np
import torch

from jkinpylib.robot import Robot
from jkinpylib.robots import get_all_robots
from jkinpylib.evaluation import pose_errors
from jkinpylib.utils import set_seed
from jkinpylib.config import DEVICE

# Set seed to ensure reproducibility
set_seed()

# suppress=True: print with decimal notation, not scientific
np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True)


# TODO: Turn this into an actual test
class TestSolutionRerfinement(unittest.TestCase):
    @classmethod
    def setUpClass(clc):
        clc.robots = get_all_robots()

    def get_current_joint_angle_and_target_pose(self, robot: Robot, center: str) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        assert center in {"lower", "upper", "middle", "lower_out_of_bounds", "upper_out_of_bounds"}
        should_be_oob = "out_of_bounds" in center
        ndofs = robot.n_dofs

        # Get the current poses (these will be the seeds)
        x_current = np.ones((3, ndofs))
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
        if center == "lower":
            diff = 0.1
        elif center == "upper":
            diff = -0.1
        elif center == "middle":
            diff = 0.1
        elif center == "lower_out_of_bounds":
            diff = -0.05
        elif center == "upper_out_of_bounds":
            diff = 0.05

        _joint_angles_for_target_pose = x_current.copy()
        _joint_angles_for_target_pose[0, :] += diff
        _joint_angles_for_target_pose[1, :] += 2 * diff
        _joint_angles_for_target_pose[2, :] += 3 * diff

        if not should_be_oob:
            _joint_angles_for_target_pose = robot.clamp_to_joint_limits(_joint_angles_for_target_pose)

        poses_target = robot.forward_kinematics(_joint_angles_for_target_pose)

        # Sanity check joint angles used to get target poses plus the target poses
        if not should_be_oob:
            self.assertTrue(
                robot.joint_angles_all_in_joint_limits(_joint_angles_for_target_pose),
                msg=(
                    "joint angles out of"
                    f" limits\n_joint_angles_for_target_pose={_joint_angles_for_target_pose}\nrobot={robot}"
                ),
            )
        else:
            self.assertFalse(
                robot.joint_angles_all_in_joint_limits(_joint_angles_for_target_pose),
                msg=(
                    "joint angles should be out of"
                    f" limits\n_joint_angles_for_target_pose={_joint_angles_for_target_pose}\nrobot={robot}"
                ),
            )

        l2_diffs, angular_diffs = pose_errors(poses_target, poses_current)
        for i, (l2_diff_i, angular_diff_i) in enumerate(zip(l2_diffs, angular_diffs)):
            self.assertGreater(l2_diff_i, 0.005, msg=f"l2_diff = {l2_diff_i} should be > 0.005 (l2_dif={l2_diff_i})")
            self.assertGreater(
                angular_diff_i,
                0.005,
                msg=f"angular_diff = {angular_diff_i} should be > 0.005 (angular_dif={angular_diff_i})",
            )

        return x_current, poses_current, poses_target

    def assert_pose_errors_decreased(
        self, _poses_target: np.ndarray, _poses_original: np.ndarray, _poses_updated: np.ndarray, _method_name: str
    ):
        """Check that the pose errors decreased"""

        l2_errs_original, angular_errs_original = pose_errors(_poses_target, _poses_original)
        l2_errs_final, angular_errs_final = pose_errors(_poses_target, _poses_updated)
        l2_errs_differences = l2_errs_final - l2_errs_original
        angular_errs_differences = angular_errs_final - angular_errs_original
        for i, (l2_err_diff_i, angular_errs_diff_i) in enumerate(zip(l2_errs_differences, angular_errs_differences)):
            self.assertLess(l2_err_diff_i, 0.0, msg=f"({_method_name}) l2_err_diff = {l2_err_diff_i} should be < 0")
            self.assertLess(
                angular_errs_diff_i,
                0.0,
                msg=f"({_method_name}) angular_errs_diff_i = {angular_errs_diff_i} should be < 0",
            )

    def assert_batch_ik_step_makes_progress(
        self, robot: Robot, alpha: float, x_current: np.ndarray, poses_current: np.ndarray, poses_target: np.ndarray
    ):
        # Run ik
        poses_target_pt = torch.tensor(poses_target, device=DEVICE)
        x_current_pt = torch.tensor(x_current.copy(), device=DEVICE)
        x_updated_np = robot.inverse_kinematics_single_step_batch_np(poses_target, x_current, alpha)
        x_updated_pt = robot.inverse_kinematics_single_step_batch_pt(poses_target_pt, x_current_pt, alpha)
        x_updated_ad_pt = robot.inverse_kinematics_autodiff_single_step_batch_pt(poses_target_pt, x_current_pt, alpha)
        x_updated_pt = x_updated_pt.cpu().numpy()
        x_updated_ad_pt = x_updated_ad_pt.cpu().numpy()

        # Check that the updated joint angles are within joint limits
        self.assertTrue(
            robot.joint_angles_all_in_joint_limits(x_updated_np),
            f"joint angles out of limits\nx_updated_np={x_updated_np}\nrobot={robot}",
        )
        self.assertTrue(
            robot.joint_angles_all_in_joint_limits(x_updated_pt),
            f"joint angles out of limits\nx_updated_pt={x_updated_pt}\nrobot={robot}",
        )
        self.assertTrue(
            robot.joint_angles_all_in_joint_limits(x_updated_ad_pt),
            f"joint angles out of limits\nx_updated_pt={x_updated_ad_pt}\nrobot={robot}",
        )

        # Get updated pose error
        poses_updated_np = robot.forward_kinematics(x_updated_np)
        poses_updated_pt = robot.forward_kinematics(x_updated_pt)
        poses_updated_ad_pt = robot.forward_kinematics(x_updated_ad_pt)

        self.assert_pose_errors_decreased(poses_target, poses_current, poses_updated_np, "numpy")
        self.assert_pose_errors_decreased(poses_target, poses_current, poses_updated_pt, "pytorch")
        self.assert_pose_errors_decreased(poses_target, poses_current, poses_updated_ad_pt, "pytorch")

    # ==================================================================================================================
    #  -- Tests
    #

    def test_batch_ik_step_functions(self):
        """Test that ik steps made with inverse_kinematics_single_step_batch_pt() and
        inverse_kinematics_single_step_batch_np() make progress
        """

        alpha = 0.1

        for robot in self.robots:
            print()
            print(robot)

            if not robot._batch_fk_enabled:
                print("ignoring robot - batch_fk is disabled")
                continue

            # Check progress is made when joints are near their lower and upper limits + when near the center
            for center in ("lower", "middle", "upper", "lower_out_of_bounds", "upper_out_of_bounds"):
                print(f"  center = '{center}' -> \t", end="")
                x_current, poses_current, poses_target = self.get_current_joint_angle_and_target_pose(
                    robot, center=center
                )
                self.assert_batch_ik_step_makes_progress(robot, alpha, x_current, poses_current, poses_target)
                print("passed")


if __name__ == "__main__":
    unittest.main()
