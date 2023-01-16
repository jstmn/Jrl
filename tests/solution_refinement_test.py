from typing import Tuple
import unittest

import numpy as np
import torch

from jkinpylib.robot import Robot
from jkinpylib.robots import Panda
from jkinpylib.conversions import geodesic_distance_between_quaternions_np
from jkinpylib.utils import set_seed
from jkinpylib.config import device

# Set seed to ensure reproducibility
set_seed()

# suppress=True: print with decimal notation, not scientific
np.set_printoptions(edgeitems=30, linewidth=100000, suppress=True)


# TODO: Turn this into an actual test
class TestSolutionRerfinement(unittest.TestCase):
    @classmethod
    def setUpClass(clc):
        clc.panda = Panda()

    # def test_inverse_kinematics_single_step_batch_np(self):
    #     """Test that ik steps made with inverse_kinematics_single_step_batch_np() are making progress"""
    #     robot = self.panda
    #     alpha = 0.1

    #     # Get the current poses (these will be the seeds)
    #     x_current = 0.0 * np.ones((2, 7))
    #     x_current[0, 0] = 0.0
    #     x_current[1, 0] = 0.1
    #     # x_current[2, 0] = 0.2
    #     # x_current[3, 0] = 0.3
    #     current_poses = robot.forward_kinematics(x_current)

    #     # Get the target poses
    #     _target_pose_xs = np.zeros((2, 7))
    #     # _target_pose_xs = np.zeros((4, 7))
    #     _target_pose_xs[:, 0] = 0.5
    #     target_poses = robot.forward_kinematics(_target_pose_xs)
    #     l2_errs_original = np.linalg.norm(target_poses[:, 0:3] - current_poses[:, 0:3], axis=1)

    #     print("\n  ------ <fn>\n")
    #     x_updated, _ = robot.inverse_kinematics_single_step_batch_np(target_poses, x_current, alpha)
    #     print("\n  ------ </fn>\n")
    #     updated_poses = robot.forward_kinematics(x_updated)
    #     l2_errs_final = np.linalg.norm(target_poses[:, 0:3] - updated_poses[:, 0:3], axis=1)

    #     # print("x_current:\n", x_current)
    #     # print("x_updated:\n", x_updated)

    #     # print("\n-----")
    #     print("target poses: \n", target_poses)
    #     print("current poses:\n", current_poses)
    #     print("updated_poses:\n", updated_poses)
    #     # print("\n-----")
    #     print("l2 errors initial:", l2_errs_original)
    #     print("l2 errors final:  ", l2_errs_final)

    def test_inverse_kinematics_single_step_batch_pytorch(self):
        """Test that ik steps made with inverse_kinematics_single_step_batch_pt() are making progress"""
        robot = self.panda
        alpha = 0.25

        # Get the current poses (these will be the seeds)
        x_current = 0.0 * torch.ones((2, 7), device=device)
        x_current[0, 0] = 0.0
        x_current[1, 0] = 0.1
        current_poses = torch.tensor(robot.forward_kinematics(x_current.cpu().numpy()), device=device)

        # Get the target poses
        _target_pose_xs = np.zeros((2, 7))
        _target_pose_xs[:, 0] = 0.5
        target_poses = torch.tensor(robot.forward_kinematics(_target_pose_xs), device=device)
        l2_errs_original = torch.norm(target_poses[:, 0:3] - current_poses[:, 0:3], dim=1)

        # print("\n  ------ <fn>\n")
        x_updated, _ = robot.inverse_kinematics_single_step_batch_pt(target_poses, x_current, alpha)
        # print("\n  ------ </fn>\n")
        updated_poses = torch.tensor(robot.forward_kinematics(x_updated.cpu().numpy()), device=device)
        l2_errs_final = torch.norm(target_poses[:, 0:3] - updated_poses[:, 0:3], dim=1)

        # # print("x_current:\n", x_current)
        # # print("x_updated:\n", x_updated)

        # # print("\n-----")
        # print("target poses: \n", target_poses)
        # print("current poses:\n", current_poses)
        # print("updated_poses:\n", updated_poses)
        # # print("\n-----")
        print("l2 errors initial:", l2_errs_original)
        print("l2 errors final:  ", l2_errs_final)


if __name__ == "__main__":
    unittest.main()
