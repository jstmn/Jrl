import unittest

import torch
import numpy as np
from typing import List

from jkinpylib.config import DEVICE
from jkinpylib.robots import Panda
from jkinpylib.evaluation import (
    solution_pose_errors,
    calculate_joint_limits_exceeded,
    angular_changes,
    angular_changes_old,
)
from jkinpylib.utils import set_seed

set_seed()
np.set_printoptions(suppress=True, precision=8)
torch.set_default_dtype(torch.float32)

_2PI = 2 * np.pi


def _array_to_pt(arr: List) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


class AngularChangesTest(unittest.TestCase):
    def assert_angular_changes_correct(self, qpath, expected_diff):
        returned_1 = angular_changes(qpath)
        torch.testing.assert_close(expected_diff, returned_1)
        returned_2 = angular_changes_old(qpath)
        torch.testing.assert_close(expected_diff, returned_2)

    def test_angular_changes_1_vs_2(self):
        qpath = 10 * torch.randn((300, 8))
        from time import time

        t0 = time()
        returned_1 = angular_changes(qpath)
        print(1000 * (time() - t0))
        t0 = time()
        returned_2 = angular_changes_old(qpath)
        print(1000 * (time() - t0))
        torch.testing.assert_close(returned_1, returned_2)

    def test_angular_changes_pt(self):
        """Test that the angular_changes() function returns the correct values."""
        qpath = _array_to_pt(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        qpath_diff_expected = _array_to_pt(
            [
                [0, 0, 0],
                [0, 0, 0],
            ]
        )
        self.assert_angular_changes_correct(qpath, qpath_diff_expected)
        print("Passed test 1")

        #
        qpath = _array_to_pt(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0.1],
            ]
        )
        qpath_diff_expected = _array_to_pt(
            [
                [0, 0, 0],
                [0, 0, 0.1],
            ]
        )
        self.assert_angular_changes_correct(qpath, qpath_diff_expected)
        print("Passed test 2")

        #
        qpath = _array_to_pt(
            [
                [0, 0, 0],
                [0, 0, 0.1],
                [0, 0, -0.1],
            ]
        )
        qpath_diff_expected = _array_to_pt(
            [
                [0, 0, 0.1],
                [0, 0, -0.2],
            ]
        )
        self.assert_angular_changes_correct(qpath, qpath_diff_expected)
        print("Passed test 3")

        #
        qpath = _array_to_pt(
            [
                [0, -0.05, 0],
                [0, 0, 0.1],
                [0, 0, -0.1],
            ]
        )
        qpath_diff_expected = _array_to_pt(
            [
                [0, 0.05, 0.1],
                [0, 0, -0.2],
            ]
        )
        self.assert_angular_changes_correct(qpath, qpath_diff_expected)
        print("Passed test 4")

        #
        qpath = _array_to_pt(
            [
                [0, 0, 0],
                [0, 0, _2PI - 0.1],
                [0, 0, 0],
            ]
        )
        qpath_diff_expected = _array_to_pt(
            [
                [0, 0, -0.1],
                [0, 0, 0.1],
            ]
        )
        self.assert_angular_changes_correct(qpath, qpath_diff_expected)
        print("Passed test 5")

        #
        qpath = _array_to_pt(
            [
                [0, 0, 0],
                [0, 0, _2PI - 0.1],
                [-0.5, 0, 0.2],
            ]
        )
        qpath_diff_expected = _array_to_pt(
            [
                [0, 0, -0.1],
                [-0.5, 0, 0.3],
            ]
        )
        self.assert_angular_changes_correct(qpath, qpath_diff_expected)
        print("Passed test 6")


class EvaluationUtilsTest(unittest.TestCase):
    def test_solution_pose_errors(self):
        """Check that solution_pose_errors() matches the expected value"""
        robot = Panda()

        target_pose = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        solutions = torch.zeros((1, 7), dtype=torch.float32, device="cpu")
        realized_pose = robot.forward_kinematics(solutions.numpy())[0]
        realized_pose_gt = np.array([0.088, 0.0, 0.926, 0.0, 0.92387953, 0.38268343, 0.0])
        np.testing.assert_allclose(realized_pose, realized_pose_gt, atol=1e-5)  # sanity check values

        # np.sqrt((1 - 0.088 )**2 +  (1 - 0.0 )**2 + (1 - 0.926 )**2 ) = 1.355440887681938
        l2_error_expected = 1.355440887681938
        angular_error_expected = 3.1415927  # From https://www.andre-gaschler.com/rotationconverter/

        l2_error_returned, angular_error_returned = solution_pose_errors(robot, solutions, target_pose)
        self.assertAlmostEqual(l2_error_returned[0].item(), l2_error_expected)
        self.assertAlmostEqual(angular_error_returned[0].item(), angular_error_expected, delta=5e-4)

    def test_calculate_joint_limits_exceeded(self):
        """Test that calculate_joint_limits_exceeded() is working as expected"""

        configs = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 0],
                [-2, 0, 0],
                [0, -1.999, 0],
                [0, 2.0001, 0],
            ]
        )
        joint_limits = [
            (-1, 1),
            (-2, 2),
            (-3, 3),
        ]
        expected = torch.tensor([False, False, True, False, True], dtype=torch.bool)
        returned = calculate_joint_limits_exceeded(configs, joint_limits)
        self.assertEqual(returned.dtype, torch.bool)
        self.assertEqual(returned.shape, (5,))
        torch.testing.assert_close(returned, expected)


if __name__ == "__main__":
    unittest.main()
