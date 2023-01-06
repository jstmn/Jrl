from typing import Tuple
import unittest

from jkinpylib import config
from jkinpylib.robots import get_all_robots
from jkinpylib.robot import Robot, forward_kinematics_kinpy
from jkinpylib.conversions import geodesic_distance_between_quaternions_np

import torch
import numpy as np

torch.manual_seed(0)

DEVICE = config.device


def decimal_range(start, stop, inc):
    while start < stop:
        yield start
        start += inc


def get_gt_samples_and_endpoints(robot_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get ground truth samples and endpoints from a file."""

    return np.load(f"data/ground_truth_fk_data/{robot_name}__joint_angles.npy"), np.load(
        f"data/ground_truth_fk_data/{robot_name}__poses.npy"
    )


MAX_ALLOWABLE_L2_ERR = 5e-4
MAX_ALLOWABLE_ANG_ERR = 0.0008726646  # .05 degrees
ROBOTS = get_all_robots()


class TestForwardKinematics(unittest.TestCase):
    # ==================================================================================================================
    # Helper functions
    #

    def assert_endpose_position_almost_equal(
        self, endpoints1: np.array, endpoints2: np.array, source_1: str = "", source_2: str = ""
    ):
        """Check that the position of each pose is nearly the same"""
        l2_errors = np.linalg.norm(endpoints1[:, 0:3] - endpoints2[:, 0:3], axis=1)
        for i in range(l2_errors.shape[0]):
            self.assertLess(l2_errors[i], MAX_ALLOWABLE_L2_ERR, msg=f"FK between '{source_1}', '{source_2}' not equal")

    def assert_endpose_rotation_almost_equal(self, endpoints1: np.array, endpoints2: np.array, threshold=None):
        """Check that the rotation of each pose is nearly the same"""
        if threshold is None:
            threshold = MAX_ALLOWABLE_ANG_ERR
        rotational_errors = geodesic_distance_between_quaternions_np(endpoints1[:, 3 : 3 + 4], endpoints2[:, 3 : 3 + 4])
        for i in range(rotational_errors.shape[0]):
            self.assertLess(rotational_errors[i], threshold)

    def get_fk_poses(self, robot: Robot, samples: np.array) -> Tuple[np.array, np.array, Tuple[np.array, np.array]]:
        """Return fk solutions calculated by kinpy, klampt, and batch_fk"""
        kinpy_fk = forward_kinematics_kinpy(robot, samples)
        klampt_fk = robot.forward_kinematics_klampt(samples)

        if robot._batch_fk_enabled:
            batch_fk, batch_fk_runtime = robot.forward_kinematics_batch(
                torch.tensor(samples, dtype=torch.float32, device=DEVICE),
                out_device=DEVICE,
                return_runtime=True,
                return_quaternion=True,
            )
            batch_fk = batch_fk.cpu().numpy()
            """
            w precaching 1000
                Took 5.38 ms to get 500 poses
                Took 8.048 ms to get 10 poses
            w/o precaching
                Took 7.566 ms to get 500 poses
                Took 5.131 ms to get 10 poses
            """
            print(f"Took {round(batch_fk_runtime*1000, 3)} ms to get {samples.shape[0]} poses")
        else:
            batch_fk = (None, None)

        # TODO(@jeremysm): Get batch_fk_R to quaternion and return (n x 7) array
        return kinpy_fk, klampt_fk, batch_fk

    # ==================================================================================================================
    # Tests
    #

    def test_forward_kinematics_batch(self):
        """Test that forward_kinematics_batch is well formatted when returning both quaternions and transformation
        matrices"""
        for robot in ROBOTS:
            # Check 1: Return is correct for homogeneous transformation format
            samples = robot.sample_joint_angles(25)
            kinpy_fk = forward_kinematics_kinpy(robot, samples)
            batch_fk_T = robot.forward_kinematics_batch(
                torch.tensor(samples, dtype=torch.float32, device=DEVICE), out_device=DEVICE, return_quaternion=False
            )
            self.assertEqual(batch_fk_T.shape, (25, 4, 4))
            batch_fk_t = batch_fk_T[:, 0:3, 3].detach().cpu().numpy()
            np.testing.assert_allclose(kinpy_fk[:, 0:3], batch_fk_t, atol=1e-4)

            # Check 2: Return is correct for quaternion format
            samples = robot.sample_joint_angles(25)
            kinpy_fk = forward_kinematics_kinpy(robot, samples)
            klampt_fk = robot.forward_kinematics_klampt(samples)
            batch_fk = (
                robot.forward_kinematics_batch(
                    torch.tensor(samples, dtype=torch.float32, device=DEVICE), out_device=DEVICE, return_quaternion=True
                )
                .cpu()
                .numpy()
            )
            self.assertEqual(batch_fk.shape, (25, 7))
            np.testing.assert_allclose(kinpy_fk[:, 0:3], batch_fk[:, 0:3], atol=1e-4)
            self.assert_endpose_rotation_almost_equal(kinpy_fk, batch_fk)

    def test_fk_matches_saved_data(self):
        """
        Test that the all three forward kinematics functions return the expected value for saved input
        """
        for robot in ROBOTS:
            samples, endpoints_expected = get_gt_samples_and_endpoints(robot.name)
            kinpy_fk, klampt_fk, batch_fk = self.get_fk_poses(robot, samples)

            if robot._batch_fk_enabled:
                self.assert_endpose_position_almost_equal(kinpy_fk, batch_fk, "kinpy_fk", "batch_fk_t")
                self.assert_endpose_position_almost_equal(
                    batch_fk, endpoints_expected, "batch_fk", "endpoints_expected"
                )
                self.assert_endpose_rotation_almost_equal(kinpy_fk, batch_fk)
                self.assert_endpose_rotation_almost_equal(batch_fk, endpoints_expected)

            # fks batch eachother
            self.assert_endpose_position_almost_equal(kinpy_fk, klampt_fk)
            self.assert_endpose_rotation_almost_equal(kinpy_fk, klampt_fk)

            # fks match saved
            self.assert_endpose_position_almost_equal(kinpy_fk, endpoints_expected)
            self.assert_endpose_position_almost_equal(klampt_fk, endpoints_expected)

            self.assert_endpose_rotation_almost_equal(kinpy_fk, endpoints_expected)
            self.assert_endpose_rotation_almost_equal(klampt_fk, endpoints_expected)

    def test_fk_functions_equal(self):
        """
        Test that kinpy, klampt, and batch_fk all return the same poses
        """
        n_samples = 500
        for robot in ROBOTS:
            samples = robot.sample_joint_angles(n_samples)
            kinpy_fk, klampt_fk, batch_fk = self.get_fk_poses(robot, samples)
            self.assert_endpose_position_almost_equal(kinpy_fk, klampt_fk)
            self.assert_endpose_rotation_almost_equal(kinpy_fk, klampt_fk)

            if robot._batch_fk_enabled:
                self.assert_endpose_position_almost_equal(kinpy_fk, batch_fk, "kinpy_fk", "batch_fk")
                self.assert_endpose_rotation_almost_equal(kinpy_fk, batch_fk)

    def test_each_dimension_actuated(self):
        """
        Test that each dimension in n_dofs is actuated. This is done by asserting that there is either a positional
        or rotational change of the end effector when there is a change along each dimension of x
        """
        pos_min_diff = 0.001
        rad_min_diff = 0.001

        n_samples = 5
        for robot in ROBOTS:
            samples = robot.sample_joint_angles(n_samples)
            samples_fks = forward_kinematics_kinpy(robot, samples)

            # Iterate through each sample
            for sample_i in range(n_samples):
                # For each sample, iterate through the number of joints
                for joint_i in range(robot.n_dofs):
                    for offset in decimal_range(-np.pi, np.pi, 1.5):
                        pertubation = np.zeros(robot.n_dofs)
                        pertubation[joint_i] = offset
                        sample = np.array([samples[sample_i, :] + pertubation])
                        fk_i = forward_kinematics_kinpy(robot, sample)

                        positional_diff = np.linalg.norm(fk_i[0, 0:3] - samples_fks[sample_i, 0:3])
                        angular_diff = geodesic_distance_between_quaternions_np(
                            fk_i[0, 3:].reshape((1, 4)), samples_fks[sample_i, 3:].reshape((1, 4))
                        )[0]

                        self.assertTrue(
                            positional_diff > pos_min_diff or angular_diff > rad_min_diff,
                            msg="positional_diff={}, angular_diff={}".format(positional_diff, angular_diff),
                        )


if __name__ == "__main__":
    unittest.main()
