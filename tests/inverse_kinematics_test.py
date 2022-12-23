from typing import Tuple
import unittest

from jkinpylib.robot import Robot
from jkinpylib import config
from jkinpylib.robots import PandaArmStanford, get_all_robots
from jkinpylib.conversions import geodesic_distance_between_quaternions_np
from jkinpylib.utils import set_seed

import torch
import numpy as np

# Set seed to ensure reproducibility
set_seed()

np.set_printoptions(edgeitems=30, linewidth=100000)


MAX_ALLOWABLE_L2_ERR = 2.5e-4
MAX_ALLOWABLE_ANG_ERR = 0.01  # degrees


class TestInverseKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.panda_arm_stanford = PandaArmStanford()
        self.robots = get_all_robots()

    # Helper functions
    def assert_pose_position_almost_equal(
        self, endpoints1: np.array, endpoints2: np.array, max_allowable_l2_err=MAX_ALLOWABLE_L2_ERR
    ):
        """Check that the position of each pose is nearly the same"""
        l2_errors = np.linalg.norm(endpoints1[:, 0:3] - endpoints2[:, 0:3], axis=1)
        self.assertLess(np.max(l2_errors), max_allowable_l2_err)
        for i in range(l2_errors.shape[0]):
            self.assertLess(l2_errors[i], max_allowable_l2_err)

    def assert_pose_rotation_almost_equal(
        self, endpoints1: np.array, endpoints2: np.array, max_allowable_rotational_err=MAX_ALLOWABLE_ANG_ERR
    ):
        """Check that the rotation of each pose is nearly the same"""
        rotational_errors = geodesic_distance_between_quaternions_np(endpoints1[:, 3 : 3 + 4], endpoints2[:, 3 : 3 + 4])
        for i in range(rotational_errors.shape[0]):
            self.assertLess(rotational_errors[i], max_allowable_rotational_err)

    def solution_valid(self, robot: Robot, solution: np.ndarray, pose_gt: np.ndarray, positional_tol: float):
        if solution is None:
            print(" -> Solution is None, failing")
            return False, -1
        self.assertEqual(solution.shape, (1, robot.n_dofs))
        poses_ik = robot.forward_kinematics_klampt(solution)
        self.assertEqual(poses_ik.shape, (1, 7))
        # Check solution error
        l2_err = np.linalg.norm(pose_gt[0:3] - poses_ik[0, 0:3])
        if l2_err > 1.5 * positional_tol:
            print(" -> Error to large, failing")
            return False, l2_err
        pose_gt = pose_gt.reshape(1, 7)
        self.assert_pose_position_almost_equal(pose_gt, poses_ik, max_allowable_l2_err=1.5 * positional_tol)
        self.assert_pose_rotation_almost_equal(pose_gt, poses_ik)
        return True, l2_err

    # --- Tests
    def test_seed_failure_recovery_klampt(self):
        """Test that inverse_kinematics_klampt() generates a solution if the original seed fails"""
        pose = np.array([-0.45741714, -0.08548167, 0.87084611, 0.09305326, -0.49179573, -0.86208266, 0.07931919])
        seed = np.array([0.46013147, -0.71480753, 1.74743252, -0.34429741, 1.08508085, 0.64453392, 1.82583597])
        positional_tol = 1e-3
        solution = self.panda_arm_stanford.inverse_kinematics_klampt(
            pose, seed=seed, positional_tolerance=positional_tol, verbosity=0
        )
        self.assertIsInstance(solution, np.ndarray)
        solution_pose = self.panda_arm_stanford.forward_kinematics_klampt(solution)
        l2_err = np.linalg.norm(pose[0:3] - solution_pose[0, 0:3])
        self.assertLess(l2_err, 2 * positional_tol)

    def test_ik_with_seed(self):
        """
        Test that fk(ik(fk(sample))) = fk(sample) using a seed in ik(...)

        This test is to ensure that the seed is working properly. Example code to print print out the realized end pose
        error of the perturbed joint angles:
            samples_noisy_poses = robot.forward_kinematics_klampt(samples_noisy)
            samples_noisy_l2_errors = np.linalg.norm(samples_noisy_poses[:, 0:3] - poses_gt[:, 0:3], axis=1)
            print("Perturbed configs L2 end pose error:", samples_noisy_l2_errors)
        """
        print("\n\n----------------------------------------\n----------------------------------------")
        print("> Test klampt IK - with seed\n")
        n = 50
        n_tries = 50
        positional_tol = 1e-4
        verbosity = 0

        for robot in self.robots:
            print(f"Testing {robot}")
            n_successes_kl = 0
            samples = robot.sample_joint_angles(n)
            poses_gt = robot.forward_kinematics_klampt(samples)
            samples_noisy = samples + np.random.normal(0, 0.1, samples.shape)

            for i, (sample_noisy, pose_gt) in enumerate(zip(samples_noisy, poses_gt)):
                solution_klampt = robot.inverse_kinematics_klampt(
                    pose_gt,
                    seed=sample_noisy,
                    positional_tolerance=positional_tol,
                    n_tries=n_tries,
                    verbosity=verbosity,
                )
                klampt_valid, l2_err = self.solution_valid(robot, solution_klampt, pose_gt, positional_tol)

                if not klampt_valid:
                    print(
                        f"Klampt failed ({i}/{n}). l2_err: {l2_err} (max is {positional_tol}) pose, seed:\n",
                        pose_gt,
                        ",",
                        sample_noisy,
                    )
                else:
                    self.assertLess(l2_err, 1e-3)
                    n_successes_kl += 1

        print(f"Success rate, klampt (local seed): {round(100*(n_successes_kl / n), 2)}% ({n_successes_kl}/{n})")

    def test_ik_with_random_seed(self):
        """Test that fk(ik(fk(sample))) = fk(sample) with a random a seed used by klampt"""
        print("\n\n----------------------------------------\n----------------------------------------")
        print("> Test klampt IK - with random seed\n")
        n = 25
        n_tries = 50
        positional_tol = 1e-4
        verbosity = 0
        robot = self.panda_arm_stanford

        for robot in self.robots:
            print(f"Testing {robot}")

            n_successes_kl = 0
            poses_gt = robot.forward_kinematics_klampt(robot.sample_joint_angles(n))

            for i, pose_gt in enumerate(poses_gt):
                solution_klampt = robot.inverse_kinematics_klampt(
                    pose_gt, seed=None, positional_tolerance=positional_tol, n_tries=n_tries, verbosity=verbosity
                )
                klampt_valid, l2_err = self.solution_valid(robot, solution_klampt, pose_gt, positional_tol)
                if not klampt_valid:
                    print(f"Klampt failed ({i}/{n}). pose:", pose_gt, f"l2_err: {l2_err} (max is {positional_tol})")
                else:
                    self.assertLess(l2_err, 1e-3)
                    n_successes_kl += 1

            print(f"Success rate, klampt (random seed): {round(100*(n_successes_kl / n), 2)}% ({n_successes_kl}/{n})")


if __name__ == "__main__":
    unittest.main()
