import unittest
from time import time

import numpy as np

from jrl.robot import Robot
from jrl.robots import Panda
from jrl.utils import set_seed, to_torch
from jrl.testing_utils import assert_pose_positions_almost_equal, assert_pose_rotations_almost_equal, all_robots

# Set seed to ensure reproducibility
set_seed()

np.set_printoptions(edgeitems=30, linewidth=100000)


class TestInverseKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.robots = all_robots
        self.panda = Panda()

    def assert_solution_is_valid(self, robot: Robot, solution: np.ndarray, pose_gt: np.ndarray, positional_tol: float):
        if solution is None:
            print(" -> Solution is None, failing")
            return False, -1
        self.assertEqual(solution.shape, (1, robot.ndof))
        poses_ik = robot.forward_kinematics_klampt(solution)
        self.assertEqual(poses_ik.shape, (1, 7))
        # Check solution error
        l2_err = np.linalg.norm(pose_gt[0:3] - poses_ik[0, 0:3])
        if l2_err > 1.5 * positional_tol:
            print(" -> Error too large, failing")
            return False, l2_err
        pose_gt = pose_gt.reshape(1, 7)
        assert_pose_positions_almost_equal(to_torch(pose_gt), to_torch(poses_ik), threshold=1.5 * positional_tol)
        assert_pose_rotations_almost_equal(to_torch(pose_gt), to_torch(poses_ik), threshold=0.0025)
        return True, l2_err

    # --- Tests
    def test_seed_failure_recovery_klampt(self):
        """Test that inverse_kinematics_klampt() generates a solution if the original seed fails"""
        pose = np.array([-0.45741714, -0.08548167, 0.87084611, 0.09305326, -0.49179573, -0.86208266, 0.07931919])
        seed = np.array([0.46013147, -0.71480753, 1.74743252, -0.34429741, 1.08508085, 0.64453392, 1.82583597])
        positional_tol = 1e-3
        solution = self.panda.inverse_kinematics_klampt(
            pose, seed=seed, positional_tolerance=positional_tol, verbosity=0
        )
        self.assertIsInstance(solution, np.ndarray)
        solution_pose = self.panda.forward_kinematics_klampt(solution)
        l2_err = np.linalg.norm(pose[0:3] - solution_pose[0, 0:3])
        self.assertLess(l2_err, 2 * positional_tol)

    def test_inverse_kinematics_klampt_with_seed(self):
        """
        Test that fk(inverse_kinematics_klampt(fk(sample))) = fk(sample) using a seed in inverse_kinematics_klampt(...)

        This test is to ensure that the seed is working properly. Example code to print print out the realized end pose
        error of the perturbed joint angles:
            samples_noisy_poses = robot.forward_kinematics_klampt(samples_noisy)
            samples_noisy_l2_errors = np.linalg.norm(samples_noisy_poses[:, 0:3] - poses_gt[:, 0:3], axis=1)
            print("Perturbed configs L2 end pose error:", samples_noisy_l2_errors)
        """
        print("\n\n----------------------------------------")
        print(" -- Test klampt IK - with seed -- ")
        n = 100
        n_tries = 50
        positional_tol = 1e-4
        verbosity = 0
        total_ik_runtime = 0

        for robot in self.robots:
            print(f"\n{robot}")
            n_successes_kl = 0
            samples = robot.sample_joint_angles(n)
            poses_gt = robot.forward_kinematics_klampt(samples)
            samples_noisy = robot.clamp_to_joint_limits(samples + np.random.normal(0, 0.1, samples.shape))

            for i, (sample_noisy, pose_gt) in enumerate(zip(samples_noisy, poses_gt)):
                t0 = time()
                solution_klampt = robot.inverse_kinematics_klampt(
                    pose_gt,
                    seed=sample_noisy,
                    positional_tolerance=positional_tol,
                    n_tries=n_tries,
                    verbosity=verbosity,
                )
                total_ik_runtime += time() - t0
                klampt_valid, l2_err = self.assert_solution_is_valid(robot, solution_klampt, pose_gt, positional_tol)

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

            print(f"  Success rate:  {round(100 * (n_successes_kl / n), 2)}% ({n_successes_kl}/{n})")
            print(
                f"  Total runtime: {round(1000 * total_ik_runtime, 3)}ms for {n} solutions\t (avg:"
                f" {round(1000 * total_ik_runtime / n, 3)}ms/sol)"
            )

    def test_inverse_kinematics_klampt_with_random_seed(self):
        """Test that fk(inverse_kinematics_klampt(fk(sample))) = fk(sample) with a random a seed used by klampt"""
        print("\n\n-----------------------------------")
        print(" -- Test klampt IK - with random seed -- ")
        n = 100
        n_tries = 50
        positional_tol = 1e-4
        verbosity = 0
        robot = self.panda
        total_ik_runtime = 0

        for robot in self.robots:
            print(f"\n{robot}")

            n_successes_kl = 0
            poses_gt = robot.forward_kinematics_klampt(robot.sample_joint_angles(n))

            for i, pose_gt in enumerate(poses_gt):
                t0 = time()
                solution_klampt = robot.inverse_kinematics_klampt(
                    pose_gt, seed=None, positional_tolerance=positional_tol, n_tries=n_tries, verbosity=verbosity
                )
                total_ik_runtime += time() - t0
                klampt_valid, l2_err = self.assert_solution_is_valid(robot, solution_klampt, pose_gt, positional_tol)
                if not klampt_valid:
                    print(f"Klampt failed ({i}/{n}). pose:", pose_gt, f"l2_err: {l2_err} (max is {positional_tol})")
                else:
                    self.assertLess(l2_err, 1e-3)
                    n_successes_kl += 1

            print(f"  Success rate:  {round(100 * (n_successes_kl / n), 2)}% ({n_successes_kl}/{n})")
            print(
                f"  Total runtime: {round(1000 * total_ik_runtime, 3)}ms for {n} solutions\t (avg:"
                f" {round(1000 * total_ik_runtime / n, 3)}ms/sol)"
            )


if __name__ == "__main__":
    unittest.main()
