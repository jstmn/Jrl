import unittest
from time import time

import torch
import numpy as np

from jrl.robot import Robot, IkLineSearchParameters
from jrl.robots import get_all_robots, Panda
from jrl.utils import set_seed, to_torch
from jrl.math_utils import geodesic_distance_between_quaternions
from .testing_utils import assert_pose_positions_almost_equal, assert_pose_rotations_almost_equal

# Set seed to ensure reproducibility
set_seed()

np.set_printoptions(edgeitems=30, linewidth=100000)


class TestInverseKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # self.robots = get_all_robots()
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

            print(f"  Success rate:  {round(100*(n_successes_kl / n), 2)}% ({n_successes_kl}/{n})")
            print(
                f"  Total runtime: {round(1000*total_ik_runtime, 3)}ms for {n} solutions\t (avg:"
                f" {round(1000*total_ik_runtime/n, 3)}ms/sol)"
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

            print(f"  Success rate:  {round(100*(n_successes_kl / n), 2)}% ({n_successes_kl}/{n})")
            print(
                f"  Total runtime: {round(1000*total_ik_runtime, 3)}ms for {n} solutions\t (avg:"
                f" {round(1000*total_ik_runtime/n, 3)}ms/sol)"
            )

    # python -m unittest tests.inverse_kinematics_test.TestInverseKinematics.test_line_search
    def test_line_search(self):
        """Test the IK line search functionality"""
        print("\n\n-----------------------------------")
        print(" -- Test line search -- ")
        n = 10
        rand_scale = 0.5
        qs_0, poses_0 = self.panda.sample_joint_angles_and_poses(n, return_torch=True)
        qs_pert = qs_0 + rand_scale*(torch.rand_like(qs_0) - 0.5)
        print("q difference:", torch.rad2deg(qs_pert - qs_0))

        import matplotlib.pyplot as plt
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(14, 8))
        fig.suptitle("alpha vs. end effector pose error convergence.\nLeft: LMA, Right: pinv(J). Number in legend: alpha")
        axl.set_xlabel("")
        axr.set_xlabel("")
        axl.set_ylabel("Error [mm]")
        axr.set_ylabel("Error [deg]")
        axl.grid("both", alpha=0.2)
        axr.grid("both", alpha=0.2)

        import matplotlib.colors

        # colors = matplotlib.colors.TABLEAU_COLORS
        def plot_qs(qs_, label, large_dot = False, color = None, offset = False):
            ee = self.panda.forward_kinematics(qs_)
            pos_errors = torch.norm(ee[:, 0:3] - poses_0[:, 0:3], dim=1) * 1000
            rot_errors = torch.rad2deg(geodesic_distance_between_quaternions(ee[:, 3 : 3 + 4], poses_0[:, 3 : 3 + 4]))
            axl.scatter(np.array(list(range(n))) + (0.25 if offset else 0.0), pos_errors.cpu().numpy(), label=label, s=75.0 if large_dot else None, color=color)
            axr.scatter(np.array(list(range(n))) + (0.25 if offset else 0.0), rot_errors.cpu().numpy(), label=label, s=75.0 if large_dot else None, color=color)


        # TODO: get convergence for different alphas
        plot_qs(qs_pert, "qs original", large_dot=True, color="black")

        # for alpha in [0.01, 0.25, 0.5, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5]:
        for alpha in [0.01, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
            plot_qs(self.panda.inverse_kinematics_step_levenburg_marquardt(poses_0, qs_pert, alpha=alpha), f"LM: {alpha}")
            plot_qs(self.panda.inverse_kinematics_step_jacobian_pinv(poses_0, qs_pert, alpha=alpha), f"pinv(J): {alpha}", offset=True)
            # delta_q = self.panda.inverse_kinematics_step_levenburg_marquardt(poses_0, qs_pert, alpha=1.0) - qs_pert
            # params = IkLineSearchParameters()
            # alphas = self.panda._perform_ik_line_search(poses_0, qs_pert, params)


        # axl.legend()
        axr.legend()
        plt.show()



if __name__ == "__main__":
    unittest.main()
