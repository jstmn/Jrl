import unittest
from time import time

import torch
import numpy as np

from jrl.robot import Robot
from jrl.robots import get_all_robots, Panda
from jrl.utils import set_seed, to_torch, evenly_spaced_colors
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
        n = 50

        import matplotlib.pyplot as plt
        noise_scales = [0.1, 0.5, 1.0, 3.1415]
        fig, axs = plt.subplots(len(noise_scales), 3, figsize=(18, 20))
        fig.suptitle("Step-size vs. end effector pose error convergence. Step direction calculated with Levenberg Marquardt")
        # fig.tight_layout()

        alphas = [0.0, 0.1, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5]
        colors = evenly_spaced_colors(len(alphas))

        for row_idx, noise_scale in enumerate(noise_scales):

            rand_scale = noise_scale
            qs_0, poses_0 = self.panda.sample_joint_angles_and_poses(n, return_torch=True)
            qs_pert = self.panda.clamp_to_joint_limits(qs_0 + rand_scale*(torch.rand_like(qs_0) - 0.5))

            # get summary stats
            ee = self.panda.forward_kinematics(qs_pert)
            mean_pos_error = (torch.norm(ee[:, 0:3] - poses_0[:, 0:3], dim=1).mean() * 100).item()
            mean_rot_error = torch.rad2deg(geodesic_distance_between_quaternions(ee[:, 3 : 3 + 4], poses_0[:, 3 : 3 + 4])).mean().item()
            # print("q difference:", torch.rad2deg(qs_pert - qs_0))

            axl = axs[row_idx, 0]
            axr = axs[row_idx, 1]
            axrr = axs[row_idx, 2]
            axl.set_title(f"noise_scale: {noise_scale}, ave pos_error: {mean_pos_error:.3f} [cm], ave rot_error: {mean_rot_error:.3f} [deg]")
            axl.set_xlabel("")
            axl.set_xlabel("")
            axr.set_xlabel("")
            axrr.set_xlabel("Alpha")
            axl.set_ylabel("Error [cm]")
            axr.set_ylabel("Error [deg]")
            axrr.set_ylabel("Error [cm]")
            axl.grid("both", alpha=0.2)
            axr.grid("both", alpha=0.2)
            axrr.grid("both", alpha=0.2)


            # colors = matplotlib.colors.TABLEAU_COLORS
            def plot_qs(qs_, label, i, large_dot = False, color = None, offset = False):
                color = colors[i] if color is None else color
                ee = self.panda.forward_kinematics(qs_)
                pos_errors = torch.norm(ee[:, 0:3] - poses_0[:, 0:3], dim=1) * 100
                rot_errors = torch.rad2deg(geodesic_distance_between_quaternions(ee[:, 3 : 3 + 4], poses_0[:, 3 : 3 + 4]))
                axl.scatter(np.array(list(range(n))) + (0.25 if offset else 0.0), pos_errors.cpu().numpy(), label=label, s=75.0 if large_dot else None, color=color)
                axr.scatter(np.array(list(range(n))) + (0.25 if offset else 0.0), rot_errors.cpu().numpy(), label=label, s=75.0 if large_dot else None, color=color)
                return pos_errors.mean().item(), pos_errors.std().item(), rot_errors.mean().item(), rot_errors.std().item()

            plot_qs(qs_pert, "qs original", None, large_dot=True, color="black")

            mean_pos_errors = []
            std_pos_errors = []
            for i, alpha in enumerate(alphas):
                mean, std, _, _ = plot_qs(self.panda.inverse_kinematics_step_levenburg_marquardt(poses_0, qs_pert, alpha=alpha), f"LM: {alpha}", i)
                mean_pos_errors.append(mean)
                std_pos_errors.append(std)
            self.assertAlmostEqual(mean_pos_error, mean_pos_errors[0], delta=1e-5)

            axrr.fill_between(
                alphas,
                np.array(mean_pos_errors) - np.array(std_pos_errors),
                np.array(mean_pos_errors) + np.array(std_pos_errors),
                alpha=0.1,
                label="std",
                color="b"
            )
            axrr.plot(alphas, mean_pos_errors, color="b")
            axrr.scatter(alphas, mean_pos_errors, color="b")
            axrr.set_ylim(0, max(mean_pos_errors)*1.5)
            axrr.plot([0, max(alphas)], [mean_pos_error, mean_pos_error], color="k", linestyle="dashed", alpha=0.5)


            # axl.legend()
        axs[0, 0].legend()
        plt.show()



if __name__ == "__main__":
    unittest.main()
