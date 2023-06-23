import unittest

import torch
import numpy as np

from jrl.urdf_utils import _len3_tuple_from_str
from jrl.utils import set_seed
from jrl.robot import Robot
from jrl.robots import get_all_robots, Panda, Fetch, Iiwa7, FetchArm
from jrl.conversions import PT_NP_TYPE
from jrl.config import DEVICE

set_seed(0)

ROBOTS = get_all_robots()


class RobotTest(unittest.TestCase):
    def _assert_joint_angles_within_limits(self, joint_angles: PT_NP_TYPE, robot: Robot):
        for joint_angle in joint_angles:
            for i in range(robot.n_dofs):
                self.assertGreaterEqual(joint_angle[i], robot.actuated_joints_limits[i][0])
                self.assertLessEqual(joint_angle[i], robot.actuated_joints_limits[i][1])

    def _assert_joint_angles_uniform(self, joint_angles: PT_NP_TYPE, robot: Robot):
        """Check that the given joint angles are uniformly distributed within the joint limits."""
        for i in range(joint_angles.shape[1]):
            angles = joint_angles[:, i]
            std = angles.std()
            range_ = angles.max() - angles.min()
            range_gt = robot.actuated_joints_limits[i][1] - robot.actuated_joints_limits[i][0]
            std_expected = np.sqrt((range_gt) ** 2 / 12)  # see https://www.statology.org/uniform-distribution-r/#
            # import matplotlib.pyplot as plt
            # plt.title(f"Joint {i} value distribution")
            # plt.hist(angles)
            # plt.show()
            self.assertAlmostEqual(
                range_,
                range_gt,
                delta=0.125,
                msg=f"Joint {i} range doesn't match expected ({range_} vs {range_gt}",
            )
            self.assertAlmostEqual(std, std_expected, delta=0.15)

    def test_split_configs_to_revolute_and_prismatic(self):
        """Test that split_configs_to_revolute_and_prismatic() returns as expected"""
        # Test 1: Fetch
        fetch = Fetch()
        qs = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9],
                [1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 1.9],
                [2.1, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        expected_revolute = torch.tensor(
            [
                [0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9],
                [1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 1.9],
                [2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        expected_prismatic = torch.tensor(
            [
                [0.1],
                [1.1],
                [2.1],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        revolute, prismatic = fetch.split_configs_to_revolute_and_prismatic(qs)
        torch.testing.assert_close(expected_prismatic, prismatic)
        torch.testing.assert_close(expected_revolute, revolute)

        # Test 2: FetchArm
        fetch = FetchArm()
        qs = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8],
                [1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8],
                [2.1, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        expected_revolute = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8],
                [1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8],
                [2.1, 2.2, 2.3, 2.5, 2.6, 2.7, 2.8],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        expected_prismatic = torch.tensor(
            [
                [],
                [],
                [],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        revolute, prismatic = fetch.split_configs_to_revolute_and_prismatic(qs)
        torch.testing.assert_close(expected_prismatic, prismatic)
        torch.testing.assert_close(expected_revolute, revolute)

    def test_sample_joint_angles(self):
        """_summary_"""
        for robot in ROBOTS:
            joint_angles = robot.sample_joint_angles(1000)
            self.assertEqual(joint_angles.shape, (1000, robot.n_dofs))
            # Check joint angles are within joint limits
            self._assert_joint_angles_within_limits(joint_angles, robot)
            self._assert_joint_angles_uniform(joint_angles, robot)

    def test_sample_joint_angles_and_poses(self):
        """_summary_"""
        print("test_sample_joint_angles_and_poses()")
        for robot in ROBOTS:
            print(robot)
            joint_angles, poses = robot.sample_joint_angles_and_poses(1000, tqdm_enabled=True)
            self.assertEqual(joint_angles.shape, (1000, robot.n_dofs))
            self.assertEqual(poses.shape, (1000, 7))

            # Check joint angles are within joint limits and uniform
            self._assert_joint_angles_within_limits(joint_angles, robot)
            self._assert_joint_angles_uniform(joint_angles, robot)

            # Check FK matches
            fk_joint_angles = robot.forward_kinematics_klampt(joint_angles)
            np.testing.assert_allclose(fk_joint_angles[:, 0:3], poses[:, 0:3], atol=1e-3)

    def test_config_self_collides_panda(self):
        """Test config_self_collides() for Panda. ground truth generated by running:
        './bin/RobotPose /path/to/<robot>/<robot>_formatted.urdf'. 'RobotPose' is from Klamp't (need to build from
        source).
        """
        panda = Panda()
        x = torch.tensor([0, 0, 0, -0.0698, 0, 3.0, 0])
        self.assertFalse(panda.config_self_collides(x, debug_mode=True))

        x = torch.tensor([0.14, -1.7628, 0, -2.9118, -0.2800, 1.0425, 0.0])
        self.assertTrue(panda.config_self_collides(x, debug_mode=True))

        x = torch.tensor([0.14, -1.7628, 0, -2.8718, -0.2800, 1.0425, 0.0])
        self.assertFalse(panda.config_self_collides(x, debug_mode=True))

    def test_config_self_collides_fetch(self):
        """Test config_self_collides() for Fetch"""
        fetch = Fetch()
        x = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        self.assertFalse(fetch.config_self_collides(x, debug_mode=True))

        x = torch.tensor([0, 0, 1.5180, 0, 0, 0, 0, 0], dtype=torch.float32)
        self.assertTrue(fetch.config_self_collides(x, debug_mode=True))

        x = torch.tensor([0, 0, 1.1126, 0, 0, 0, 0, 0], dtype=torch.float32)
        self.assertFalse(fetch.config_self_collides(x, debug_mode=True))

    def test_config_self_collides_iiwa7(self):
        """Test config_self_collides() for Iiwa7"""
        iiwa = Iiwa7()
        x = torch.tensor([0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        self.assertFalse(iiwa.config_self_collides(x, debug_mode=True))

        x = torch.tensor([0, -1.6200, -2.9671, -2.0944, 0.1371, 2.0944, 0], dtype=torch.float32)
        self.assertTrue(iiwa.config_self_collides(x, debug_mode=True))

        x = torch.tensor([0, -1.3, -2.9671, -2.0944, 0.1371, 2.0944, 0], dtype=torch.float32)
        self.assertFalse(iiwa.config_self_collides(x, debug_mode=True))

    def test_joint_limits(self):
        # panda_joint_lims:
        # (-2.8973, 2.8973),
        # (-1.7628, 1.7628),
        # (-2.8973, 2.8973),
        # (-3.0718, -0.0698),
        # (-2.8973, 2.8973),
        # (-0.0175, 3.7525),
        # (-2.8973, 2.8973)

        # Test 1:
        joint_angles_unclamped = torch.tensor(
            [
                [0, 0, 0, -0.1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 0],
            ]
        )
        panda = Panda()
        returned = panda.clamp_to_joint_limits(joint_angles_unclamped)
        expected = torch.tensor(
            [
                [0, 0, 0, -0.1, 0, 0, 0],
                [0, 0, 0, -0.0698, 0, 0, 0],
                [2.8973, 0, 0, -0.0698, 0, 0, 0],
            ]
        )
        torch.testing.assert_close(returned, expected)

    def test_jacobian_np(self):
        for robot in ROBOTS:
            x = np.zeros(robot.n_dofs)
            J = robot.jacobian_np(x)
            self.assertEqual(J.shape, (6, robot.n_dofs))

    def test_jacobian_klampt_pt_match(self):
        atol = 1e-5
        for robot in ROBOTS:
            for i in range(10):
                batch_size = 1
                x = robot.sample_joint_angles(batch_size)
                Jklampt = robot.jacobian_batch_np(x)
                Jpt = robot.jacobian_batch_pt(torch.tensor(x)).cpu().numpy()
                self.assertEqual(Jklampt.shape, (batch_size, 6, robot.n_dofs))
                self.assertEqual(Jpt.shape, (batch_size, 6, robot.n_dofs))
                if not np.allclose(Jklampt, Jpt, atol=atol):
                    print("robot:", robot)
                    print("Jklampt:\n", Jklampt)
                    print("Jpt:\n", Jpt)
                    print("difference:\n", Jklampt - Jpt)
                np.testing.assert_allclose(Jklampt, Jpt, atol=atol)

    def test_list_from_str(self):
        inputs = [
            "0 0 0.333",
            "0 0 1.0",
            "0 0 1",
            "1.5707963267948966   0 3.141592653589793",
        ]
        expected_outputs = [
            (0, 0, 0.333),
            (0, 0, 1.0),
            (0, 0, 1.0),
            (1.5707963267948966, 0, 3.141592653589793),
        ]
        for input_, expected_output in zip(inputs, expected_outputs):
            output = _len3_tuple_from_str(input_)
            self.assertIsInstance(output, tuple)
            self.assertEqual(len(output), 3)
            self.assertAlmostEqual(output[0], expected_output[0])
            self.assertAlmostEqual(output[1], expected_output[1])
            self.assertAlmostEqual(output[2], expected_output[2])

    def test_n_dofs(self):
        ground_truth_n_dofs = {
            "panda": 7,
            "baxter": 7,
            "fetch": 8,
            "fetch_arm": 7,
            "iiwa7": 7,
        }
        for robot in ROBOTS:
            self.assertEqual(robot.n_dofs, ground_truth_n_dofs[robot.name])

    def test_joint_limits(self):
        ground_truth_joint_limits = {
            "fetch": [
                (0, 0.38615),
                (-1.6056, 1.6056),
                (-1.221, 1.518),  # shoulder_lift_joint
                (-np.pi, np.pi),  # upperarm_roll_joint
                (-2.251, 2.251),  # elbow_flex_joint
                (-np.pi, np.pi),  # forearm_roll_joint
                (-2.16, 2.16),  # wrist_flex_joint
                (-np.pi, np.pi),  # wrist_roll_joint
            ],
            "fetch_arm": [
                (-1.6056, 1.6056),
                (-1.221, 1.518),  # shoulder_lift_joint
                (-np.pi, np.pi),  # upperarm_roll_joint
                (-2.251, 2.251),  # elbow_flex_joint
                (-np.pi, np.pi),  # forearm_roll_joint
                (-2.16, 2.16),  # wrist_flex_joint
                (-np.pi, np.pi),  # wrist_roll_joint
            ],
            "panda": [
                (-2.8973, 2.8973),
                (-1.7628, 1.7628),
                (-2.8973, 2.8973),
                (-3.0718, -0.0698),
                (-2.8973, 2.8973),
                (-0.0175, 3.7525),
                (-2.8973, 2.8973),
            ],
            "baxter": [
                (-1.70167993878, 1.70167993878),
                (-2.147, 1.047),
                (-3.05417993878, 3.05417993878),
                (-0.05, 2.618),
                (-3.059, 3.059),
                (-1.57079632679, 2.094),
                (-3.059, 3.059),
            ],
            "iiwa7": [
                (-2.9670597283903604, 2.9670597283903604),
                (-2.0943951023931953, 2.0943951023931953),
                (-2.9670597283903604, 2.9670597283903604),
                (-2.0943951023931953, 2.0943951023931953),
                (-2.9670597283903604, 2.9670597283903604),
                (-2.0943951023931953, 2.0943951023931953),
                (-3.0543261909900763, 3.0543261909900763),
            ],
        }
        for robot in ROBOTS:
            self.assertEqual(len(robot.actuated_joints_limits), robot.n_dofs)
            for gt_limit, parsed_limit in zip(ground_truth_joint_limits[robot.name], robot.actuated_joints_limits):
                self.assertAlmostEqual(gt_limit[0], parsed_limit[0])
                self.assertAlmostEqual(gt_limit[1], parsed_limit[1])

    def test_actuated_joint_names(self):
        print("test_actuated_joint_names()")
        ground_truth_actuated_joints = {
            "panda": [
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ],
            "baxter": [
                "left_s0",
                "left_s1",
                "left_e0",
                "left_e1",
                "left_w0",
                "left_w1",
                "left_w2",
            ],
            "fetch": [
                "torso_lift_joint",
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "upperarm_roll_joint",  # continuous
                "elbow_flex_joint",
                "forearm_roll_joint",  # continuous
                "wrist_flex_joint",
                "wrist_roll_joint",  # continous
            ],
            "fetch_arm": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "upperarm_roll_joint",  # continuous
                "elbow_flex_joint",
                "forearm_roll_joint",  # continuous
                "wrist_flex_joint",
                "wrist_roll_joint",  # continous
            ],
            "iiwa7": [
                "iiwa_joint_1",
                "iiwa_joint_2",
                "iiwa_joint_3",
                "iiwa_joint_4",
                "iiwa_joint_5",
                "iiwa_joint_6",
                "iiwa_joint_7",
            ],
        }
        for robot in ROBOTS:
            self.assertEqual(len(robot.actuated_joint_names), robot.n_dofs)
            self.assertListEqual(robot.actuated_joint_names, ground_truth_actuated_joints[robot.name])

    def test_q_x_conversion(self):
        print("test_q_x_conversion()")
        for robot in ROBOTS:
            print(robot)
            ndofs = robot.n_dofs
            x_original = np.array(
                [
                    [0] * ndofs,
                    list(i * 0.1 for i in range(ndofs)),
                ]
            )
            q = robot._x_to_qs(x_original)
            print("q:", q)
            x_returned = robot._qs_to_x(q)
            print("x_returned:", x_returned)
            np.testing.assert_allclose(x_original, x_returned)

    def test_x_driver_vec_conversion_panda(self):
        print("test_x_driver_vec_conversion_panda()")
        gt_klampt_vector_dimensionality = {
            "fetch": 14,  # 24 total joints, 10 of them are fixed
            "fetch_arm": 14,  # 24 total joints, 10 of them are fixed
            "panda": 8,  # panda has 1 non user specified actuated joint
            "iiwa7": 7,  #
        }

        for robot in ROBOTS:
            gt_vector_dim = gt_klampt_vector_dimensionality[robot.name]
            ndof = robot.n_dofs

            # test 1
            x_original = np.zeros(ndof)
            driver_vec = robot._driver_vec_from_x(x_original)
            self.assertEqual(len(driver_vec), gt_vector_dim)
            x_returned = robot._x_from_driver_vec(driver_vec)
            np.testing.assert_allclose(x_original, x_returned)

            # test 2
            x_original = 0.1 * np.arange(ndof)
            driver_vec = robot._driver_vec_from_x(x_original)
            self.assertEqual(len(driver_vec), gt_vector_dim)  # panda has 1 non user specified actuated joint
            x_returned = robot._x_from_driver_vec(driver_vec)
            np.testing.assert_allclose(x_original, x_returned)

    def test_self_collision(self):
        panda = Panda()
        print("panda._collision_capsules:", panda._collision_capsules)
        print("panda._collision_idx0:", panda._collision_idx0)
        print("panda._collision_idx1:", panda._collision_idx1)
        print("panda._collision_capsules[5]:", panda._collision_capsules[5])
        print("panda._collision_capsules[7]:", panda._collision_capsules[7])
        x = torch.tensor(panda.sample_joint_angles(1), device=DEVICE, dtype=torch.float32)
        x[:] = 0
        dists = panda.self_collision_distances_batch(x)
        print(dists)
        print(dists.device)

    def test_robot_self_collision_distances(self):
        """Test that self_collision_distances_batch() returns the expected distances"""
        atol = 1e-5
        for robot in [Panda()]:
            for i in range(10):
                # batch_size = 5
                batch_size = 1
                x = robot.sample_joint_angles(batch_size)
                x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
                dists = robot.self_collision_distances_batch(x)
                # Check that the distances are correct by comparing to the qpth
                # implementation
                dists_qpth = robot.self_collision_distances_batch(x, use_qpth=True)
                np.testing.assert_allclose(dists.cpu().numpy(), dists_qpth.cpu().numpy(), atol=atol)

    def test_robot_self_collision_distances_jacobian(self):
        """Test that the jacobian of self_collision_distances_batch() is correct"""
        atol = 1e-3
        set_seed(54321)
        for robot in [Panda()]:
            for i in range(10):
                batch_size = 5
                x = robot.sample_joint_angles(batch_size)
                x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
                dists = robot.self_collision_distances_batch(x)
                ndists = dists.shape[1]
                J = robot.self_collision_distances_jacobian_batch(x)
                self.assertEqual(J.shape, (batch_size, ndists, robot.n_dofs))
                # Check that the jacobian is correct by comparing to finite
                # differences
                eps = 1e-3
                J_fd = torch.zeros_like(J)
                for i in range(batch_size):
                    for j in range(robot.n_dofs):
                        x_plus = x.clone()
                        x_plus[i, j] += eps
                        x_minus = x.clone()
                        x_minus[i, j] -= eps
                        dists_plus = robot.self_collision_distances_batch(x_plus)
                        dists_minus = robot.self_collision_distances_batch(x_minus)
                        J_fd[i, :, j] = (dists_plus[i] - dists_minus[i]) / (2 * eps)

                num_larger = torch.sum(torch.abs(J) > 100 * atol)

                np.testing.assert_allclose(J.cpu(), J_fd.cpu(), atol=atol)
                self.assertGreater(num_larger, 0)
                print("Number of elements larger than 100 * atol:", num_larger)


if __name__ == "__main__":
    unittest.main()
